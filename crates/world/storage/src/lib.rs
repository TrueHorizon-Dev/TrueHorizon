//! # storage —— 体素存储
//!
//! 世界状态的**唯一权威容器**：所有体素数据都以 32³ [`Chunk`] 为单位驻留或换出。
//!
//! * [`morton`]：Z-order 空间索引（局部 15 位键 / 宏观 63 位键）。
//! * [`chunk`]：调色板压缩的 Chunk 与占用位图。
//! * [`ChunkPool`]：由 `foundation::Arena` 支撑的 Chunk 池（句柄稳定、回收即失效）。
//! * [`MemoryWorld`]：实现 `world-contracts::WorldView` 的内存世界，供网格化、
//!   物理与测试直接消费。
//!
//! 并发模型：`ChunkPool` 是**单线程**的写侧容器（世界 tick 独占）；
//! 只读阶段通过 `MemoryWorld` 的不可变借用跨线程共享，配合 `rayon` 做并行网格化。
//! 无锁多读者版本（分片 + 句柄租约）列在路线图上。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod chunk;
pub mod morton;

use std::collections::HashMap;

use block_schema::BlockId;
use foundation::{Arena, Handle};
use planet_coords::VoxelCoord;
use world_contracts::{ChunkCoord, WorldView, CHUNK_EDGE};

pub use chunk::{Chunk, StorageError};
pub use morton::{decode, encode};

/// Chunk 池：句柄稳定的内存容器。
///
/// * 容量在构造时确定（内存预算必须显式化，禁止无界增长）；
/// * `Arena` 提供 O(1) 插入/删除与世代校验，Chunk 换出后旧句柄自动失效；
/// * 额外维护 `coord → Handle` 索引，使「按 Chunk 坐标取数据」无需遍历。
#[derive(Debug)]
pub struct ChunkPool {
    arena: Arena<Chunk>,
    by_coord: HashMap<ChunkCoord, Handle<Chunk>>,
    capacity: usize,
}

impl ChunkPool {
    /// 以容量构造。
    pub fn new(capacity: usize) -> Self {
        Self {
            arena: Arena::with_capacity(capacity),
            by_coord: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// 容量上限。
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 当前驻留的 Chunk 数。
    #[inline]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// 是否已满。
    #[inline]
    pub fn is_full(&self) -> bool {
        self.arena.len() >= self.capacity
    }

    /// 插入 Chunk（同坐标的旧 Chunk 会先被换出）。
    pub fn insert(&mut self, chunk: Chunk) -> Result<Handle<Chunk>, StorageError> {
        if self.is_full() && !self.by_coord.contains_key(&chunk.coord()) {
            return Err(StorageError::PoolFull {
                capacity: self.capacity,
            });
        }
        let coord = chunk.coord();
        if let Some(old) = self.by_coord.remove(&coord) {
            self.arena.remove(old);
        }
        let handle = self.arena.insert(chunk);
        self.by_coord.insert(coord, handle);
        Ok(handle)
    }

    /// 按坐标取只读引用。
    pub fn get(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.by_coord
            .get(&coord)
            .and_then(|handle| self.arena.get(*handle))
    }

    /// 按坐标取可变引用。
    pub fn get_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        let handle = *self.by_coord.get(&coord)?;
        self.arena.get_mut(handle)
    }

    /// 按句柄取只读引用。
    pub fn get_by_handle(&self, handle: Handle<Chunk>) -> Option<&Chunk> {
        self.arena.get(handle)
    }

    /// 按坐标取句柄。
    pub fn handle(&self, coord: ChunkCoord) -> Option<Handle<Chunk>> {
        self.by_coord.get(&coord).copied()
    }

    /// 换出并返回 Chunk。
    pub fn remove(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        let handle = self.by_coord.remove(&coord)?;
        self.arena.remove(handle)
    }

    /// 遍历全部驻留 Chunk。
    pub fn iter(&self) -> impl Iterator<Item = (Handle<Chunk>, &Chunk)> + '_ {
        self.arena.iter()
    }

    /// 收集所有脏 Chunk 的坐标（交予 `persistence` 落盘后调用 `mark_clean`）。
    pub fn dirty_coords(&self) -> Vec<ChunkCoord> {
        self.arena
            .iter()
            .filter(|(_, chunk)| chunk.is_dirty())
            .map(|(_, chunk)| chunk.coord())
            .collect()
    }

    /// 内存占用估算（按压缩后的理论字节数计）。
    pub fn resident_bytes(&self) -> usize {
        self.arena
            .iter()
            .map(|(_, chunk)| chunk.packed_bytes())
            .sum()
    }
}

/// 内存世界：Chunk 池 + 只读视图。
///
/// 这是世界 tick 与渲染/物理之间的**交接面**：写侧持有 `&mut ChunkPool`，
/// 只读侧拿到 `&MemoryWorld` 后即可并行网格化、射线拾取或做权威校验。
#[derive(Debug)]
pub struct MemoryWorld {
    pool: ChunkPool,
    radius_m: f64,
}

impl MemoryWorld {
    /// 以 Chunk 容量与行星半径构造。
    pub fn new(capacity: usize, radius_m: f64) -> Self {
        Self {
            pool: ChunkPool::new(capacity),
            radius_m,
        }
    }

    /// Chunk 池（写侧入口）。
    #[inline]
    pub fn pool_mut(&mut self) -> &mut ChunkPool {
        &mut self.pool
    }

    /// Chunk 池（只读）。
    #[inline]
    pub fn pool(&self) -> &ChunkPool {
        &self.pool
    }

    /// 行星半径（米）。
    #[inline]
    pub fn radius_m(&self) -> f64 {
        self.radius_m
    }

    /// 驻留 Chunk 数。
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.pool.len()
    }

    /// 把全局体素坐标拆成 `(Chunk 坐标, Chunk 内局部坐标)`。
    fn split(voxel: VoxelCoord) -> (ChunkCoord, u32, u32, u32) {
        let edge = CHUNK_EDGE as i64;
        let local = |value: i64| value.rem_euclid(edge) as u32;
        (
            ChunkCoord::from_voxel(voxel),
            local(voxel.x),
            local(voxel.y),
            local(voxel.z),
        )
    }
}

impl WorldView for MemoryWorld {
    fn block_at(&self, voxel: VoxelCoord) -> BlockId {
        let (coord, x, y, z) = Self::split(voxel);
        match self.pool.get(coord) {
            Some(chunk) => chunk.get(x, y, z),
            // 未加载区域一律视为空气：流式加载边界是常态，不是异常。
            None => BlockId::AIR,
        }
    }

    fn has_chunk(&self, coord: ChunkCoord) -> bool {
        self.pool.get(coord).is_some()
    }

    fn surface_height_m(&self, _dir: glam::DVec3) -> Option<f64> {
        // 内存世界不保留地形场（那是 `geology` 的职责）。返回 `None` 而非编造数值，
        // 迫使调用方走 `StratumSampler`，避免出现两套互相矛盾的高度语义。
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_at(x: i64, y: i64, z: i64, block: BlockId) -> Chunk {
        let coord = ChunkCoord::new(x, y, z);
        let mut chunk = Chunk::new_empty(coord);
        chunk.set(1, 2, 3, block).unwrap();
        chunk
    }

    #[test]
    fn pool_tracks_len_and_capacity() {
        let mut pool = ChunkPool::new(2);
        let handle = pool.insert(chunk_at(0, 0, 0, BlockId::new(1))).unwrap();
        assert!(!pool.is_full());
        pool.insert(chunk_at(1, 0, 0, BlockId::new(1))).unwrap();
        assert!(pool.is_full());
        assert_eq!(
            pool.insert(chunk_at(2, 0, 0, BlockId::new(1))),
            Err(StorageError::PoolFull { capacity: 2 })
        );
        assert!(pool.get_by_handle(handle).is_some());
        assert!(pool.resident_bytes() > 0);
    }

    #[test]
    fn replacing_same_coord_evicts_old_chunk() {
        let mut pool = ChunkPool::new(4);
        let old = pool.insert(chunk_at(0, 0, 0, BlockId::new(1))).unwrap();
        pool.insert(chunk_at(0, 0, 0, BlockId::new(2))).unwrap();
        assert_eq!(pool.len(), 1, "同坐标不得重复驻留");
        assert!(pool.get_by_handle(old).is_none(), "换出的旧句柄必须失效");
        assert_eq!(
            pool.get(ChunkCoord::default()).unwrap().get(1, 2, 3),
            BlockId::new(2)
        );
    }

    #[test]
    fn remove_returns_chunk_and_frees_capacity() {
        let mut pool = ChunkPool::new(1);
        pool.insert(chunk_at(0, 0, 0, BlockId::new(1))).unwrap();
        let chunk = pool.remove(ChunkCoord::default()).expect("应当取回 Chunk");
        assert_eq!(chunk.get(1, 2, 3), BlockId::new(1));
        assert!(pool.is_empty());
        assert!(
            pool.insert(chunk_at(5, 0, 0, BlockId::new(3))).is_ok(),
            "容量应被释放"
        );
    }

    #[test]
    fn dirty_coords_reflect_writes() {
        let mut pool = ChunkPool::new(2);
        pool.insert(chunk_at(0, 0, 0, BlockId::new(1))).unwrap();
        pool.insert(Chunk::new_empty(ChunkCoord::new(9, 9, 9)))
            .unwrap();
        let dirty = pool.dirty_coords();
        assert_eq!(dirty, vec![ChunkCoord::new(0, 0, 0)]);
        pool.get_mut(ChunkCoord::new(0, 0, 0)).unwrap().mark_clean();
        assert!(pool.dirty_coords().is_empty());
    }

    #[test]
    fn memory_world_reads_voxels_across_chunk_boundaries() {
        let mut world = MemoryWorld::new(8, 6_371_000.0);
        world
            .pool_mut()
            .insert(chunk_at(-1, 0, 0, BlockId::new(7)))
            .unwrap();

        // Chunk (-1, 0, 0) 的最小角是体素 (-32, 0, 0)，局部 (1, 2, 3) → 全局 (-31, 2, 3)。
        assert_eq!(world.block_at(VoxelCoord::new(-31, 2, 3)), BlockId::new(7));
        assert!(world.is_solid(VoxelCoord::new(-31, 2, 3)));
        assert_eq!(world.block_at(VoxelCoord::new(-30, 2, 3)), BlockId::AIR);
        assert_eq!(world.chunk_count(), 1);
        assert!(world.has_chunk(ChunkCoord::new(-1, 0, 0)));
        assert!(!world.has_chunk(ChunkCoord::new(0, 0, 0)));
        assert_eq!(world.surface_height_m(glam::DVec3::X), None);
    }
}
