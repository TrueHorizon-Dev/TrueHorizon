//! 稀疏修改差量表（Delta）。
//!
//! 世界是程序化生成的，**只有被玩家改过的地方才需要落盘**。因此存档的体积与
//! 「被挖过的方块数」成正比，而与行星大小无关。
//!
//! 差量的单位是 Chunk：`(Chunk 坐标, 基准版本, 若干 (体素下标, 方块) 对)`。
//! 应用到 `storage::Chunk` 时只改动被记录的位置，避免整块 32³ 的读写放大。

use std::collections::HashMap;

use block_schema::BlockId;
use storage::{Chunk, ChunkPool, StorageError};
use world_contracts::{ChunkCoord, CHUNK_EDGE, CHUNK_VOLUME};

/// 单个体素的修改。
///
/// `index` 是 Chunk 局部线性下标（`0..32768`，恰好能塞进 `u16`）；
/// `block` 是修改后的方块 ID。4 字节/条，一个 Chunk 的完整差量通常 < 1 KiB。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelDelta {
    /// Chunk 局部线性下标。
    pub index: u16,
    /// 修改后的方块。
    pub block: u16,
}

impl VoxelDelta {
    /// 由 Chunk 局部坐标构造。
    pub fn at(x: u32, y: u32, z: u32, block: BlockId) -> Result<Self, StorageError> {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return Err(StorageError::OutOfBounds { x, y, z });
        }
        Ok(Self {
            index: (x + y * CHUNK_EDGE + z * CHUNK_EDGE * CHUNK_EDGE) as u16,
            block: block.raw(),
        })
    }

    /// 由下标还原局部坐标。
    pub fn local_coords(self) -> (u32, u32, u32) {
        let index = self.index as u32;
        (
            index % CHUNK_EDGE,
            (index / CHUNK_EDGE) % CHUNK_EDGE,
            index / (CHUNK_EDGE * CHUNK_EDGE),
        )
    }

    /// 修改后的方块。
    pub fn block(self) -> BlockId {
        BlockId::new(self.block)
    }
}

/// 一个 Chunk 的差量。
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkDelta {
    /// Chunk 坐标。
    pub coord: ChunkCoord,
    /// 差量所基于的 Chunk 版本号。不匹配意味着差量与基准脱节，必须先重新生成基准。
    pub base_version: u32,
    /// 修改列表（按记录顺序，同一位置重复修改会折叠为最后一次）。
    pub edits: Vec<VoxelDelta>,
}

impl ChunkDelta {
    /// 新建空差量。
    pub fn new(coord: ChunkCoord, base_version: u32) -> Self {
        Self {
            coord,
            base_version,
            edits: Vec::new(),
        }
    }

    /// 记录一次修改（同位置覆盖）。
    pub fn record(&mut self, x: u32, y: u32, z: u32, block: BlockId) -> Result<(), StorageError> {
        let delta = VoxelDelta::at(x, y, z, block)?;
        match self.edits.iter_mut().find(|e| e.index == delta.index) {
            // `edits` 通常只有几十项（局部挖掘），线性查找比哈希表更快且无分配。
            Some(existing) => *existing = delta,
            None => self.edits.push(delta),
        }
        Ok(())
    }

    /// 修改条数。
    #[inline]
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// 是否没有修改。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// 落盘字节数。
    #[inline]
    pub fn bytes(&self) -> usize {
        self.edits.len() * core::mem::size_of::<VoxelDelta>()
    }

    /// 应用到 Chunk，返回**实际生效**的修改数。
    pub fn apply(&self, chunk: &mut Chunk) -> Result<u32, StorageError> {
        let mut applied = 0;
        for delta in &self.edits {
            let (x, y, z) = delta.local_coords();
            if chunk.set(x, y, z, delta.block())? {
                applied += 1;
            }
        }
        Ok(applied)
    }
}

/// 待落盘差量的集合。
#[derive(Debug, Default)]
pub struct DeltaTable {
    entries: HashMap<ChunkCoord, ChunkDelta>,
}

impl DeltaTable {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一次体素修改。
    pub fn record(
        &mut self,
        coord: ChunkCoord,
        x: u32,
        y: u32,
        z: u32,
        block: BlockId,
        base_version: u32,
    ) -> Result<(), StorageError> {
        self.entries
            .entry(coord)
            .or_insert_with(|| ChunkDelta::new(coord, base_version))
            .record(x, y, z, block)
    }

    /// 某 Chunk 的差量。
    pub fn get(&self, coord: ChunkCoord) -> Option<&ChunkDelta> {
        self.entries.get(&coord)
    }

    /// 涉及的 Chunk 数。
    #[inline]
    pub fn pending_chunks(&self) -> usize {
        self.entries.len()
    }

    /// 是否有待落盘内容。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 修改总条数。
    pub fn total_edits(&self) -> usize {
        self.entries.values().map(ChunkDelta::len).sum()
    }

    /// 落盘字节数估算。
    pub fn bytes(&self) -> usize {
        self.entries.values().map(ChunkDelta::bytes).sum()
    }

    /// 取出全部差量并清空（交予 WAL 写线程后即可继续累积新差量）。
    pub fn drain(&mut self) -> Vec<ChunkDelta> {
        let mut drained: Vec<ChunkDelta> = self.entries.drain().map(|(_, delta)| delta).collect();
        drained.sort_by_key(|delta| delta.coord.key());
        drained
    }

    /// 把全部差量应用到 Chunk 池（差量所指的 Chunk 必须已在池中）。
    pub fn apply_all(&self, pool: &mut ChunkPool) -> Result<u32, StorageError> {
        let mut applied = 0;
        for (coord, delta) in &self.entries {
            let Some(chunk) = pool.get_mut(*coord) else {
                // 目标 Chunk 不在池中：跳过而不是报错——它可能是被换出的远处区块，
                // 下次加载时会重新生成再重放 WAL。
                continue;
            };
            if chunk.version() != delta.base_version {
                // 版本不符说明基准已变，差量不再可信；交由上层决定重建策略。
                continue;
            }
            applied += delta.apply(chunk)?;
        }
        Ok(applied)
    }
}

/// 单块 Chunk 的最大差量条数（超过则建议整块重写而不是继续追加差量）。
pub const MAX_DELTA_EDITS: usize = CHUNK_VOLUME / 8;

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_chunk(coord: ChunkCoord) -> Chunk {
        Chunk::new_empty(coord)
    }

    #[test]
    fn voxel_delta_coordinate_round_trip() {
        for (x, y, z) in [(0, 0, 0), (1, 2, 3), (31, 31, 31), (0, 31, 7)] {
            let delta = VoxelDelta::at(x, y, z, BlockId::new(5)).unwrap();
            assert_eq!(delta.local_coords(), (x, y, z));
            assert_eq!(delta.block(), BlockId::new(5));
        }
        assert_eq!(
            VoxelDelta::at(32, 0, 0, BlockId::new(1)),
            Err(StorageError::OutOfBounds { x: 32, y: 0, z: 0 })
        );
    }

    #[test]
    fn repeated_edits_at_same_voxel_collapse() {
        let mut delta = ChunkDelta::new(ChunkCoord::default(), 0);
        delta.record(1, 1, 1, BlockId::new(1)).unwrap();
        delta.record(1, 1, 1, BlockId::new(2)).unwrap();
        delta.record(1, 1, 1, BlockId::new(3)).unwrap();
        assert_eq!(delta.len(), 1, "同位置必须折叠为最后一次");
        assert_eq!(delta.edits[0].block(), BlockId::new(3));
    }

    #[test]
    fn apply_returns_effective_change_count() {
        let coord = ChunkCoord::new(2, 0, -1);
        let mut chunk = empty_chunk(coord);
        chunk.set(5, 5, 5, BlockId::new(9)).unwrap();
        chunk.mark_clean();
        let base_version = chunk.version();

        let mut delta = ChunkDelta::new(coord, base_version);
        delta.record(5, 5, 5, BlockId::new(9)).unwrap(); // 无变化
        delta.record(6, 5, 5, BlockId::new(9)).unwrap(); // 有效
        let applied = delta.apply(&mut chunk).unwrap();
        assert_eq!(applied, 1);
        assert_eq!(chunk.get(6, 5, 5), BlockId::new(9));
        assert!(chunk.is_dirty(), "重放差量必须重新置脏，否则会漏落盘");
    }

    #[test]
    fn table_groups_by_chunk_and_stale_base_is_skipped() {
        let mut table = DeltaTable::new();
        let coord = ChunkCoord::new(0, 0, 0);
        table.record(coord, 1, 0, 0, BlockId::new(1), 0).unwrap();
        table.record(coord, 2, 0, 0, BlockId::new(1), 0).unwrap();
        table
            .record(ChunkCoord::new(1, 0, 0), 0, 0, 0, BlockId::new(1), 0)
            .unwrap();
        assert_eq!(table.pending_chunks(), 2);
        assert_eq!(table.total_edits(), 3);
        assert_eq!(table.bytes(), 3 * 4);

        // 两个 Chunk 的版本号都不等于差量的 base_version（0）：
        // 必须整体跳过，而不是把差量写到已经不匹配的基准上。
        let mut modified_first = empty_chunk(coord);
        modified_first.set(0, 0, 0, BlockId::new(9)).unwrap();
        let mut modified_second = empty_chunk(ChunkCoord::new(1, 0, 0));
        modified_second.set(0, 0, 0, BlockId::new(9)).unwrap();
        let mut pool = ChunkPool::new(4);
        pool.insert(modified_first).unwrap();
        pool.insert(modified_second).unwrap();
        assert_eq!(
            table.apply_all(&mut pool).unwrap(),
            0,
            "基准版本不符必须整体跳过"
        );

        let drained = table.drain();
        assert_eq!(drained.len(), 2);
        assert!(table.is_empty());
        assert_eq!(drained[0].coord.key(), [0, 0, 0]);
    }

    #[test]
    fn delta_bytes_are_tiny_compared_to_full_chunk() {
        let mut delta = ChunkDelta::new(ChunkCoord::default(), 0);
        for i in 0..64u32 {
            // 32³ 的局部坐标必须落在 Chunk 之内。
            delta.record(i % 32, i / 32, 0, BlockId::new(1)).unwrap();
        }
        assert!(delta.bytes() < MAX_DELTA_EDITS);
        assert!(delta.bytes() * 100 < CHUNK_VOLUME * 2);
    }
}
