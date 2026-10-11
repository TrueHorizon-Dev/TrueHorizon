//! 整块 Chunk 快照：`rkyv` 0.8 归档格式（可零拷贝读取）。
//!
//! 与差量（[`crate::delta`]）互补：
//!
//! * 差量回答「玩家改了什么」——体积与被挖过的方块数成正比；
//! * 快照回答「这一块现在是什么」——用于 WAL 压实（把上万条差量折叠成少量整块）、
//!   离线烘焙分发（`world-tool bake`）与崩溃后的快速对账。
//!
//! ## 为什么用 rkyv 而不是 serde
//!
//! 快照是**读多写少**的数据：加载存档时希望直接对着 mmap/缓冲里的字节读数，
//! 而不是先构造 65536 个 `u16` 的中间结构。rkyv 的归档是「就地可用」的：
//! `access` 校验一遍字节（防越界与伪造指针），之后 `archived.blocks` 就是
//! 归档缓冲里的切片，没有任何拷贝与反序列化。
//!
//! ## 对齐约束（必须遵守）
//!
//! rkyv 默认（`aligned`）要求读取缓冲按归档类型对齐——**未对齐的 `access` 是未定义行为**，
//! 且这条约束只在 debug 构建里由断言兜底。因此本模块只提供两条路径：
//!
//! * [`ChunkSnapshot::access`] / [`ChunkSnapshot::blocks_zero_copy`]：只接受
//!   [`AlignedBytes`]（`rkyv::util::AlignedVec`，16 字节对齐），零拷贝；
//! * [`ChunkSnapshot::from_archive`]：接受任意 `&[u8]`（例如刚从 WAL 帧里读出的 `Vec<u8>`），
//!   内部**先复制进对齐缓冲**再校验、再反序列化成拥有所有权的值。
//!
//! 文件读写走 [`ChunkSnapshot::write_file`] / [`ChunkSnapshot::read_file`]：
//! 读入 [`AlignedBytes`] 后即可对文件内容零拷贝访问，不需要先反序列化。

use std::fs::File;
use std::io::Read;
use std::path::Path;

use block_schema::BlockId;
use rkyv::rancor;
use rkyv::rend::u16_le;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use storage::{Chunk, StorageError};
use world_contracts::{ChunkCoord, CHUNK_EDGE, CHUNK_VOLUME};

use crate::PersistError;

/// 归档缓冲类型：16 字节对齐，满足 rkyv 的读取前提。
pub type AlignedBytes = AlignedVec;

/// 归档里方块数组的**零拷贝只读视图**。
///
/// rkyv 的元素类型是显式小端包装（`u16_le`），因此这里不暴露裸 `&[u16]`，
/// 而是给出「按下标/迭代取原生 `u16`」的视图：读取过程一次拷贝都没有，
/// 需要独立数组时显式调用 [`Self::to_vec`]。
#[derive(Clone, Copy, Debug)]
pub struct ArchivedBlocks<'a> {
    inner: &'a [u16_le],
}

impl<'a> ArchivedBlocks<'a> {
    /// 体素数。
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 是否为空（只有被截断的归档才为空）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// 按下标取原生 `u16`。
    #[inline]
    pub fn get(&self, index: usize) -> Option<u16> {
        self.inner.get(index).map(|raw| raw.to_native())
    }

    /// 迭代原生 `u16`。
    pub fn iter(&self) -> impl Iterator<Item = u16> + 'a {
        self.inner.iter().map(|raw| raw.to_native())
    }

    /// 原始归档切片（需要直接比对字节时使用）。
    #[inline]
    pub fn raw(&self) -> &'a [u16_le] {
        self.inner
    }

    /// 复制成原生数组（唯一会分配内存的出口）。
    pub fn to_vec(&self) -> Vec<u16> {
        self.iter().collect()
    }
}

/// 一个 Chunk 的完整快照（32³ 体素，稠密存储）。
///
/// 稠密而非调色板：归档本身就是「就地可读的字节」，再叠一层调色板压缩会让读取路径
/// 重新引入解压步骤；体积敏感的场景用 WAL 差量（4 字节/条）而不是快照。
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct ChunkSnapshot {
    /// Chunk 坐标（`[x, y, z]`）。
    pub coord: [i64; 3],
    /// 快照对应的 Chunk 版本号，用于与差量的 `base_version` 对齐。
    pub version: u32,
    /// 稠密方块数组，下标顺序与 `VoxelDelta::index` 一致（`x + y*32 + z*1024`）。
    pub blocks: Vec<u16>,
}

impl ChunkSnapshot {
    /// 一个 Chunk 的体素总数。
    pub const BLOCK_COUNT: usize = CHUNK_VOLUME;

    /// 从 Chunk 提取快照（读全部体素，不修改源 Chunk）。
    pub fn from_chunk(chunk: &Chunk) -> Self {
        let mut blocks = Vec::with_capacity(Self::BLOCK_COUNT);
        for z in 0..CHUNK_EDGE {
            for y in 0..CHUNK_EDGE {
                for x in 0..CHUNK_EDGE {
                    blocks.push(chunk.get(x, y, z).raw());
                }
            }
        }
        Self {
            coord: chunk.coord().key(),
            version: chunk.version(),
            blocks,
        }
    }

    /// Chunk 坐标。
    pub fn coord(&self) -> ChunkCoord {
        ChunkCoord::new(self.coord[0], self.coord[1], self.coord[2])
    }

    /// 某个局部坐标的方块（越界返回 [`BlockId::AIR`]）。
    pub fn block_at(&self, x: u32, y: u32, z: u32) -> BlockId {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return BlockId::AIR;
        }
        let index = (x + y * CHUNK_EDGE + z * CHUNK_EDGE * CHUNK_EDGE) as usize;
        self.blocks
            .get(index)
            .map(|raw| BlockId::new(*raw))
            .unwrap_or(BlockId::AIR)
    }

    /// 重建 Chunk（快照不完整时报错，而不是产出半截世界）。
    ///
    /// 建出来的 Chunk 会带上快照记录的版本号并标记为**干净**——快照本身就是落盘状态，
    /// 版本号必须一致，否则随后的差量会因基准不匹配而被跳过。
    pub fn to_chunk(&self) -> Result<Chunk, StorageError> {
        if !self.is_complete() {
            return Err(StorageError::OutOfBounds {
                x: 0,
                y: 0,
                z: self.blocks.len() as u32,
            });
        }
        let voxels: Vec<BlockId> = self.blocks.iter().map(|raw| BlockId::new(*raw)).collect();
        let mut chunk = Chunk::from_voxels(self.coord(), &voxels)?;
        chunk.restore_version(self.version);
        chunk.mark_clean();
        Ok(chunk)
    }

    /// 体素数是否正好等于一个 Chunk。
    pub fn is_complete(&self) -> bool {
        self.blocks.len() == Self::BLOCK_COUNT
    }

    /// 序列化为 rkyv 归档（对齐缓冲，可直接 [`Self::access`]）。
    pub fn archive(&self) -> Result<AlignedBytes, PersistError> {
        rkyv::to_bytes::<rancor::Error>(self)
            .map_err(|error| PersistError::Malformed(format!("rkyv 序列化失败: {error}")))
    }

    /// 校验并零拷贝访问归档。
    ///
    /// **只接受对齐缓冲**（[`AlignedBytes`]）；任意字节请走 [`Self::from_archive`]。
    pub fn access(bytes: &AlignedBytes) -> Result<&ArchivedChunkSnapshot, PersistError> {
        rkyv::access::<ArchivedChunkSnapshot, rancor::Error>(bytes.as_slice())
            .map_err(|error| PersistError::Malformed(format!("rkyv 归档校验失败: {error}")))
    }

    /// 目录：读取归档里的稠密方块数组（零拷贝，切片直接指向归档缓冲）。
    pub fn blocks_zero_copy(bytes: &AlignedBytes) -> Result<ArchivedBlocks<'_>, PersistError> {
        Ok(ArchivedBlocks {
            inner: Self::access(bytes)?.blocks.as_slice(),
        })
    }

    /// 从任意字节反序列化（未对齐输入会被复制进对齐缓冲）。
    pub fn from_archive(bytes: &[u8]) -> Result<Self, PersistError> {
        let aligned = if is_aligned_for::<ArchivedChunkSnapshot>(bytes) {
            None
        } else {
            let mut buffer = AlignedBytes::with_capacity(bytes.len());
            buffer.extend_from_slice(bytes);
            Some(buffer)
        };
        let slice: &[u8] = match &aligned {
            Some(buffer) => buffer.as_slice(),
            None => bytes,
        };
        rkyv::from_bytes::<Self, rancor::Error>(slice)
            .map_err(|error| PersistError::Malformed(format!("rkyv 反序列化失败: {error}")))
    }

    /// 写入文件（内容就是归档字节）。
    pub fn write_file(&self, path: &Path) -> Result<usize, PersistError> {
        let bytes = self.archive()?;
        std::fs::write(path, bytes.as_slice())?;
        Ok(bytes.len())
    }

    /// 读入文件并对齐缓冲返回（内容本身不做反序列化，便于零拷贝访问）。
    pub fn read_file(path: &Path) -> Result<AlignedBytes, PersistError> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len() as usize;
        let mut bytes = AlignedBytes::with_capacity(length);
        bytes.resize(length, 0);
        file.read_exact(bytes.as_mut_slice())?;
        Ok(bytes)
    }
}

/// 指针是否满足归档类型的对齐要求。
fn is_aligned_for<T>(bytes: &[u8]) -> bool {
    let alignment = core::mem::align_of::<T>();
    (bytes.as_ptr() as usize) % alignment == 0
}

/// 归档根的位置（rkyv 把根放在缓冲末尾，长度即为位置依据）。
///
/// 这里只用于测试与诊断输出，避免上层为了算「快照有多大」而去读内部布局。
pub fn archive_root_size<T: rkyv::Portable>() -> usize {
    core::mem::size_of::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> Chunk {
        let coord = ChunkCoord::new(-2, 5, 7);
        let mut chunk = Chunk::new_empty(coord);
        chunk.set(0, 0, 0, BlockId::new(1)).unwrap();
        chunk.set(31, 0, 0, BlockId::new(2)).unwrap();
        chunk.set(0, 31, 0, BlockId::new(3)).unwrap();
        chunk.set(0, 0, 31, BlockId::new(4)).unwrap();
        chunk.set(31, 31, 31, BlockId::new(5)).unwrap();
        chunk.mark_clean();
        chunk
    }

    #[test]
    fn snapshot_mirrors_the_source_chunk() {
        let chunk = sample_chunk();
        let snapshot = ChunkSnapshot::from_chunk(&chunk);
        assert!(snapshot.is_complete());
        assert_eq!(snapshot.coord(), chunk.coord());
        assert_eq!(snapshot.version, chunk.version());
        assert_eq!(snapshot.block_at(0, 0, 0), BlockId::new(1));
        assert_eq!(snapshot.block_at(31, 31, 31), BlockId::new(5));
        assert_eq!(snapshot.block_at(1, 1, 1), BlockId::AIR);
        // 越界读取必须是 AIR，而不是 panic。
        assert_eq!(snapshot.block_at(32, 0, 0), BlockId::AIR);

        let rebuilt = snapshot.to_chunk().unwrap();
        assert_eq!(rebuilt.coord(), chunk.coord());
        assert_eq!(rebuilt.non_air_count(), chunk.non_air_count());
        assert_eq!(rebuilt.version(), chunk.version());
        assert!(!rebuilt.is_dirty(), "快照重建出来的是落盘状态，不应被标脏");
    }

    #[test]
    fn archive_round_trip_keeps_every_voxel() {
        let chunk = sample_chunk();
        let snapshot = ChunkSnapshot::from_chunk(&chunk);
        let bytes = snapshot.archive().unwrap();
        let restored = ChunkSnapshot::from_archive(bytes.as_slice()).unwrap();
        assert_eq!(restored, snapshot);

        // 重新读出来的 Chunk 必须逐体素一致。
        let rebuilt = restored.to_chunk().unwrap();
        for z in 0..CHUNK_EDGE {
            for y in 0..CHUNK_EDGE {
                for x in 0..CHUNK_EDGE {
                    assert_eq!(rebuilt.get(x, y, z), chunk.get(x, y, z));
                }
            }
        }
    }

    #[test]
    fn zero_copy_read_points_into_the_archive_buffer() {
        let snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        let bytes = snapshot.archive().unwrap();
        let blocks = ChunkSnapshot::blocks_zero_copy(&bytes).unwrap();
        assert_eq!(blocks.len(), ChunkSnapshot::BLOCK_COUNT);
        assert_eq!(blocks.to_vec(), snapshot.blocks);
        assert_eq!(blocks.get(0), snapshot.blocks.first().copied());
        assert_eq!(blocks.get(usize::MAX), None);

        // 零拷贝的硬证据：视图落在归档缓冲的地址区间内。
        let start = bytes.as_ptr() as usize;
        let end = start + bytes.len();
        let slice_start = blocks.raw().as_ptr() as usize;
        assert!(
            slice_start >= start && slice_start < end,
            "方块数组必须直接指向归档缓冲，而不是被复制出来"
        );
    }

    #[test]
    fn corrupted_archives_are_rejected_not_panicking() {
        let snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        let bytes = snapshot.archive().unwrap();

        // 1) 整体截断：根位置算不出来。
        let truncated = &bytes.as_slice()[..bytes.len() / 2];
        assert!(matches!(
            ChunkSnapshot::from_archive(truncated),
            Err(PersistError::Malformed(_))
        ));
        // 2) 空缓冲。
        assert!(matches!(
            ChunkSnapshot::from_archive(&[]),
            Err(PersistError::Malformed(_))
        ));
        // 3) 长度正确但内容被改坏：校验必须发现（而不是读出越界切片）。
        let mut garbage = bytes.as_slice().to_vec();
        let length = garbage.len();
        for byte in garbage.iter_mut().skip(length / 3).take(64) {
            *byte = 0xFF;
        }
        // 任意字节走 from_archive（内部会对齐化），结果只能是 Err 或合法值，绝不能 panic。
        let _ = ChunkSnapshot::from_archive(&garbage);
    }

    #[test]
    fn identical_input_produces_identical_bytes() {
        let snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        let first = snapshot.archive().unwrap();
        let second = snapshot.archive().unwrap();
        assert_eq!(first.as_slice(), second.as_slice(), "归档必须可复现");
    }

    #[test]
    fn archive_has_no_padding_blow_up() {
        let snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        let bytes = snapshot.archive().unwrap();
        let naive = ChunkSnapshot::BLOCK_COUNT * core::mem::size_of::<u16>();
        assert_eq!(naive, 64 * 1024);
        assert!(
            bytes.len() <= naive + 64,
            "归档 {} 字节不应显著大于朴素数组 {naive} 字节",
            bytes.len()
        );
        // 归档 = 稠密体素 + 少量头部/根，压缩来自差量而不是快照，这里不做过头承诺。
        assert!(bytes.len() >= naive);
    }

    #[test]
    fn file_round_trip_supports_zero_copy_access() {
        let directory = std::env::temp_dir().join("truehorizon-snapshot-test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("chunk.snapshot");
        let snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        let written = snapshot.write_file(&path).unwrap();
        assert!(written > 0);

        let bytes = ChunkSnapshot::read_file(&path).unwrap();
        assert_eq!(bytes.len(), written);
        assert_eq!(
            ChunkSnapshot::blocks_zero_copy(&bytes).unwrap().to_vec(),
            snapshot.blocks
        );
        assert_eq!(
            ChunkSnapshot::from_archive(bytes.as_slice()).unwrap(),
            snapshot
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn incomplete_snapshot_cannot_become_a_chunk() {
        let mut snapshot = ChunkSnapshot::from_chunk(&sample_chunk());
        snapshot.blocks.truncate(16);
        assert!(!snapshot.is_complete());
        assert!(snapshot.to_chunk().is_err());
        assert!(archive_root_size::<ArchivedChunkSnapshot>() > 0);
    }
}
