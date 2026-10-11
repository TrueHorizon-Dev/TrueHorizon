//! 32³ Chunk 内存表示：调色板压缩 + 占用位图。

use bitvec::prelude::*;
use block_schema::BlockId;
use world_contracts::{ChunkCoord, CHUNK_EDGE, CHUNK_VOLUME};

/// 存储层错误。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// 体素坐标越出 Chunk 边界。
    OutOfBounds {
        /// X 坐标。
        x: u32,
        /// Y 坐标。
        y: u32,
        /// Z 坐标。
        z: u32,
    },
    /// 输入体素缓冲长度与 `CHUNK_VOLUME` 不符。
    VolumeSizeMismatch {
        /// 期望长度。
        expected: usize,
        /// 实际长度。
        actual: usize,
    },
    /// 调色板项数超出索引位宽可表达的范围。
    PaletteOverflow {
        /// 当前调色板长度。
        len: usize,
    },
    /// Chunk 池已满。
    PoolFull {
        /// 池容量。
        capacity: usize,
    },
}

impl core::fmt::Display for StorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBounds { x, y, z } => write!(f, "体素坐标越界: ({x}, {y}, {z})"),
            Self::VolumeSizeMismatch { expected, actual } => {
                write!(f, "体素缓冲长度不匹配：期望 {expected}，实际 {actual}")
            },
            Self::PaletteOverflow { len } => write!(f, "调色板超出位宽上限: {len} 项"),
            Self::PoolFull { capacity } => write!(f, "Chunk 池已满（容量 {capacity}）"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Chunk 局部体素线性索引。
#[inline]
fn local_index(x: u32, y: u32, z: u32) -> usize {
    (x + y * CHUNK_EDGE + z * CHUNK_EDGE * CHUNK_EDGE) as usize
}

/// 一个 32³ 的体素区块。
///
/// 布局：`palette`（去重后的方块，第 0 项恒为空气）+ `indices`（每体素一个 `u16` 调色板下标）
/// + `occupancy` 位图（1 = 非空气，1 位/体素 = 4 KiB/Chunk）。
///
/// * 地形以「少数几种方块铺满大片空间」为主，调色板通常只有个位数项，
///   因此 [`Chunk::index_bits`] 报出的位宽远小于 16，落到磁盘时应使用该位宽打包；
/// * 占用位图让「空中 Chunk 快速跳过」与「占用统计」都变成位运算，无需遍历调色板。
#[derive(Clone, Debug)]
pub struct Chunk {
    coord: ChunkCoord,
    palette: Vec<BlockId>,
    indices: Vec<u16>,
    occupancy: BitVec,
    non_air: u32,
    dirty: bool,
    version: u32,
}

impl Chunk {
    /// 创建一个全空气 Chunk。
    pub fn new_empty(coord: ChunkCoord) -> Self {
        Self {
            coord,
            palette: vec![BlockId::AIR],
            indices: vec![0; CHUNK_VOLUME],
            occupancy: BitVec::repeat(false, CHUNK_VOLUME),
            non_air: 0,
            dirty: false,
            version: 0,
        }
    }

    /// 由体素缓冲构造（会自动建立调色板）。
    pub fn from_voxels(coord: ChunkCoord, voxels: &[BlockId]) -> Result<Self, StorageError> {
        if voxels.len() != CHUNK_VOLUME {
            return Err(StorageError::VolumeSizeMismatch {
                expected: CHUNK_VOLUME,
                actual: voxels.len(),
            });
        }
        let mut chunk = Self::new_empty(coord);
        for (index, block) in voxels.iter().copied().enumerate() {
            if block.is_air() {
                continue;
            }
            let slot = chunk.palette_slot(block)?;
            chunk.indices[index] = slot;
            chunk.occupancy.set(index, true);
            chunk.non_air += 1;
        }
        chunk.dirty = true;
        Ok(chunk)
    }

    /// 查找或插入调色板槽位。空气恒为 0。
    fn palette_slot(&mut self, block: BlockId) -> Result<u16, StorageError> {
        if let Some(position) = self
            .palette
            .iter()
            .position(|candidate| *candidate == block)
        {
            return Ok(position as u16);
        }
        if self.palette.len() > u16::MAX as usize {
            return Err(StorageError::PaletteOverflow {
                len: self.palette.len(),
            });
        }
        self.palette.push(block);
        Ok((self.palette.len() - 1) as u16)
    }

    /// Chunk 坐标。
    #[inline]
    pub fn coord(&self) -> ChunkCoord {
        self.coord
    }

    /// 读取体素；越界返回 [`BlockId::AIR`]（边界必须宽容，否则每个采样点都要判界）。
    #[inline]
    pub fn get(&self, x: u32, y: u32, z: u32) -> BlockId {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return BlockId::AIR;
        }
        let index = local_index(x, y, z);
        self.palette
            .get(self.indices[index] as usize)
            .copied()
            .unwrap_or(BlockId::AIR)
    }

    /// 写入体素，返回 **内容是否真的发生变化**（用于脏标记与网络差量）。
    pub fn set(&mut self, x: u32, y: u32, z: u32, block: BlockId) -> Result<bool, StorageError> {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return Err(StorageError::OutOfBounds { x, y, z });
        }
        let index = local_index(x, y, z);
        let slot = self.palette_slot(block)?;
        let changed = self.indices[index] != slot;
        if !changed {
            return Ok(false);
        }
        self.indices[index] = slot;
        let occupied = !block.is_air();
        let was_occupied = self.occupancy[index];
        self.occupancy.set(index, occupied);
        match (was_occupied, occupied) {
            (false, true) => self.non_air += 1,
            (true, false) => self.non_air -= 1,
            _ => {},
        }
        self.dirty = true;
        self.version = self.version.wrapping_add(1);
        Ok(true)
    }

    /// 非空气体素数量。
    #[inline]
    pub fn non_air_count(&self) -> u32 {
        self.non_air
    }

    /// 是否完全为空气（可整块跳过网格化）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.non_air == 0
    }

    /// 是否存在未落盘的修改。
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// 清除脏标记（落盘后调用）。
    #[inline]
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// 内容版本号，每次实际修改自增（用于差量同步与缓存失效）。
    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// 还原版本号（用于从快照/存档恢复基准）。
    ///
    /// 版本号是差量同步的基准：从快照重建的 Chunk 必须拿回快照里记录的版本，
    /// 否则带着旧 `base_version` 的差量会被 `DeltaTable::apply_all` 判为「基准已变」
    /// 而整体跳过，看起来就像玩家改动凭空消失。
    #[inline]
    pub fn restore_version(&mut self, version: u32) {
        self.version = version;
    }

    /// 调色板项数。
    #[inline]
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// 调色板内容。
    #[inline]
    pub fn palette(&self) -> &[BlockId] {
        &self.palette
    }

    /// 每体素所需的索引位宽（至少 1 位）。
    pub fn index_bits(&self) -> u32 {
        let mut bits = 1;
        while (1usize << bits) < self.palette.len() {
            bits += 1;
        }
        bits as u32
    }

    /// 落盘时的理论字节数（调色板 + 按位宽打包的索引 + 占用位图）。
    pub fn packed_bytes(&self) -> usize {
        let bits = self.index_bits() as usize;
        self.palette_len() * core::mem::size_of::<BlockId>()
            + (CHUNK_VOLUME * bits).div_ceil(8)
            + CHUNK_VOLUME.div_ceil(8)
    }

    /// 未压缩的朴素字节数（`BlockId` 数组），用于对比压缩收益。
    pub fn naive_bytes(&self) -> usize {
        CHUNK_VOLUME * core::mem::size_of::<BlockId>()
    }

    /// 占用位图（1 = 非空气）的只读视图。
    #[inline]
    pub fn occupancy(&self) -> &BitSlice {
        self.occupancy.as_bitslice()
    }

    /// 遍历全部非空气体素：`(x, y, z, block)`。
    pub fn iter_solid(&self) -> impl Iterator<Item = (u32, u32, u32, BlockId)> + '_ {
        self.occupancy
            .iter()
            .by_vals()
            .enumerate()
            .filter(|(_, occupied)| *occupied)
            .map(move |(index, _)| {
                let index = index as u32;
                let x = index % CHUNK_EDGE;
                let y = (index / CHUNK_EDGE) % CHUNK_EDGE;
                let z = index / (CHUNK_EDGE * CHUNK_EDGE);
                (x, y, z, self.get(x, y, z))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voxels_with(blocks: &[(u32, u32, u32, BlockId)]) -> Vec<BlockId> {
        let mut voxels = vec![BlockId::AIR; CHUNK_VOLUME];
        for &(x, y, z, block) in blocks {
            voxels[local_index(x, y, z)] = block;
        }
        voxels
    }

    #[test]
    fn empty_chunk_is_all_air() {
        let chunk = Chunk::new_empty(ChunkCoord::default());
        assert!(chunk.is_empty());
        assert_eq!(chunk.non_air_count(), 0);
        assert!(!chunk.is_dirty());
        assert_eq!(chunk.palette_len(), 1);
        assert_eq!(chunk.get(0, 0, 0), BlockId::AIR);
        assert_eq!(chunk.get(CHUNK_EDGE, 0, 0), BlockId::AIR, "越界读必须宽容");
        assert_eq!(chunk.iter_solid().count(), 0);
    }

    #[test]
    fn palette_deduplicates_and_counts_occupancy() {
        let block = BlockId::new(1);
        let chunk = Chunk::from_voxels(
            ChunkCoord::new(1, -1, 0),
            &voxels_with(&[
                (0, 0, 0, block),
                (1, 0, 0, block),
                (31, 31, 31, BlockId::new(2)),
            ]),
        )
        .unwrap();
        assert_eq!(chunk.palette_len(), 3, "空气 + 2 种方块");
        assert_eq!(chunk.non_air_count(), 3);
        assert_eq!(chunk.get(1, 0, 0), block);
        assert_eq!(chunk.get(31, 31, 31), BlockId::new(2));
        assert!(chunk.is_dirty());
        assert_eq!(chunk.iter_solid().count(), 3);
        assert_eq!(chunk.coord(), ChunkCoord::new(1, -1, 0));
    }

    #[test]
    fn set_reports_real_changes_only() {
        let mut chunk = Chunk::new_empty(ChunkCoord::default());
        assert!(chunk.set(5, 5, 5, BlockId::new(4)).unwrap());
        assert_eq!(chunk.version(), 1);
        assert!(
            !chunk.set(5, 5, 5, BlockId::new(4)).unwrap(),
            "同值写入不算变化"
        );
        assert_eq!(chunk.version(), 1);
        assert!(chunk.set(5, 5, 5, BlockId::AIR).unwrap());
        assert_eq!(chunk.non_air_count(), 0);
        assert!(chunk.is_empty());
        assert!(matches!(
            chunk.set(CHUNK_EDGE, 0, 0, BlockId::new(1)),
            Err(StorageError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn volume_size_is_validated() {
        let err = Chunk::from_voxels(ChunkCoord::default(), &[BlockId::AIR; 4]).unwrap_err();
        assert_eq!(
            err,
            StorageError::VolumeSizeMismatch {
                expected: CHUNK_VOLUME,
                actual: 4
            }
        );
    }

    #[test]
    fn compression_is_reported_against_naive_layout() {
        let mut chunk = Chunk::new_empty(ChunkCoord::default());
        chunk.set(1, 1, 1, BlockId::new(1)).unwrap();
        chunk.set(2, 1, 1, BlockId::new(2)).unwrap();
        assert_eq!(chunk.index_bits(), 2, "3 项调色板需要 2 位");
        assert_eq!(chunk.naive_bytes(), CHUNK_VOLUME * 2);
        assert!(
            chunk.packed_bytes() * 4 < chunk.naive_bytes(),
            "调色板压缩至少应有 4 倍收益"
        );
    }
}
