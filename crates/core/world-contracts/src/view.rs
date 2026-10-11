//! 体素世界只读视图契约。

use block_schema::BlockId;
use planet_coords::VoxelCoord;

/// Chunk 边长（体素）。32 是「一个 warp 的 32 线程」与「16 位 Morton 键可寻址」的交点。
pub const CHUNK_EDGE: u32 = 32;

/// 单个 Chunk 的体素数。
pub const CHUNK_VOLUME: usize = (CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE) as usize;

/// Chunk 的全局坐标（每格 32 体素）。
///
/// 用 `i64` 而非 `i32`：半径 1e9 米的巨型行星上，Chunk 索引仍可达 3e7 量级，
/// 且中间量（体素索引 × 32）需要留出安全余量。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct ChunkCoord {
    /// Chunk X 索引。
    pub x: i64,
    /// Chunk Y 索引。
    pub y: i64,
    /// Chunk Z 索引。
    pub z: i64,
}

impl ChunkCoord {
    /// 直接构造。
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// 体素坐标落在哪个 Chunk（向下取整，负数同样正确）。
    pub const fn from_voxel(voxel: VoxelCoord) -> Self {
        Self {
            x: voxel.x.div_euclid(CHUNK_EDGE as i64),
            y: voxel.y.div_euclid(CHUNK_EDGE as i64),
            z: voxel.z.div_euclid(CHUNK_EDGE as i64),
        }
    }

    /// 该 Chunk 最小角（体素坐标）。
    pub const fn min_voxel(self) -> VoxelCoord {
        VoxelCoord::new(
            self.x * CHUNK_EDGE as i64,
            self.y * CHUNK_EDGE as i64,
            self.z * CHUNK_EDGE as i64,
        )
    }

    /// Chunk 中心的世界坐标（米）。
    pub fn center_m(self) -> glam::DVec3 {
        let base = self.min_voxel();
        glam::DVec3::new(
            (base.x as f64 + CHUNK_EDGE as f64 * 0.5) * planet_coords::voxel_edge_m(),
            (base.y as f64 + CHUNK_EDGE as f64 * 0.5) * planet_coords::voxel_edge_m(),
            (base.z as f64 + CHUNK_EDGE as f64 * 0.5) * planet_coords::voxel_edge_m(),
        )
    }

    /// 用于哈希与落盘的稳定键。
    pub const fn key(self) -> [i64; 3] {
        [self.x, self.y, self.z]
    }
}

/// 体素世界的**只读**视图。
///
/// 网格生成、物理拾取、服务端权威校验都只依赖这个接口，从而对「Chunk 从哪来」
/// 完全无感（内存池 / 磁盘 / 程序化生成 / 网络快照）。
///
/// 实现者必须保证：
///
/// * 越界或未加载一律返回 [`BlockId::AIR`]（绝不能 panic）；
/// * 方法可任意并发调用（`Send + Sync`）；
/// * 在一次网格生成任务内，视图内容保持稳定（由调用方持有租约保证）。
pub trait WorldView: Send + Sync {
    /// 查询某个全局体素坐标的方块。
    fn block_at(&self, voxel: VoxelCoord) -> BlockId;

    /// 该 Chunk 是否已在视图中可用。
    fn has_chunk(&self, coord: ChunkCoord) -> bool;

    /// 块是否实体（参与碰撞与遮挡）。默认走注册表语义：非空气即实体。
    fn is_solid(&self, voxel: VoxelCoord) -> bool {
        !self.block_at(voxel).is_air()
    }

    /// 表面高度（米，相对参考半径）在给定方向上的估计值。
    ///
    /// 供物理拾取与 LOD 选择使用；实现可以返回基于已加载 Chunk 的近似值。
    fn surface_height_m(&self, dir: glam::DVec3) -> Option<f64>;

    /// 体素边长（米）。
    fn voxel_edge_m(&self) -> f64 {
        planet_coords::voxel_edge_m()
    }
}

/// 一个恒为空气的空视图，供测试与「尚未加载世界」的启动阶段使用。
#[derive(Clone, Copy, Debug, Default)]
pub struct EmptyWorld;

impl WorldView for EmptyWorld {
    fn block_at(&self, _voxel: VoxelCoord) -> BlockId {
        BlockId::AIR
    }

    fn has_chunk(&self, _coord: ChunkCoord) -> bool {
        false
    }

    fn surface_height_m(&self, _dir: glam::DVec3) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_to_chunk_handles_negative_coordinates() {
        let chunk = ChunkCoord::from_voxel(VoxelCoord::new(-1, 0, 31));
        assert_eq!(chunk, ChunkCoord::new(-1, 0, 0));
        assert_eq!(chunk.min_voxel(), VoxelCoord::new(-32, 0, 0));
        assert_eq!(
            ChunkCoord::from_voxel(VoxelCoord::new(32, -33, 63)),
            ChunkCoord::new(1, -2, 1)
        );
        assert_eq!(
            ChunkCoord::new(2, 0, 0).min_voxel(),
            VoxelCoord::new(64, 0, 0)
        );
    }

    #[test]
    fn empty_world_never_panics() {
        let world = EmptyWorld;
        assert!(world
            .block_at(VoxelCoord::new(i64::MIN, i64::MAX, 0))
            .is_air());
        assert!(!world.has_chunk(ChunkCoord::default()));
        assert_eq!(world.surface_height_m(glam::DVec3::X), None);
        assert_eq!(CHUNK_VOLUME, 32 * 32 * 32);
    }
}
