//! # planet-coords —— 行星坐标与拓扑
//!
//! 整个引擎的空间基准。所有跨越「行星尺度」的量都必须经过本 crate，禁止在别处裸用 `f32`
//! 表示世界坐标。
//!
//! * [`face`]：Cube-Sphere 六面拓扑流形与面无扭曲映射。
//! * [`sphere`]：球面点、测地距离、切线标架与局部米制尺度。
//! * [`origin`]：双精度浮动原点（Floating Origin），渲染前才降到 `f32`。
//! * [`occlusion`]：曲率遮挡，用于兴趣管理（AoI）与地平线可视性裁剪。
//!
//! 单位约定：**长度一律为米**，角度为弧度，方向向量为单位长度（本 crate 会在入口处归一化）。
//! `glam` 统一开启 `scalar-math`，保证 CPU 结果与 GPU 逐位一致、可复现。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod face;
pub mod occlusion;
pub mod origin;
pub mod sphere;

pub use face::CubeFace;
pub use occlusion::CurvatureOcclusion;
pub use origin::FloatingOrigin;
pub use sphere::{PackedSurfacePoint, SphereGeometry, SurfacePoint};

/// 体素级整数坐标（全局格子，原点在行星质心）。
///
/// 体素边长由 [`SphereGeometry::radius_m`] 与 [`voxel_edge_m`] 共同决定；`i64` 保证在
/// 半径 1e9 米的尺度下依然不溢出（`1e9 * 2 < i64::MAX`）。
pub const fn voxel_edge_m() -> f64 {
    1.0
}

/// 全局体素坐标。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct VoxelCoord {
    /// X 轴体素索引。
    pub x: i64,
    /// Y 轴体素索引。
    pub y: i64,
    /// Z 轴体素索引。
    pub z: i64,
}

impl VoxelCoord {
    /// 构造体素坐标。
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// 体素中心的世界坐标（米）。
    pub fn center_m(self) -> glam::DVec3 {
        glam::DVec3::new(
            (self.x as f64 + 0.5) * voxel_edge_m(),
            (self.y as f64 + 0.5) * voxel_edge_m(),
            (self.z as f64 + 0.5) * voxel_edge_m(),
        )
    }

    /// 世界坐标落在哪个体素内。
    pub fn from_world_m(world: glam::DVec3) -> Self {
        Self::new(
            (world.x / voxel_edge_m()).floor() as i64,
            (world.y / voxel_edge_m()).floor() as i64,
            (world.z / voxel_edge_m()).floor() as i64,
        )
    }
}
