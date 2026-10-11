//! # physics —— 局部物理
//!
//! 行星表面的物理必须建立在**局部切线标架**上，而不是全局笛卡尔坐标：
//! 在半径 6371 km 的球面上，两个相距 1 km 的点之间的「直线」已经偏离地表 2 cm，
//! 相距 100 km 时偏离 200 m——直接用全局 AABB 做碰撞会得到一堆莫名其妙的卡顿。
//!
//! 本 crate 的解法：
//!
//! 1. **局部切线标架**：以角色所在球面点为原点，`(east, north, up)` 为基，
//!    在局部坐标中做普通的 AABB / 扫掠碰撞（与平面世界完全一致的代码路径）；
//! 2. **球面重力**（[`SphericalGravity`]）：重力方向随位置变化，按高度衰减；
//! 3. **光标射线拾取**（[`raycast`]）：体素 DDA，不漏薄壁、不用逐点采样。
//!
//! 局部标架的坐标变换与「角色胶囊 vs Chunk AABB」求解器列在路线图上
//! （见 [`LocalFrame`] 的 TODO）；拾取与重力已可用于实际交互。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod gravity;
pub mod picking;

use glam::DVec3;
use planet_coords::{SphereGeometry, SurfacePoint};

pub use gravity::SphericalGravity;
pub use picking::{raycast, Ray, RayHit};

/// 局部切线标架。
///
/// 把球面点附近的一小块区域「拍平」成正交坐标系，使平面物理求解器可以原样复用：
///
/// * `up` = 球面外法线；
/// * `east` / `north` = 切平面上的两个正交方向；
/// * `origin_m` = 标架原点的世界坐标（米）。
///
/// 精度保证：在标架原点 ±2 km 内，`world ↔ local` 往返误差 < 1 mm。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalFrame {
    /// 标架原点（世界坐标，米）。
    pub origin_m: DVec3,
    /// 上方向（单位向量，指向天顶）。
    pub up: DVec3,
    /// 东方向（单位向量）。
    pub east: DVec3,
    /// 北方向（单位向量）。
    pub north: DVec3,
}

impl LocalFrame {
    /// 由球面点与行星半径构造。
    pub fn at(point: SurfacePoint, radius_m: f64, elevation_m: f64) -> Self {
        let up = point.to_dir();
        let (east, north) = SphereGeometry::tangent_frame(up);
        Self {
            origin_m: up * (radius_m + elevation_m),
            up,
            east,
            north,
        }
    }

    /// 世界坐标 → 局部切线坐标。
    ///
    /// TODO(physics): 增加球面投影修正项，使标架在 ±50 km 范围仍保持毫米级精度。
    pub fn to_local(&self, world_m: DVec3) -> DVec3 {
        let delta = world_m - self.origin_m;
        DVec3::new(
            delta.dot(self.east),
            delta.dot(self.up),
            delta.dot(self.north),
        )
    }

    /// 局部切线坐标 → 世界坐标。
    pub fn to_world(&self, local: DVec3) -> DVec3 {
        self.origin_m + self.east * local.x + self.up * local.y + self.north * local.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use planet_coords::CubeFace;

    #[test]
    fn frame_is_orthonormal() {
        let frame = LocalFrame::at(
            SurfacePoint::new(CubeFace::PosZ, 0.3, -0.4),
            6_371_000.0,
            0.0,
        );
        assert!((frame.up.length() - 1.0).abs() < 1e-12);
        assert!((frame.east.length() - 1.0).abs() < 1e-12);
        assert!((frame.north.length() - 1.0).abs() < 1e-12);
        assert!(frame.east.dot(frame.up).abs() < 1e-12);
        assert!(frame.east.dot(frame.north).abs() < 1e-12);
        assert!(frame.up.dot(frame.north).abs() < 1e-12);
    }

    #[test]
    fn local_and_world_round_trip_near_the_origin() {
        let frame = LocalFrame::at(
            SurfacePoint::new(CubeFace::NegX, -0.7, 0.2),
            6_371_000.0,
            12.0,
        );
        for local in [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(100.0, 1.7, -50.0),
            DVec3::new(-1_500.0, 300.0, 900.0),
        ] {
            let back = frame.to_local(frame.to_world(local));
            assert!((back - local).length() < 1e-9, "往返误差过大: {back:?}");
        }
    }
}
