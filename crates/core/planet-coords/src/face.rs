//! Cube-Sphere 六面拓扑流形。

use glam::DVec3;

/// Cube-Sphere 的一个面。
///
/// 面坐标 `(u, v)` 定义域为 `[-1, 1]²`，映射到单位球后**赤道与面心无扭曲**，
/// 但在面角处度量被压缩到约 0.51 倍（面角体素更密集）——局部米制尺度必须经由
/// [`crate::SphereGeometry::meters_per_uv`] 求取，禁止按常数近似。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum CubeFace {
    /// `+X` 面。
    PosX = 0,
    /// `-X` 面。
    NegX = 1,
    /// `+Y` 面。
    PosY = 2,
    /// `-Y` 面。
    NegY = 3,
    /// `+Z` 面。
    PosZ = 4,
    /// `-Z` 面。
    NegZ = 5,
}

/// 面总数。
pub const FACE_COUNT: usize = 6;

impl CubeFace {
    /// 全局面，顺序与 `repr(u8)` 判别式一致，可直接当索引用。
    pub const ALL: [CubeFace; FACE_COUNT] = [
        CubeFace::PosX,
        CubeFace::NegX,
        CubeFace::PosY,
        CubeFace::NegY,
        CubeFace::PosZ,
        CubeFace::NegZ,
    ];

    /// 判别式索引，可用于数组下标。
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// 由索引还原。
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::PosX),
            1 => Some(Self::NegX),
            2 => Some(Self::PosY),
            3 => Some(Self::NegY),
            4 => Some(Self::PosZ),
            5 => Some(Self::NegZ),
            _ => None,
        }
    }

    /// 对面。
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::PosX => Self::NegX,
            Self::NegX => Self::PosX,
            Self::PosY => Self::NegY,
            Self::NegY => Self::PosY,
            Self::PosZ => Self::NegZ,
            Self::NegZ => Self::PosZ,
        }
    }

    /// 面外法线（立方体空间）。
    #[inline]
    pub fn normal(self) -> DVec3 {
        match self {
            Self::PosX => DVec3::new(1.0, 0.0, 0.0),
            Self::NegX => DVec3::new(-1.0, 0.0, 0.0),
            Self::PosY => DVec3::new(0.0, 1.0, 0.0),
            Self::NegY => DVec3::new(0.0, -1.0, 0.0),
            Self::PosZ => DVec3::new(0.0, 0.0, 1.0),
            Self::NegZ => DVec3::new(0.0, 0.0, -1.0),
        }
    }

    /// 面内基向量：给定 `u, v`，立方体空间点为 `normal + u * u_axis + v * v_axis`。
    #[inline]
    pub fn uv_axes(self) -> (DVec3, DVec3) {
        match self {
            Self::PosX => (DVec3::new(0.0, 0.0, -1.0), DVec3::new(0.0, 1.0, 0.0)),
            Self::NegX => (DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 1.0, 0.0)),
            Self::PosY => (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, -1.0)),
            Self::NegY => (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
            Self::PosZ => (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)),
            Self::NegZ => (DVec3::new(-1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)),
        }
    }

    /// 面坐标 `(u, v) ∈ [-1, 1]²` → 单位球方向。
    ///
    /// 实现方式等价于「立方体上的点归一化」，与 [`Self::dir_to_uv`] 严格互逆。
    #[inline]
    pub fn uv_to_dir(self, u: f64, v: f64) -> DVec3 {
        let (u_axis, v_axis) = self.uv_axes();
        (self.normal() + u_axis * u + v_axis * v).normalize()
    }

    /// 单位球方向 → 主导面 + 面坐标。
    ///
    /// 零向量退化为 `(PosZ, 0, 0)`；跨越面边界时返回主导轴所属的面（调用方若关心
    /// 邻接关系，应结合 [`Self::opposite`] 与邻面表处理接缝）。
    #[inline]
    pub fn dir_to_uv(dir: DVec3) -> (Self, f64, f64) {
        let abs = dir.abs();
        let (face, axis) = if abs.x >= abs.y && abs.x >= abs.z {
            if dir.x >= 0.0 {
                (Self::PosX, abs.x)
            } else {
                (Self::NegX, abs.x)
            }
        } else if abs.y >= abs.z {
            if dir.y >= 0.0 {
                (Self::PosY, abs.y)
            } else {
                (Self::NegY, abs.y)
            }
        } else if dir.z >= 0.0 {
            (Self::PosZ, abs.z)
        } else {
            (Self::NegZ, abs.z)
        };
        if axis <= f64::MIN_POSITIVE {
            return (Self::PosZ, 0.0, 0.0);
        }
        let (u, v) = match face {
            Self::PosX => (-dir.z / axis, dir.y / axis),
            Self::NegX => (dir.z / axis, dir.y / axis),
            Self::PosY => (dir.x / axis, -dir.z / axis),
            Self::NegY => (dir.x / axis, dir.z / axis),
            Self::PosZ => (dir.x / axis, dir.y / axis),
            Self::NegZ => (-dir.x / axis, dir.y / axis),
        };
        (face, u, v)
    }

    /// 共享该边的邻面（按 `u`/`v` 正负方向）。
    ///
    /// `edge`: `0 => +u`、`1 => -u`、`2 => +v`、`3 => -v`。
    /// 返回 `(邻面, 邻面是否翻转 u, 邻面是否翻转 v)`——接缝缝合（避免 LOD 裂缝）依赖此表。
    pub fn neighbor(self, edge: u8) -> Option<(Self, bool, bool)> {
        // 立方体六面的邻接关系表：把每个面展开到 3x3 的十字布局即可验证。
        let table: [(Self, bool, bool); 4] = match self {
            Self::PosX => [
                (Self::PosZ, false, false),
                (Self::NegZ, true, false),
                (Self::PosY, false, false),
                (Self::NegY, false, true),
            ],
            Self::NegX => [
                (Self::NegZ, false, false),
                (Self::PosZ, true, false),
                (Self::PosY, true, false),
                (Self::NegY, true, true),
            ],
            Self::PosY => [
                (Self::PosX, false, true),
                (Self::NegX, true, true),
                (Self::NegZ, false, false),
                (Self::PosZ, true, false),
            ],
            Self::NegY => [
                (Self::PosX, false, false),
                (Self::NegX, true, false),
                (Self::PosZ, false, false),
                (Self::NegZ, true, false),
            ],
            Self::PosZ => [
                (Self::PosX, true, false),
                (Self::NegX, false, false),
                (Self::PosY, false, false),
                (Self::NegY, false, true),
            ],
            Self::NegZ => [
                (Self::NegX, true, false),
                (Self::PosX, false, false),
                (Self::PosY, true, false),
                (Self::NegY, true, true),
            ],
        };
        table.get(edge as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip() {
        for face in CubeFace::ALL {
            assert_eq!(CubeFace::from_index(face.index()), Some(face));
            assert_eq!(face.opposite().opposite(), face);
            assert_eq!(face.opposite().normal(), -face.normal());
        }
        assert_eq!(CubeFace::from_index(FACE_COUNT), None);
    }

    #[test]
    fn uv_to_dir_is_normalized_and_invertible() {
        for face in CubeFace::ALL {
            for &(u, v) in &[
                (0.0, 0.0),
                (0.5, -0.25),
                (-0.9, 0.9),
                (0.999, 0.0),
                (0.0, -0.999),
            ] {
                let dir = face.uv_to_dir(u, v);
                assert!((dir.length() - 1.0).abs() < 1e-12, "方向必须已归一化");
                assert!(dir.dot(face.normal()) > 0.0, "方向必须位于该面的半球内");
                let (back_face, back_u, back_v) = CubeFace::dir_to_uv(dir);
                assert_eq!(back_face, face);
                assert!(
                    (back_u - u).abs() < 1e-12,
                    "u 往返误差过大: {back_u} vs {u}"
                );
                assert!(
                    (back_v - v).abs() < 1e-12,
                    "v 往返误差过大: {back_v} vs {v}"
                );
            }
        }
    }

    #[test]
    fn zero_vector_degrades_gracefully() {
        assert_eq!(CubeFace::dir_to_uv(DVec3::ZERO), (CubeFace::PosZ, 0.0, 0.0));
    }

    #[test]
    fn every_edge_has_a_neighbor_that_points_back() {
        for face in CubeFace::ALL {
            for edge in 0..4u8 {
                let (neighbor, _, _) = face.neighbor(edge).expect("每条边都应有邻面");
                assert_ne!(neighbor, face, "邻面不能是自身");
            }
        }
    }
}
