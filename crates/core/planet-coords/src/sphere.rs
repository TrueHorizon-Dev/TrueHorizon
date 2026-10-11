//! 球面点、测地距离与局部标架。

use crate::face::CubeFace;
use glam::DVec3;

/// 行星表面的一个位置：`面 + 面坐标`。
///
/// 相比三维单位向量，这种表示是**无冗余**的（3 个自由度对 2 个），因此在跨面插值、
/// 序列化与 Morton 编码时不会引入「归一化漂移」。
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SurfacePoint {
    /// 所属面。
    pub face: CubeFace,
    /// 面内 u 坐标，`[-1, 1]`。
    pub u: f64,
    /// 面内 v 坐标，`[-1, 1]`。
    pub v: f64,
}

impl SurfacePoint {
    /// 直接构造（不做范围检查）。
    pub const fn new(face: CubeFace, u: f64, v: f64) -> Self {
        Self { face, u, v }
    }

    /// 由方向向量构造。
    pub fn from_dir(dir: DVec3) -> Self {
        let (face, u, v) = CubeFace::dir_to_uv(dir);
        Self { face, u, v }
    }

    /// 转成单位球方向。
    pub fn to_dir(self) -> DVec3 {
        self.face.uv_to_dir(self.u, self.v)
    }
}

/// 供 GPU / 网络使用的紧凑表示：面 + 半精度面坐标。
///
/// 注意：`u`/`v` 降为 `f32` 后，在半径 `R` 米的行星上定位误差约为 `R * 6e-8` 米
/// （`R = 6.371e6` 时约 0.4 米）。因此本类型**只用于渲染与兴趣管理**，
/// 不能回写世界状态。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedSurfacePoint {
    /// 面索引。
    pub face: u8,
    /// 对齐填充。
    pub _pad: [u8; 3],
    /// 面内 u 坐标。
    pub u: f32,
    /// 面内 v 坐标。
    pub v: f32,
}

impl PackedSurfacePoint {
    /// 由精确球面点降精度打包。
    pub fn from_surface(point: SurfacePoint) -> Self {
        Self {
            face: point.face as u8,
            _pad: [0; 3],
            u: point.u as f32,
            v: point.v as f32,
        }
    }

    /// 还原为精确类型（含降精度误差）。
    pub fn to_surface(self) -> Option<SurfacePoint> {
        Some(SurfacePoint::new(
            CubeFace::from_index(self.face as usize)?,
            self.u as f64,
            self.v as f64,
        ))
    }
}

/// 球面几何工具：把「行星半径」这个唯一的全局尺度参数收敛到一处。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereGeometry {
    /// 行星平均半径（米）。地球取 6_371_000.0。
    pub radius_m: f64,
}

impl SphereGeometry {
    /// 以半径构造。
    pub const fn new(radius_m: f64) -> Self {
        Self { radius_m }
    }

    /// 两个单位方向之间的圆心角（弧度）。
    pub fn angular_distance(a: DVec3, b: DVec3) -> f64 {
        let cross = a.cross(b).length();
        let dot = a.dot(b);
        // atan2 形式在极小与极大夹角下都比 acos 稳定。
        cross.atan2(dot)
    }

    /// 沿球面的测地距离（米）。
    pub fn arc_len(&self, a: DVec3, b: DVec3) -> f64 {
        Self::angular_distance(a, b) * self.radius_m
    }

    /// 球面点处的东/北切线标架（右手系，`east × north = 外法线`）。
    ///
    /// 极点附近 `east` 退化，此时取任意正交方向。
    pub fn tangent_frame(dir: DVec3) -> (DVec3, DVec3) {
        let up = dir.normalize();
        let east = DVec3::new(0.0, 1.0, 0.0).cross(up);
        let east = if east.length_squared() < 1e-24 {
            DVec3::new(1.0, 0.0, 0.0)
        } else {
            east.normalize()
        };
        let north = up.cross(east);
        (east, north)
    }

    /// 面坐标每单位 `u`/`v` 对应的米数（含 Cube-Sphere 的角部拉伸与曲率效应）。
    ///
    /// 用中心差分求 `|d(dir)/du| * R`，精度足够且不依赖各面的解析度量。
    pub fn meters_per_uv(&self, point: SurfacePoint) -> (f64, f64) {
        const H: f64 = 1e-6;
        let du = (point.face.uv_to_dir(point.u + H, point.v)
            - point.face.uv_to_dir(point.u - H, point.v))
        .length()
            / (2.0 * H)
            * self.radius_m;
        let dv = (point.face.uv_to_dir(point.u, point.v + H)
            - point.face.uv_to_dir(point.u, point.v - H))
        .length()
            / (2.0 * H)
            * self.radius_m;
        (du, dv)
    }

    /// 高度为 `height_m` 的观察点能看到的最远球面弧长（米）。
    pub fn horizon_arc_m(&self, height_m: f64) -> f64 {
        let cos_theta = (self.radius_m / (self.radius_m + height_m.max(0.0))).clamp(-1.0, 1.0);
        self.radius_m * cos_theta.acos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH: SphereGeometry = SphereGeometry::new(6_371_000.0);

    #[test]
    fn surface_point_round_trip() {
        let dir = DVec3::new(0.3, -0.7, 0.2).normalize();
        let point = SurfacePoint::from_dir(dir);
        assert!((point.to_dir() - dir).length() < 1e-12);
    }

    #[test]
    fn angular_distance_matches_known_quarter_turn() {
        let a = DVec3::X;
        let b = DVec3::Y;
        assert!(
            (SphereGeometry::angular_distance(a, b) - std::f64::consts::FRAC_PI_2).abs() < 1e-15
        );
        assert!((EARTH.arc_len(a, b) - EARTH.radius_m * std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(SphereGeometry::angular_distance(a, a), 0.0);
    }

    #[test]
    fn tangent_frame_is_orthonormal_and_handed() {
        for dir in [
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.3, 0.5, -0.8).normalize(),
            -DVec3::Z,
        ] {
            let (east, north) = SphereGeometry::tangent_frame(dir);
            assert!((east.length() - 1.0).abs() < 1e-12);
            assert!((north.length() - 1.0).abs() < 1e-12);
            assert!(east.dot(north).abs() < 1e-12);
            let up = dir.normalize();
            assert!((east.cross(north) - up).length() < 1e-12, "必须构成右手系");
        }
    }

    #[test]
    fn meters_per_uv_is_isometric_at_center_and_compressed_at_corners() {
        // 面心是等距点：映射在此处的雅可比行列式为 1，故每单位 u/v 恰为 R 米。
        let (du_center, dv_center) =
            EARTH.meters_per_uv(SurfacePoint::new(CubeFace::PosZ, 0.0, 0.0));
        assert!(
            (du_center - EARTH.radius_m).abs() / EARTH.radius_m < 1e-4,
            "du = {du_center}"
        );
        assert!(
            (dv_center - EARTH.radius_m).abs() / EARTH.radius_m < 1e-4,
            "dv = {dv_center}"
        );

        // 面角处的度量压缩到约 0.5135 倍：这是立方体映射的固有性质
        // （面角处体素更密集），也正是「禁止用常数近似局部尺度」的原因。
        let (du_corner, dv_corner) =
            EARTH.meters_per_uv(SurfacePoint::new(CubeFace::PosZ, 0.9, 0.9));
        let ratio = du_corner / du_center;
        assert!(
            (ratio - 0.5135).abs() < 5e-3,
            "角部度量比例应约 0.5135，实得 {ratio}"
        );
        assert!(du_corner < du_center, "角部必然被压缩");
        assert!((dv_corner - du_corner).abs() < 1.0, "面角处两轴度量应对称");
    }

    #[test]
    fn packed_point_round_trips_within_f32() {
        let point = SurfacePoint::new(CubeFace::NegY, 0.123456789, -0.987654321);
        let packed = PackedSurfacePoint::from_surface(point);
        let back = packed.to_surface().expect("面索引必须合法");
        assert_eq!(back.face, point.face);
        assert!((back.u - point.u).abs() < 1e-6);
        assert!((back.v - point.v).abs() < 1e-6);
    }

    #[test]
    fn horizon_grows_with_height() {
        let sea = EARTH.horizon_arc_m(0.0);
        let eye = EARTH.horizon_arc_m(2.0);
        let orbit = EARTH.horizon_arc_m(400_000.0);
        assert!(sea < eye && eye < orbit);
    }
}
