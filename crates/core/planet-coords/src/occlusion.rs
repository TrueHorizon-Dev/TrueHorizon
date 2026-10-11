//! 曲率遮挡：球面上的视距与兴趣管理裁剪。

use glam::DVec3;

/// 球体自身的遮挡模型。
///
/// 行星不是平面：两点之间的**直线**会被球面顶部的「隆起」切断。对半径 `R`、夹角 `θ`
/// 的两个表面点，弦中点位于球面之下的距离为
///
/// ```text
/// bulge = R * (1 - cos(θ / 2))
/// ```
///
/// 该量随 `θ²` 增长：`θ` 为 1° 时约 0.008 m，30° 时已达 0.11·R。
/// 服务器用它做兴趣管理（AoI）与视线校验，客户端用它做地平线裁剪。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureOcclusion {
    /// 行星半径（米）。
    pub radius_m: f64,
}

impl CurvatureOcclusion {
    /// 以半径构造。
    pub const fn new(radius_m: f64) -> Self {
        Self { radius_m }
    }

    /// 两点之间的隆起高度（米），即「遮挡余量」。
    ///
    /// 输入方向会在内部归一化，无需调用方保证。
    pub fn bulge_m(&self, eye_dir: DVec3, target_dir: DVec3) -> f64 {
        let theta = (eye_dir.normalize().cross(target_dir.normalize()))
            .length()
            .atan2(eye_dir.normalize().dot(target_dir.normalize()));
        self.radius_m * (1.0 - (theta * 0.5).cos())
    }

    /// 从 `eye_dir` 到 `target_dir` 是否被球面本身挡住。
    ///
    /// `clearance_m` 是允许「穿透」的地形起伏余量（例如允许越过高 20 米的山脊）。
    pub fn is_blocked(&self, eye_dir: DVec3, target_dir: DVec3, clearance_m: f64) -> bool {
        self.bulge_m(eye_dir, target_dir) > clearance_m
    }

    /// 高度 `height_m` 处的观察者能看到的地平线夹角（弧度）。
    pub fn horizon_angle(&self, height_m: f64) -> f64 {
        (self.radius_m / (self.radius_m + height_m.max(0.0)))
            .clamp(-1.0, 1.0)
            .acos()
    }

    /// 高度 `height_m` 处的地平线弧长（米）。
    pub fn horizon_arc_m(&self, height_m: f64) -> f64 {
        self.radius_m * self.horizon_angle(height_m)
    }

    /// 双向可见性：两个高度不同、方向不同的观察者互相是否可见。
    pub fn mutually_visible(
        &self,
        a_dir: DVec3,
        a_height_m: f64,
        b_dir: DVec3,
        b_height_m: f64,
        clearance_m: f64,
    ) -> bool {
        let theta = ((a_dir.normalize().cross(b_dir.normalize())).length())
            .atan2(a_dir.normalize().dot(b_dir.normalize()));
        // 视线不被球面遮挡的条件：两端地平线夹角之和 ≥ 圆心角。
        let reach = self.horizon_angle(a_height_m) + self.horizon_angle(b_height_m);
        if reach + 1e-9 < theta {
            return false;
        }
        // 再叠加一层地形余量修正。
        self.bulge_m(a_dir, b_dir) <= clearance_m + self.horizon_arc_m(a_height_m.max(b_height_m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_R: f64 = 6_371_000.0;
    const OCC: CurvatureOcclusion = CurvatureOcclusion::new(EARTH_R);

    #[test]
    fn bulge_is_zero_at_zero_angle_and_grows_quadratically() {
        assert_eq!(OCC.bulge_m(DVec3::X, DVec3::X), 0.0);
        let small = OCC.bulge_m(DVec3::X, DVec3::new(1.0, 0.01, 0.0).normalize());
        let double = OCC.bulge_m(DVec3::X, DVec3::new(1.0, 0.02, 0.0).normalize());
        assert!(small > 0.0);
        assert!(
            (double / small - 4.0).abs() < 0.05,
            "小角度下隆起应随角度平方增长: {small} -> {double}"
        );
    }

    #[test]
    fn nearby_surface_points_see_each_other_but_far_ones_do_not() {
        let near = DVec3::new(1.0, 0.001, 0.0).normalize();
        // 光滑球面上两点的连线总会略微沉入地表（1 km 处约 0.8 m），
        // 所以「可见」必须带地形余量；余量为 0 时一律判为遮挡。
        assert!(
            OCC.is_blocked(DVec3::X, near, 0.0),
            "零余量下必然被球面遮挡"
        );
        assert!(!OCC.is_blocked(DVec3::X, near, 2.0), "给足余量即可见");
        assert!(
            OCC.is_blocked(DVec3::X, DVec3::Y, 0.0),
            "90° 之外必然被球体挡住"
        );
        assert!(
            !OCC.is_blocked(DVec3::X, DVec3::Y, 2.0 * EARTH_R),
            "足够大的余量应当放行"
        );
    }

    #[test]
    fn higher_observation_point_sees_further() {
        let ground = OCC.horizon_arc_m(0.0);
        let mast = OCC.horizon_arc_m(30.0);
        let orbit = OCC.horizon_arc_m(400_000.0);
        assert_eq!(ground, 0.0);
        assert!(
            mast > 15_000.0 && mast < 25_000.0,
            "30 米桅杆视线约 19.5 公里，实得 {mast}"
        );
        assert!(orbit > 2_000_000.0);
    }

    #[test]
    fn mutual_visibility_respects_heights() {
        let target = DVec3::new(1.0, 0.02, 0.0).normalize();
        assert!(!OCC.mutually_visible(DVec3::X, 0.0, target, 0.0, 0.0));
        assert!(OCC.mutually_visible(DVec3::X, 500.0, target, 500.0, 0.0));
    }
}
