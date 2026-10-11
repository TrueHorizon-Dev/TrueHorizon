//! 球面重力场。

use glam::DVec3;

/// 球对称重力模型（足够精确：真实行星的 J2 扁率项对 1 米体素无可见影响）。
///
/// * 表面重力加速度：`g0 = μ / R²`；
/// * 高度 `h` 处：`g(h) = g0 · (R / (R + h))²`；
/// * 方向：永远指向质心（即 `-dir`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalGravity {
    /// 行星半径（米）。
    pub radius_m: f64,
    /// 表面重力加速度（m/s²）。地球取 9.80665。
    pub surface_g: f64,
}

impl SphericalGravity {
    /// 构造。
    pub const fn new(radius_m: f64, surface_g: f64) -> Self {
        Self {
            radius_m,
            surface_g,
        }
    }

    /// 地球参数。
    pub const fn earth() -> Self {
        Self::new(6_371_000.0, 9.806_65)
    }

    /// 引力常数与质量的乘积 `μ = g0 · R²`（m³/s²）。
    pub fn mu(&self) -> f64 {
        self.surface_g * self.radius_m * self.radius_m
    }

    /// 高度 `height_m` 处的重力加速度大小。
    pub fn magnitude_at_height_m(&self, height_m: f64) -> f64 {
        let r = self.radius_m + height_m;
        if r <= 0.0 {
            return 0.0;
        }
        self.mu() / (r * r)
    }

    /// 给定球面方向的表面重力向量（指向质心）。
    pub fn at_dir(&self, dir: DVec3) -> DVec3 {
        -dir.normalize() * self.surface_g
    }

    /// 给定**世界坐标**处的重力向量（自动按高度衰减）。
    pub fn at_position(&self, position_m: DVec3) -> DVec3 {
        let distance = position_m.length();
        if distance <= f64::MIN_POSITIVE {
            return DVec3::ZERO;
        }
        let magnitude = self.mu() / (distance * distance);
        -position_m / distance * magnitude
    }

    /// 环绕速度（米/秒）——`v = sqrt(μ / r)`。
    pub fn orbital_speed_m_s(&self, radius_m: f64) -> f64 {
        if radius_m <= 0.0 {
            return 0.0;
        }
        (self.mu() / radius_m).sqrt()
    }

    /// 逃逸速度（米/秒）——`v = sqrt(2μ / r)`。
    pub fn escape_speed_m_s(&self, radius_m: f64) -> f64 {
        self.orbital_speed_m_s(radius_m) * 2.0f64.sqrt()
    }
}

impl Default for SphericalGravity {
    fn default() -> Self {
        Self::earth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_magnitude_matches_configured_g() {
        let gravity = SphericalGravity::earth();
        assert!((gravity.magnitude_at_height_m(0.0) - 9.806_65).abs() < 1e-9);
        assert_eq!(gravity.at_dir(DVec3::Y), DVec3::new(0.0, -9.806_65, 0.0));
    }

    #[test]
    fn gravity_falls_off_with_height() {
        let gravity = SphericalGravity::earth();
        let surface = gravity.magnitude_at_height_m(0.0);
        let high = gravity.magnitude_at_height_m(400_000.0);
        assert!(high < surface);
        // 400 km 处约为表面的 89%。
        assert!(
            (high / surface - 0.887).abs() < 0.01,
            "比值 {}",
            high / surface
        );
    }

    #[test]
    fn position_vector_points_to_center() {
        let gravity = SphericalGravity::earth();
        let position = DVec3::new(6_371_000.0, 0.0, 0.0);
        let acceleration = gravity.at_position(position);
        assert!(acceleration.x < 0.0 && acceleration.y == 0.0 && acceleration.z == 0.0);
        assert!((acceleration.length() - 9.806_65).abs() < 1e-9);
        assert_eq!(gravity.at_position(DVec3::ZERO), DVec3::ZERO);
    }

    #[test]
    fn orbital_and_escape_speeds_are_physical() {
        let gravity = SphericalGravity::earth();
        let low_orbit = gravity.orbital_speed_m_s(6_771_000.0);
        assert!(
            (low_orbit - 7_672.0).abs() < 30.0,
            "近地轨道速度 {low_orbit}"
        );
        assert!(gravity.escape_speed_m_s(6_371_000.0) > 11_000.0);
        assert_eq!(gravity.orbital_speed_m_s(0.0), 0.0);
    }
}
