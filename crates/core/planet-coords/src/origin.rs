//! 双精度浮动原点。

use glam::{DVec3, Vec3};

/// 浮动原点（Floating Origin）状态。
///
/// 世界状态**只以 `f64` 存在**；渲染与物理求解需要 `f32` 时，一律先减去当前原点：
///
/// * 世界坐标 = `origin + render`，`f64` 精度，覆盖整个行星；
/// * 渲染坐标 = `world - origin`，`f32` 精度，量级被限制在相机附近数公里内，
///   从而把 32 位浮点的抖动压制到亚毫米级。
///
/// 当相机离原点超过阈值时调用 [`FloatingOrigin::recenter`] 搬移原点：
/// 世界内容在视觉上纹丝不动，只是数值表示整体平移。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FloatingOrigin {
    offset_m: DVec3,
}

impl FloatingOrigin {
    /// 以给定世界偏移作为原点。
    pub const fn new(offset_m: DVec3) -> Self {
        Self { offset_m }
    }

    /// 当前原点在世界坐标系中的位置（米）。
    #[inline]
    pub const fn offset_m(self) -> DVec3 {
        self.offset_m
    }

    /// 世界坐标 → 渲染坐标（`f32`，相对当前原点）。
    #[inline]
    pub fn to_render(self, world_m: DVec3) -> Vec3 {
        (world_m - self.offset_m).as_vec3()
    }

    /// 渲染坐标 → 世界坐标。
    #[inline]
    pub fn to_world(self, render: Vec3) -> DVec3 {
        self.offset_m + render.as_dvec3()
    }

    /// 若相机离原点超过 `threshold_m`，把原点整体搬到相机处并返回 `true`。
    ///
    /// 返回 `true` 时，调用方必须同步刷新所有已缓存的渲染侧位置（GPU 实例缓冲等）。
    pub fn recenter(&mut self, camera_world_m: DVec3, threshold_m: f64) -> bool {
        debug_assert!(threshold_m >= 0.0, "阈值必须非负");
        if (camera_world_m - self.offset_m).length_squared() > threshold_m * threshold_m {
            self.offset_m = camera_world_m;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_render_round_trip_keeps_f64_precision() {
        let origin = FloatingOrigin::new(DVec3::new(6_000_000.0, 1_000_000.0, -2_000_000.0));
        let world = DVec3::new(6_000_123.456, 1_000_777.25, -2_000_111.5);
        let render = origin.to_render(world);
        // 渲染侧量大被压到 10 公里以内，f32 误差应在毫米以下。
        assert!(render.length() < 10_000.0);
        let back = origin.to_world(render);
        assert!(
            (back - world).length() < 1e-3,
            "往返误差 {:?}",
            (back - world).length()
        );
    }

    #[test]
    fn recenter_only_fires_beyond_threshold() {
        let mut origin = FloatingOrigin::default();
        let threshold = 4096.0;
        assert!(!origin.recenter(DVec3::new(1000.0, 0.0, 0.0), threshold));
        assert_eq!(origin.offset_m(), DVec3::ZERO);
        assert!(origin.recenter(DVec3::new(9000.0, 0.0, 0.0), threshold));
        assert_eq!(origin.offset_m(), DVec3::new(9000.0, 0.0, 0.0));
    }
}
