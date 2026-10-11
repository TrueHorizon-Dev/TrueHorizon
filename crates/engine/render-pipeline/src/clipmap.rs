//! 球面 Clipmap：LOD 环形分块。
//!
//! 近景（0 ~ [`SphereClipmapConfig::near_field_radius_m`]）由 `world-meshing` 的
//! 实体素网格负责；本模块负责其外的宏观呈现，环带按 2 的幂次扩张：
//!
//! ```text
//! level 0: [256 m,   512 m)   patch 边长 32 m
//! level 1: [512 m,  1024 m)   patch 边长 64 m
//! level 2: [  1 km,   2 km)   patch 边长 128 m
//! ...
//! ```
//!
//! 每层环带的径向宽度等于其内径，横向被均分为 `patches_across_ring` 块，因此
//! **patch 数量随层数线性增长而非平方增长**，几何总量对深度不敏感。
//!
//! 已知限制（骨架阶段）：
//!
//! * 只覆盖相机所在的主导面，跨面接缝的补丁尚未生成（`TODO(clipmap): 跨面缝合`）；
//! * 尚未做 patch 级视锥剔除（候选筛选已按弧长与地平线收敛，量级可控）。

use glam::DVec3;
use planet_coords::{CubeFace, SphereGeometry};

/// Patch 标识：`(面, 层, 面内块坐标)`。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PatchKey {
    /// 所属面。
    pub face: CubeFace,
    /// LOD 层。
    pub level: u32,
    /// 面内块 X 索引（沿 u 方向）。
    pub x: u32,
    /// 面内块 Y 索引（沿 v 方向）。
    pub y: u32,
}

/// Clipmap 配置。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereClipmapConfig {
    /// 最大层数。层数上限决定了宏观可视距离（第 `l` 层外径为
    /// `near_field_radius_m * 2^(l+1)`）。
    pub levels: u32,
    /// 近景体素网格的半径（米），即 Clipmap 的起始内径。
    pub near_field_radius_m: f64,
    /// 每层环带横向被均分的块数。
    pub patches_across_ring: u32,
    /// 单次选择允许返回的 patch 上限，防止极端视角下爆量。
    pub max_patches: usize,
}

impl Default for SphereClipmapConfig {
    fn default() -> Self {
        Self {
            levels: 16,
            near_field_radius_m: 256.0,
            patches_across_ring: 8,
            max_patches: 4096,
        }
    }
}

impl SphereClipmapConfig {
    /// 第 `level` 层的环带内径（米）。
    #[inline]
    pub fn ring_inner_m(&self, level: u32) -> f64 {
        self.near_field_radius_m * (1u64 << level.min(62)) as f64
    }

    /// 第 `level` 层的环带外径（米）。
    #[inline]
    pub fn ring_outer_m(&self, level: u32) -> f64 {
        self.ring_inner_m(level) * 2.0
    }

    /// 第 `level` 层的 patch 边长（米）。
    #[inline]
    pub fn patch_edge_m(&self, level: u32) -> f64 {
        self.ring_inner_m(level) / self.patches_across_ring.max(1) as f64
    }
}

/// 球面 Clipmap 选择器。
#[derive(Clone, Debug)]
pub struct SphereClipmap {
    config: SphereClipmapConfig,
    geometry: SphereGeometry,
}

impl SphereClipmap {
    /// 以配置与行星半径构造。
    pub fn new(config: SphereClipmapConfig, radius_m: f64) -> Self {
        Self {
            config,
            geometry: SphereGeometry::new(radius_m),
        }
    }

    /// 配置。
    #[inline]
    pub fn config(&self) -> &SphereClipmapConfig {
        &self.config
    }

    /// 行星几何。
    #[inline]
    pub fn geometry(&self) -> &SphereGeometry {
        &self.geometry
    }

    /// 第 `level` 层每个面的分块数。面心处弧长约为 `π/2 · R`。
    pub fn patches_per_face(&self, level: u32) -> u32 {
        let face_arc = self.geometry.radius_m * std::f64::consts::FRAC_PI_2;
        let edge = self.config.patch_edge_m(level).max(1e-3);
        ((face_arc / edge).round() as u64).clamp(1, u32::MAX as u64) as u32
    }

    /// 由可视距离反推需要的最大层号（不裁剪到 `config.levels`）。
    pub fn level_for_distance_m(&self, distance_m: f64) -> u32 {
        if distance_m <= self.config.near_field_radius_m {
            return 0;
        }
        let ratio = distance_m / self.config.near_field_radius_m;
        (ratio.log2().ceil() as i64 - 1).clamp(0, i64::from(u32::MAX)) as u32
    }

    /// patch 中心方向。
    pub fn patch_center_dir(&self, key: PatchKey) -> DVec3 {
        let per_face = self.patches_per_face(key.level) as f64;
        let u = (key.x as f64 + 0.5) / per_face * 2.0 - 1.0;
        let v = (key.y as f64 + 0.5) / per_face * 2.0 - 1.0;
        key.face.uv_to_dir(u, v)
    }

    /// patch 近似世界坐标（米），用于实例化与包围盒计算。
    pub fn patch_center_world_m(&self, key: PatchKey, elevation_m: f64) -> DVec3 {
        self.patch_center_dir(key) * (self.geometry.radius_m + elevation_m)
    }

    /// 选择注视点周围需要绘制的全部 patch。
    ///
    /// * `camera_dir`：视线中心的球面方向（会自动归一化）；
    /// * `horizon_arc_m`：当前观察高度对应的地平线弧长——超出它的一切 patch 都被
    ///   行星自身曲率遮挡，直接剔除（见 `planet-coords::CurvatureOcclusion`）。
    ///
    /// 返回结果按层号升序排列，便于调用方按 LOD 分流到不同管线。
    pub fn select(&self, camera_dir: DVec3, horizon_arc_m: f64) -> Vec<PatchKey> {
        let camera_dir = camera_dir.normalize();
        let (face, cam_u, cam_v) = CubeFace::dir_to_uv(camera_dir);
        let window = self.config.patches_across_ring as i64;
        let mut out = Vec::new();

        for level in 0..self.config.levels {
            let inner = self.config.ring_inner_m(level);
            if inner >= horizon_arc_m {
                break;
            }
            let outer = self.config.ring_outer_m(level).min(horizon_arc_m);
            let per_face = self.patches_per_face(level) as i64;
            let center_x = uv_to_index(cam_u, per_face);
            let center_y = uv_to_index(cam_v, per_face);

            for dy in -window..=window {
                for dx in -window..=window {
                    let x = center_x + dx;
                    let y = center_y + dy;
                    if x < 0 || y < 0 || x >= per_face || y >= per_face {
                        continue;
                    }
                    let key = PatchKey {
                        face,
                        level,
                        x: x as u32,
                        y: y as u32,
                    };
                    let distance = self
                        .geometry
                        .arc_len(camera_dir, self.patch_center_dir(key));
                    if distance >= inner && distance < outer {
                        out.push(key);
                        if out.len() >= self.config.max_patches {
                            return out;
                        }
                    }
                }
            }
        }
        out
    }
}

/// 面坐标 → 分块索引。
fn uv_to_index(u: f64, per_face: i64) -> i64 {
    let normalized = ((u + 1.0) * 0.5 * per_face as f64).floor() as i64;
    normalized.clamp(0, per_face - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_R: f64 = 6_371_000.0;

    fn clipmap() -> SphereClipmap {
        SphereClipmap::new(SphereClipmapConfig::default(), EARTH_R)
    }

    #[test]
    fn patch_edge_and_ring_double_each_level() {
        let config = SphereClipmapConfig::default();
        for level in 0..8 {
            assert_eq!(config.ring_inner_m(level + 1), config.ring_outer_m(level));
            assert_eq!(
                config.patch_edge_m(level + 1),
                2.0 * config.patch_edge_m(level)
            );
        }
        assert_eq!(config.ring_inner_m(0), 256.0);
        assert_eq!(config.patch_edge_m(0), 32.0, "内径 256m 均分 8 块");
    }

    #[test]
    fn level_for_distance_is_monotonic_and_clamped_at_zero() {
        let clip = clipmap();
        assert_eq!(clip.level_for_distance_m(0.0), 0);
        assert_eq!(clip.level_for_distance_m(256.0), 0);
        assert_eq!(clip.level_for_distance_m(300.0), 0);
        assert_eq!(clip.level_for_distance_m(600.0), 1);
        assert_eq!(clip.level_for_distance_m(1200.0), 2);
        let mut last = 0;
        for distance in [100.0, 1_000.0, 10_000.0, 1e6, 1e8] {
            let level = clip.level_for_distance_m(distance);
            assert!(level >= last);
            last = level;
        }
    }

    #[test]
    fn selection_respects_near_field_and_horizon() {
        let clip = clipmap();
        let camera = DVec3::new(0.4, 0.6, 0.7).normalize();
        let horizon = 30_000.0;
        let patches = clip.select(camera, horizon);
        assert!(!patches.is_empty(), "地平线内必须有可见 patch");
        assert!(patches.len() <= clip.config().max_patches);
        for key in &patches {
            let distance = clip.geometry().arc_len(camera, clip.patch_center_dir(*key));
            assert!(
                distance >= clip.config().ring_inner_m(key.level),
                "patch 落在环带内径以内: level {} 距离 {distance}",
                key.level
            );
            assert!(distance < horizon, "patch 超出地平线: {distance}");
            assert!(distance < clip.config().ring_outer_m(key.level));
        }
        // 层号升序，便于按 LOD 分流。
        assert!(patches.windows(2).all(|w| w[0].level <= w[1].level));
    }

    #[test]
    fn ground_observer_sees_only_nearest_ring() {
        let clip = clipmap();
        // 海平面观察者：地平线为 0，任何环带都不应生成。
        assert!(clip.select(DVec3::X, 0.0).is_empty());
        // 30 米桅杆：地平线约 19.5 km，应覆盖前若干层而不会触及 32 km 以上。
        let horizon = clip.geometry().horizon_arc_m(30.0);
        let patches = clip.select(DVec3::X, horizon);
        assert!(!patches.is_empty());
        assert!(patches
            .iter()
            .all(|k| clip.config().ring_inner_m(k.level) < horizon));
    }

    #[test]
    fn patch_centers_stay_inside_their_face() {
        let clip = clipmap();
        for level in [0, 3, 7] {
            let per_face = clip.patches_per_face(level);
            for key in [
                PatchKey {
                    face: CubeFace::PosY,
                    level,
                    x: 0,
                    y: 0,
                },
                PatchKey {
                    face: CubeFace::PosY,
                    level,
                    x: per_face - 1,
                    y: per_face - 1,
                },
            ] {
                let dir = clip.patch_center_dir(key);
                assert!((dir.length() - 1.0).abs() < 1e-12);
                assert!(dir.dot(CubeFace::PosY.normal()) > 0.0);
            }
        }
    }

    #[test]
    fn max_patches_budget_is_enforced() {
        let config = SphereClipmapConfig {
            levels: 16,
            max_patches: 10,
            ..SphereClipmapConfig::default()
        };
        let clip = SphereClipmap::new(config, EARTH_R);
        let patches = clip.select(DVec3::new(1.0, 0.2, 0.3).normalize(), 4.0e6);
        assert_eq!(patches.len(), 10);
    }
}
