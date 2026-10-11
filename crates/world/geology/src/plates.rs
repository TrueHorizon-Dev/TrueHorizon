//! 板块构造：球面 Voronoi 划分与漂移场。

use crate::noise;
use glam::DVec3;

/// 板块标识。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct PlateId(pub u16);

/// 一个板块的种子站点。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlateSite {
    /// 板块 ID。
    pub id: PlateId,
    /// 站点方向（单位向量）。
    pub dir: DVec3,
    /// 漂移速度方向（单位向量，切向）。
    pub drift: DVec3,
    /// 漂移速率（米/百万年，量级参考真实板块 10 ~ 100 mm/yr）。
    pub drift_rate_mm_per_year: f64,
    /// 板块基准高程（米，正为陆地、负为洋盆）。
    pub base_elevation_m: f64,
    /// 是否为大陆性板块（决定碰撞时是造山还是俯冲）。
    pub continental: bool,
}

/// 球面板块场（Voronoi 划分）。
///
/// 站点用 Fibonacci 球面均匀撒点，再由哈希决定各板块属性；查询走朴素最近邻
/// （板块数在 12 ~ 64 之间，单次查询约 1 µs，无需层级索引）。
#[derive(Clone, Debug)]
pub struct PlateField {
    seed: u64,
    sites: Vec<PlateSite>,
}

impl PlateField {
    /// 以种子与板块数构造。
    ///
    /// `count` 会被夹到 `[1, 4096]`，避免调用方传入 0 或病态数量。
    pub fn new(seed: u64, count: u16) -> Self {
        let count = count.clamp(1, 4096) as u64;
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        let mut sites = Vec::with_capacity(count as usize);

        for index in 0..count {
            // Fibonacci 球面：沿 Y 轴等分，方位角按黄金角旋转 → 面积近似均匀。
            let y = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
            let radius = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden_angle * index as f64;
            let mut dir = DVec3::new(radius * theta.cos(), y, radius * theta.sin());

            // 轻微扰动，避免极端规整的人工感（扰动幅度 < 5°，不破坏均匀性）。
            let jitter = noise::hash_unit(seed, index as i64, 0, 0) - 0.5;
            let spin = glam::DQuat::from_axis_angle(DVec3::Y, jitter * 0.08);
            dir = (spin * dir).normalize();

            let elevation_roll = noise::hash_unit(seed, index as i64, 1, 0);
            let continental = elevation_roll > 0.45;
            let base_elevation_m = if continental {
                // 大陆：+200 m ~ +1200 m。
                200.0 + 1000.0 * noise::hash_unit(seed, index as i64, 2, 0)
            } else {
                // 洋盆：-6000 m ~ -2500 m。
                -6000.0 + 3500.0 * noise::hash_unit(seed, index as i64, 3, 0)
            };

            // 漂移方向取站点处的切向。
            let (east, north) = planet_coords::SphereGeometry::tangent_frame(dir);
            let azimuth = noise::hash_unit(seed, index as i64, 4, 0) * std::f64::consts::TAU;
            let drift = (east * azimuth.cos() + north * azimuth.sin()).normalize();

            sites.push(PlateSite {
                id: PlateId(index as u16),
                dir,
                drift,
                drift_rate_mm_per_year: 10.0 + 90.0 * noise::hash_unit(seed, index as i64, 5, 0),
                base_elevation_m,
                continental,
            });
        }

        Self { seed, sites }
    }

    /// 种子。
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// 板块数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// 是否没有板块（构造时已排除，仅用于满足 lint 惯例）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// 全部站点。
    #[inline]
    pub fn sites(&self) -> &[PlateSite] {
        &self.sites
    }

    /// 按 ID 取站点。
    pub fn site(&self, id: PlateId) -> Option<&PlateSite> {
        self.sites.get(id.0 as usize)
    }

    /// 最近站点及其角距（弧度）。
    pub fn nearest(&self, dir: DVec3) -> (&PlateSite, f64) {
        let dir = dir.normalize();
        let mut best = &self.sites[0];
        let mut best_angle = f64::MAX;
        for site in &self.sites {
            let angle = (dir.cross(site.dir)).length().atan2(dir.dot(site.dir));
            if angle < best_angle {
                best_angle = angle;
                best = site;
            }
        }
        (best, best_angle)
    }

    /// 到最近与次近站点角距之差，归一化为 **[0, 1]** 的「板块内部度」。
    ///
    /// * `1.0` = 深居板块内部（远离任何边界）；
    /// * `0.0` = 恰在边界上（造山、裂谷、火山的高发区）。
    pub fn interior_factor(&self, dir: DVec3) -> f64 {
        let dir = dir.normalize();
        let mut first = f64::MAX;
        let mut second = f64::MAX;
        for site in &self.sites {
            let angle = (dir.cross(site.dir)).length().atan2(dir.dot(site.dir));
            if angle < first {
                second = first;
                first = angle;
            } else if angle < second {
                second = angle;
            }
        }
        if second <= f64::MIN_POSITIVE || !second.is_finite() {
            return 1.0;
        }
        ((second - first) / second).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sites_are_unit_vectors_and_ids_are_contiguous() {
        let field = PlateField::new(2026, 24);
        assert_eq!(field.len(), 24);
        for (index, site) in field.sites().iter().enumerate() {
            assert!((site.dir.length() - 1.0).abs() < 1e-12);
            assert!((site.drift.length() - 1.0).abs() < 1e-12);
            assert!(site.drift.dot(site.dir).abs() < 1e-9, "漂移必须是切向");
            assert_eq!(site.id, PlateId(index as u16));
            assert_eq!(field.site(site.id), Some(site));
        }
    }

    #[test]
    fn nearest_is_deterministic_and_matches_brute_force() {
        let field = PlateField::new(7, 16);
        let probe = DVec3::new(0.3, -0.5, 0.81).normalize();
        let (site, angle) = field.nearest(probe);
        assert!((0.0..=std::f64::consts::PI).contains(&angle));
        assert_eq!(site.dir, field.nearest(probe).0.dir);
        // 最近邻角距不得超过均匀分布下任一站点的角距。
        for other in field.sites() {
            let other_angle = (probe.cross(other.dir))
                .length()
                .atan2(probe.dot(other.dir));
            assert!(angle <= other_angle + 1e-12);
        }
    }

    #[test]
    fn interior_factor_is_one_on_a_site_and_drops_towards_the_boundary() {
        let field = PlateField::new(11, 32);
        let site = field.sites()[3];
        // 站点就是最近邻本身（d1 = 0），因此「板块内部度」达到最大值 1。
        let on_site = field.interior_factor(site.dir);
        assert!(on_site > 0.9, "站点应深居板块内部，实得 {on_site}");

        // 边界检验改用只有两个板块的场：中点与两个站点严格等距，理论值就是 0。
        // （多站点场里可能有第三个站点更近，那样的点并不是真正的边界。）
        let pair = PlateField::new(11, 2);
        let midpoint = (pair.sites()[0].dir + pair.sites()[1].dir).normalize();
        let near_boundary = pair.interior_factor(midpoint);
        assert!(near_boundary < 1e-9, "等距点即边界，实得 {near_boundary}");

        // 任何方向的内部度都必须落在 [0, 1]。
        for probe in [site.dir, midpoint, DVec3::X, -DVec3::Y] {
            let factor = field.interior_factor(probe);
            assert!((0.0..=1.0).contains(&factor), "内部度越界: {factor}");
        }
    }

    #[test]
    fn single_plate_field_covers_everything() {
        let field = PlateField::new(1, 1);
        assert_eq!(field.len(), 1);
        assert_eq!(field.interior_factor(DVec3::X), 1.0);
        // 数量夹取：0 会被提升为 1。
        assert_eq!(PlateField::new(1, 0).len(), 1);
    }
}
