//! 连续地层剖面：把板块场与噪声转成 `world-contracts` 的采样契约。

use block_schema::BlockId;
use glam::DVec3;
use world_contracts::{StrataProfile, Stratum, StratumSampler};

use crate::noise;
use crate::plates::PlateField;

/// 地质层使用的方块 ID。
///
/// **必须与 `block-schema` 发布的注册表一致**；不一致时应在启动自检中失败
/// （见 `apps/world-tool diagnose`）。之所以不直接持有 `BlockRegistry`：地质层
/// 需要 `Send + Sync` 且零锁，只传 ID 最省事。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeologyBlocks {
    /// 表层土壤。
    pub topsoil: BlockId,
    /// 沉积层（含水下沙层）。
    pub sediment: BlockId,
    /// 基岩。
    pub rock: BlockId,
    /// 地幔（高压变质岩）。
    pub mantle: BlockId,
}

impl GeologyBlocks {
    /// 引擎默认映射（0 = 空气，1 = 岩石，2 = 土壤，3 = 沙，5 = 地幔）。
    pub const fn canonical() -> Self {
        Self {
            topsoil: BlockId::new(2),
            sediment: BlockId::new(3),
            rock: BlockId::new(1),
            mantle: BlockId::new(5),
        }
    }
}

impl Default for GeologyBlocks {
    fn default() -> Self {
        Self::canonical()
    }
}

/// 地质生成参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeologyConfig {
    /// 世界种子。**决定整颗行星**，一经发布不可更改。
    pub seed: u64,
    /// 板块数量。
    pub plate_count: u16,
    /// 海平面高程（米，相对参考半径）。
    pub sea_level_m: f64,
    /// 地壳厚度（米）；超过此深度进入地幔。
    pub crust_depth_m: f64,
    /// 表层土壤厚度（米）。
    pub topsoil_m: f64,
    /// 沉积层厚度（米）。
    pub sediment_m: f64,
    /// 细节起伏振幅（米）。
    pub detail_amplitude_m: f64,
    /// 板块边界的造山抬升上限（米）。
    pub boundary_uplift_m: f64,
    /// 噪声倍频数。
    pub octaves: u32,
}

impl Default for GeologyConfig {
    fn default() -> Self {
        Self {
            seed: 0x7A0E_2026,
            plate_count: 24,
            sea_level_m: 0.0,
            // 8 km：约等于近景 8 个 32³ Chunk 的深度，够用且不浪费剖面层数。
            crust_depth_m: 8_192.0,
            topsoil_m: 1.0,
            sediment_m: 12.0,
            detail_amplitude_m: 320.0,
            boundary_uplift_m: 4_800.0,
            octaves: 5,
        }
    }
}

/// 程序化地质生成器。
///
/// 纯函数式：所有输出只由 [`GeologyConfig::seed`] 与查询方向决定，因此
/// **同一世界在任何机器、任何线程上都完全一致**（这是多人联机与存档兼容的前提）。
#[derive(Clone, Debug)]
pub struct GeologyGenerator {
    config: GeologyConfig,
    plates: PlateField,
    blocks: GeologyBlocks,
}

impl GeologyGenerator {
    /// 以配置构造（默认方块映射）。
    pub fn new(config: GeologyConfig) -> Self {
        Self::with_blocks(config, GeologyBlocks::canonical())
    }

    /// 以配置与方块映射构造。
    pub fn with_blocks(config: GeologyConfig, blocks: GeologyBlocks) -> Self {
        Self {
            plates: PlateField::new(config.seed, config.plate_count),
            config,
            blocks,
        }
    }

    /// 配置。
    #[inline]
    pub fn config(&self) -> &GeologyConfig {
        &self.config
    }

    /// 板块场。
    #[inline]
    pub fn plates(&self) -> &PlateField {
        &self.plates
    }

    /// 表面高程（米，相对参考半径，正为凸起）。
    ///
    /// 组成：`板块基准高程 + 边界造山抬升 + 分形细节`。
    pub fn elevation_m(&self, dir: DVec3) -> f64 {
        let dir = dir.normalize();
        let (site, _) = self.plates.nearest(dir);
        let interior = self.plates.interior_factor(dir);

        // 大陆与大陆相撞 → 造山；越靠近边界抬升越强（二次衰减）。
        let ridge = if site.continental {
            (1.0 - interior).powi(2) * self.config.boundary_uplift_m
        } else {
            0.0
        };

        let p = [dir.x * 6.0, dir.y * 6.0, dir.z * 6.0];
        let detail = (noise::fbm(self.config.seed, p, self.config.octaves, 2.0, 0.5) - 0.5)
            * 2.0
            * self.config.detail_amplitude_m;

        site.base_elevation_m + ridge + detail
    }

    /// 该方向是否被海洋覆盖。
    pub fn is_submerged(&self, dir: DVec3) -> bool {
        self.elevation_m(dir) < self.config.sea_level_m
    }

    /// 是否大陆性板块。
    pub fn is_continental(&self, dir: DVec3) -> bool {
        self.plates.nearest(dir).0.continental
    }

    /// 构造该方向的地层剖面。
    pub fn build_profile(&self, dir: DVec3) -> StrataProfile {
        let elevation = self.elevation_m(dir);
        let mut profile = StrataProfile::new(elevation);
        let submerged = elevation < self.config.sea_level_m;

        let topsoil = if submerged {
            self.blocks.sediment
        } else {
            self.blocks.topsoil
        };
        let sediment = if submerged {
            self.blocks.sediment
        } else {
            self.blocks.topsoil
        };

        let layers = [
            Stratum {
                top_m: 0.0,
                thickness_m: self.config.topsoil_m,
                block: topsoil,
                cohesion: if submerged { 0.15 } else { 0.45 },
            },
            Stratum {
                top_m: self.config.topsoil_m,
                thickness_m: self.config.sediment_m,
                block: sediment,
                cohesion: 0.55,
            },
            Stratum {
                top_m: self.config.topsoil_m + self.config.sediment_m,
                thickness_m: (self.config.crust_depth_m
                    - self.config.topsoil_m
                    - self.config.sediment_m)
                    .max(0.0),
                block: self.blocks.rock,
                cohesion: 0.9,
            },
            Stratum {
                top_m: self.config.crust_depth_m,
                thickness_m: 2_048.0,
                block: self.blocks.mantle,
                cohesion: 1.0,
            },
        ];
        for layer in layers {
            if !profile.push(layer) {
                // 剖面容量耗尽属于配置错误：宁可少写一层，也不静默错位。
                break;
            }
        }
        profile
    }
}

impl StratumSampler for GeologyGenerator {
    fn strata_at(&self, dir: DVec3) -> StrataProfile {
        self.build_profile(dir)
    }

    fn surface_height_m(&self, dir: DVec3) -> f64 {
        self.elevation_m(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator() -> GeologyGenerator {
        GeologyGenerator::new(GeologyConfig {
            seed: 4242,
            ..GeologyConfig::default()
        })
    }

    #[test]
    fn generation_is_deterministic_across_instances() {
        let a = generator();
        let b = generator();
        for dir in [
            DVec3::X,
            DVec3::new(0.3, -0.71, 0.4).normalize(),
            DVec3::new(-0.5, 0.5, -0.7).normalize(),
        ] {
            assert_eq!(a.elevation_m(dir), b.elevation_m(dir));
            assert_eq!(a.strata_at(dir).layers(), b.strata_at(dir).layers());
        }
    }

    #[test]
    fn strata_are_contiguous_and_ordered_by_depth() {
        let geo = generator();
        let profile = geo.build_profile(DVec3::new(0.2, 0.9, 0.1).normalize());
        let layers = profile.layers();
        assert!(layers.len() >= 4, "至少应有表土/沉积/基岩/地幔四层");
        assert_eq!(layers[0].top_m, 0.0, "剖面从地表开始");
        for pair in layers.windows(2) {
            assert!(
                (pair[0].bottom_m() - pair[1].top_m).abs() < 1e-9,
                "地层必须首尾相接"
            );
            assert!(pair[0].thickness_m > 0.0);
        }
        assert_eq!(
            layers.last().unwrap().block,
            GeologyBlocks::canonical().mantle
        );
    }

    #[test]
    fn surface_height_matches_profile_and_trait() {
        let geo = generator();
        let dir = DVec3::new(-0.4, 0.2, 0.9).normalize();
        let profile = geo.strata_at(dir);
        assert_eq!(profile.surface_height_m, geo.surface_height_m(dir));
        assert_eq!(profile.layers()[0].top_m, 0.0);
    }

    #[test]
    fn submerged_columns_start_with_sediment() {
        let geo = generator();
        // 找一个洋盆方向。
        let mut found = false;
        for i in 0..64 {
            let theta = i as f64 / 64.0 * std::f64::consts::TAU;
            let dir = DVec3::new(theta.cos(), 0.1, theta.sin()).normalize();
            if geo.is_submerged(dir) {
                let profile = geo.build_profile(dir);
                assert_eq!(
                    profile.layers()[0].block,
                    GeologyBlocks::canonical().sediment
                );
                found = true;
                break;
            }
        }
        assert!(found, "种子 4242 应当至少存在一个洋盆方向");
    }

    #[test]
    fn continental_plates_sit_above_oceanic_ones_on_average() {
        let geo = generator();
        let mut continental = Vec::new();
        let mut oceanic = Vec::new();
        for site in geo.plates().sites() {
            let sample = geo.elevation_m(site.dir);
            if site.continental {
                continental.push(sample);
            } else {
                oceanic.push(sample);
            }
        }
        if !continental.is_empty() && !oceanic.is_empty() {
            let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
            assert!(mean(&continental) > mean(&oceanic));
        }
    }
}
