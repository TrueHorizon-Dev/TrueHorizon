//! 地层采样契约。

use block_schema::BlockId;
use glam::DVec3;

/// 单个地层剖面最多容纳的层数。
///
/// 固定容量使剖面可以完全放栈上（`Copy` 语义、零分配），这对每帧数百万次的
/// 宏观 LOD 采样至关重要。超出容量时 [`StrataProfile::push`] 返回 `false`，
/// 由调用方决定是合并薄层还是丢弃——**不允许静默截断**。
pub const MAX_STRATA: usize = 32;

/// 一个地层。
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Stratum {
    /// 该层顶面的深度（米，从参考椭球面起算，向下为正）。
    pub top_m: f64,
    /// 层厚（米）。
    pub thickness_m: f64,
    /// 该层的代表方块。
    pub block: BlockId,
    /// 抗侵蚀强度（`0..=1`），参与水文侵蚀与滑坡规则。
    pub cohesion: f32,
}

impl Stratum {
    /// 底面的深度（米）。
    #[inline]
    pub fn bottom_m(&self) -> f64 {
        self.top_m + self.thickness_m
    }

    /// 该深度是否落入此层。
    #[inline]
    pub fn contains_depth(&self, depth_m: f64) -> bool {
        depth_m >= self.top_m && depth_m < self.bottom_m()
    }
}

impl Default for Stratum {
    fn default() -> Self {
        Self {
            top_m: 0.0,
            thickness_m: 0.0,
            block: BlockId::AIR,
            cohesion: 0.0,
        }
    }
}

/// 沿某个方向从地表向下的地层剖面。
///
/// 层按深度升序排列，`layers()[0].top_m` 即该方向的地表高度（负值表示海平面之上）。
#[derive(Clone, Copy, Debug)]
pub struct StrataProfile {
    layers: [Stratum; MAX_STRATA],
    len: u8,
    /// 该方向的表面高程（米，相对参考半径；正值表示凸起）。
    pub surface_height_m: f64,
}

impl StrataProfile {
    /// 空剖面。
    pub fn new(surface_height_m: f64) -> Self {
        Self {
            layers: [Stratum::default(); MAX_STRATA],
            len: 0,
            surface_height_m,
        }
    }

    /// 追加一层。容量耗尽时返回 `false` 且不写入。
    pub fn push(&mut self, stratum: Stratum) -> bool {
        if (self.len as usize) >= MAX_STRATA {
            return false;
        }
        self.layers[self.len as usize] = stratum;
        self.len += 1;
        true
    }

    /// 层数。
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// 是否没有层。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 已写入的层切片。
    #[inline]
    pub fn layers(&self) -> &[Stratum] {
        &self.layers[..self.len as usize]
    }

    /// 总厚度（米）。
    pub fn total_thickness_m(&self) -> f64 {
        self.layers().iter().map(|s| s.thickness_m).sum()
    }

    /// 按深度查询方块；超出剖面底部时返回最后一层的方块。
    pub fn block_at_depth(&self, depth_m: f64) -> BlockId {
        let mut last = BlockId::AIR;
        for layer in self.layers() {
            if layer.contains_depth(depth_m) {
                return layer.block;
            }
            last = layer.block;
        }
        last
    }
}

/// 地层采样器：地质层对外暴露的唯一入口。
///
/// 实现必须是纯函数式的：同一个 `dir` 在任何时刻、任何线程都返回同一剖面
/// （随机性只能来自世界种子，不能来自运行期状态）。
pub trait StratumSampler: Send + Sync {
    /// 采样某个方向的完整地层剖面。
    fn strata_at(&self, dir: DVec3) -> StrataProfile;

    /// 只取表面高度（米，相对参考半径）。宏观四叉树 LOD 只依赖它，故单独开一个
    /// 廉价入口，允许实现走缓存而不构建完整剖面。
    fn surface_height_m(&self, dir: DVec3) -> f64;

    /// 取某个方向的某个深度处的方块。默认走剖面查询。
    fn block_at_depth(&self, dir: DVec3, depth_m: f64) -> BlockId {
        self.strata_at(dir).block_at_depth(depth_m)
    }

    /// 该方向是否在海洋覆盖之下（深度为负即高于海平面）。
    fn is_submerged(&self, dir: DVec3, sea_level_m: f64) -> bool {
        self.surface_height_m(dir) < sea_level_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_schema::material::BlockPhysics;

    fn rock() -> BlockId {
        let _ = BlockPhysics::ROCK;
        BlockId::new(1)
    }

    #[test]
    fn push_is_bounded_and_never_silently_truncates() {
        let mut profile = StrataProfile::new(0.0);
        for i in 0..MAX_STRATA {
            assert!(profile.push(Stratum {
                top_m: i as f64,
                thickness_m: 1.0,
                block: rock(),
                cohesion: 0.5,
            }));
        }
        assert_eq!(profile.len(), MAX_STRATA);
        assert!(!profile.push(Stratum::default()), "容量耗尽必须显式失败");
        assert_eq!(profile.len(), MAX_STRATA);
    }

    #[test]
    fn block_lookup_walks_layers_in_order() {
        let mut profile = StrataProfile::new(10.0);
        profile.push(Stratum {
            top_m: 0.0,
            thickness_m: 2.0,
            block: BlockId::new(1),
            cohesion: 0.4,
        });
        profile.push(Stratum {
            top_m: 2.0,
            thickness_m: 3.0,
            block: BlockId::new(2),
            cohesion: 0.9,
        });
        assert_eq!(profile.total_thickness_m(), 5.0);
        assert_eq!(profile.block_at_depth(1.5), BlockId::new(1));
        assert_eq!(profile.block_at_depth(2.0), BlockId::new(2), "边界归下一层");
        assert_eq!(profile.block_at_depth(4.999), BlockId::new(2));
        assert_eq!(
            profile.block_at_depth(999.0),
            BlockId::new(2),
            "底部之下取最底层"
        );
    }
}
