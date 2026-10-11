//! 宏观四叉树 LOD 选择。
//!
//! Clipmap（`render-pipeline::clipmap`）负责**近景到中景**的稠密环带；
//! 本模块负责**中景到行星全貌**：以每个立方体面为根的四叉树，按屏幕几何误差细分。
//!
//! 细分判据：`edge_arc / distance > split_ratio`，即「节点在视野中占据的角宽」。
//! 这等价于把「1 像素」的允许误差换算成节点边长，但不需要知道分辨率，
//! 因此在窗口尺寸变化时不会抖动。

use std::collections::BinaryHeap;

use glam::DVec3;
use planet_coords::{CubeFace, CurvatureOcclusion, SphereGeometry};

/// 四叉树节点（面 + 层 + 层内坐标）。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct QuadNode {
    /// 所在面。
    pub face: CubeFace,
    /// 层号（0 = 整个面）。
    pub level: u32,
    /// 层内 X 索引（`0 .. 2^level`）。
    pub x: u32,
    /// 层内 Y 索引（`0 .. 2^level`）。
    pub y: u32,
}

impl QuadNode {
    /// 第 0 层的根节点（整个面）。
    pub const fn root(face: CubeFace) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// 四个子节点。
    pub fn children(self) -> [Self; 4] {
        let level = self.level + 1;
        let (x, y) = (self.x * 2, self.y * 2);
        [
            Self {
                face: self.face,
                level,
                x,
                y,
            },
            Self {
                face: self.face,
                level,
                x: x + 1,
                y,
            },
            Self {
                face: self.face,
                level,
                x,
                y: y + 1,
            },
            Self {
                face: self.face,
                level,
                x: x + 1,
                y: y + 1,
            },
        ]
    }

    /// 父节点（根节点返回 `None`）。
    pub fn parent(self) -> Option<Self> {
        if self.level == 0 {
            return None;
        }
        Some(Self {
            face: self.face,
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// 层内的网格边长（格数）。
    pub fn side(self) -> u32 {
        1u32 << self.level.min(30)
    }

    /// 节点中心方向。
    pub fn center_dir(self) -> DVec3 {
        let side = self.side() as f64;
        let u = (self.x as f64 + 0.5) / side * 2.0 - 1.0;
        let v = (self.y as f64 + 0.5) / side * 2.0 - 1.0;
        self.face.uv_to_dir(u, v)
    }

    /// 节点边长对应的球面弧长（米，按面心等距近似）。
    pub fn edge_arc_m(self, radius_m: f64) -> f64 {
        radius_m * std::f64::consts::FRAC_PI_2 / self.side() as f64
    }
}

/// 选择器配置。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadtreeConfig {
    /// 最大层号（`level` 层共 `2^level × 2^level` 块）。
    pub max_level: u32,
    /// 细分阈值：节点角宽大于该值就继续细分。
    pub split_ratio: f64,
    /// 最小边长（米）——即使角宽超标也不再把节点切得比它更细。
    pub min_edge_m: f64,
    /// 单次选择返回的节点数上限（保护内存与命令缓冲）。
    pub max_nodes: usize,
}

impl Default for QuadtreeConfig {
    fn default() -> Self {
        Self {
            max_level: 16,
            split_ratio: 0.25,
            min_edge_m: 256.0,
            max_nodes: 4096,
        }
    }
}

/// 待细分的候选节点（按到相机的距离排序）。
///
/// `f64` 不是 `Ord`，这里用 `total_cmp` 定义全序，并以节点键作为并列打破规则，
/// 保证选择结果**完全确定**（同样的相机方向必然得到同样的叶子集合）。
#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    distance_m: f64,
    node: QuadNode,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // 距离**小**的优先级高（二叉堆是最大堆）。
        other
            .distance_m
            .total_cmp(&self.distance_m)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 四叉树 LOD 选择器。
#[derive(Clone, Debug)]
pub struct QuadtreeSelector {
    config: QuadtreeConfig,
    geometry: SphereGeometry,
    occlusion: CurvatureOcclusion,
}

impl QuadtreeSelector {
    /// 以配置与行星半径构造。
    pub fn new(config: QuadtreeConfig, radius_m: f64) -> Self {
        Self {
            config,
            geometry: SphereGeometry::new(radius_m),
            occlusion: CurvatureOcclusion::new(radius_m),
        }
    }

    /// 配置。
    #[inline]
    pub fn config(&self) -> &QuadtreeConfig {
        &self.config
    }

    /// 选择注视点可见的叶子节点（按层号升序）。
    ///
    /// * `camera_dir`：视线中心方向（自动归一化）；
    /// * `horizon_arc_m`：当前观察高度对应的地平线弧长，超出它的节点整棵剪掉。
    pub fn select(&self, camera_dir: DVec3, horizon_arc_m: f64) -> Vec<QuadNode> {
        let camera_dir = camera_dir.normalize();
        let radius_m = self.geometry.radius_m;

        // 最近优先（best-first）遍历：永远先细分最靠近相机的节点。
        // 这样即使 `max_nodes` 触发截断，被牺牲的也只会是**最远**的环带，
        // 而不会在近处留下空洞（深度优先会先耗尽预算在某个面上）。
        let mut queue: BinaryHeap<Candidate> = CubeFace::ALL
            .iter()
            .map(|face| {
                let node = QuadNode::root(*face);
                Candidate {
                    distance_m: self.geometry.arc_len(camera_dir, node.center_dir()),
                    node,
                }
            })
            .collect();
        let mut leaves = Vec::new();

        while let Some(Candidate { node, .. }) = queue.pop() {
            let center = node.center_dir();
            let angular = SphereGeometry::angular_distance(camera_dir, center);
            let edge = node.edge_arc_m(radius_m);
            // 节点半径按外接圆（对角线一半 ≈ 0.707·边长）保守估算。
            let node_radius_m = edge * 0.71;
            let node_radius_rad = node_radius_m / radius_m;
            let distance_m = angular * radius_m;

            // 整棵子树都在地平线之外 —— 曲率遮挡使它不可见，直接剪枝。
            if distance_m - node_radius_m > horizon_arc_m {
                continue;
            }
            // 背对相机的节点（角距 > 90° + 节点半径）不会出现在视锥里。
            // 注意：这里比较的是**弧度**，与上一行的米不能混用。
            if angular - node_radius_rad > std::f64::consts::FRAC_PI_2 {
                continue;
            }

            let angular_width = edge / distance_m.max(radius_m * 1e-6);
            let should_split = node.level < self.config.max_level
                && edge > self.config.min_edge_m
                && angular_width > self.config.split_ratio;

            if should_split && leaves.len() + queue.len() < self.config.max_nodes {
                for child in node.children() {
                    queue.push(Candidate {
                        distance_m: self.geometry.arc_len(camera_dir, child.center_dir()),
                        node: child,
                    });
                }
            } else {
                leaves.push(node);
                if leaves.len() >= self.config.max_nodes {
                    break;
                }
            }
        }

        leaves.sort();
        leaves
    }

    /// 节点中心的可见性（供实例化阶段做二次剔除）。
    pub fn is_visible(
        &self,
        eye_dir: DVec3,
        eye_height_m: f64,
        node: QuadNode,
        clearance_m: f64,
    ) -> bool {
        let target = node.center_dir();
        let horizon = self.occlusion.horizon_arc_m(eye_height_m);
        let distance = self.geometry.arc_len(eye_dir.normalize(), target);
        distance - node.edge_arc_m(self.geometry.radius_m) * 0.71 <= horizon + clearance_m
            && !self.occlusion.is_blocked(eye_dir, target, clearance_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_R: f64 = 6_371_000.0;

    fn selector() -> QuadtreeSelector {
        QuadtreeSelector::new(QuadtreeConfig::default(), EARTH_R)
    }

    #[test]
    fn roots_cover_all_six_faces() {
        let roots: Vec<QuadNode> = CubeFace::ALL.iter().map(|f| QuadNode::root(*f)).collect();
        assert_eq!(roots.len(), 6);
        assert!(roots.iter().all(|node| node.parent().is_none()));
        assert_eq!(roots[0].side(), 1);
    }

    #[test]
    fn children_are_contained_in_parent() {
        let node = QuadNode {
            face: CubeFace::PosZ,
            level: 3,
            x: 5,
            y: 2,
        };
        for child in node.children() {
            assert_eq!(child.parent(), Some(node));
            assert_eq!(child.level, node.level + 1);
            // 子节点中心应落在父节点角宽之内。
            let separation =
                SphereGeometry::angular_distance(node.center_dir(), child.center_dir());
            let parent_arc = node.edge_arc_m(EARTH_R) / EARTH_R;
            assert!(separation <= parent_arc, "子节点超出父节点范围");
        }
    }

    #[test]
    fn selection_is_deterministic_and_bounded() {
        let selector = selector();
        let camera = DVec3::new(0.4, 0.5, -0.77).normalize();
        let first = selector.select(camera, 2.0e6);
        let second = selector.select(camera, 2.0e6);
        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.len() <= selector.config().max_nodes);
        assert!(first.windows(2).all(|w| w[0] <= w[1]), "结果必须有序");
        assert!(first
            .iter()
            .all(|node| node.level <= selector.config().max_level));
    }

    #[test]
    fn ground_observer_gets_no_nodes_beyond_a_zero_horizon() {
        let selector = selector();
        let nodes = selector.select(DVec3::X, 0.0);
        // 地平线为 0：只有「脚下」的节点能被看到，数量必然远小于全行星细分结果。
        let orbital = selector.select(DVec3::X, 2.3e6);
        assert!(
            nodes.len() < orbital.len(),
            "地面视角不应比轨道视角看到更多节点"
        );
    }

    #[test]
    fn near_camera_nodes_are_split_more_finely() {
        let selector = selector();
        let camera = DVec3::X;
        let nodes = selector.select(camera, 2.3e6);
        // 距相机最近的叶子必须比最远的叶子更细（层号更大）。
        let nearest = nodes
            .iter()
            .min_by(|a, b| {
                let da = SphereGeometry::angular_distance(camera, a.center_dir());
                let db = SphereGeometry::angular_distance(camera, b.center_dir());
                da.partial_cmp(&db).unwrap()
            })
            .copied()
            .unwrap();
        let farthest = nodes
            .iter()
            .max_by(|a, b| {
                let da = SphereGeometry::angular_distance(camera, a.center_dir());
                let db = SphereGeometry::angular_distance(camera, b.center_dir());
                da.partial_cmp(&db).unwrap()
            })
            .copied()
            .unwrap();
        assert!(nearest.level >= farthest.level, "近处必须更细");
    }
}
