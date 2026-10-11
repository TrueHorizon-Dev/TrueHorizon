//! # server-authority —— 服务端权威
//!
//! 专用服务器的核心职责：**不信任任何客户端输入**，并且**只把玩家看得见的东西发给他**。
//!
//! * [`MovementValidator`]：速度/瞬移校验。客户端可以用任何方式伪造输入，服务端用
//!   「位移 / 时间」这一物理量做硬约束，越界即纠正或拒绝。
//! * [`InterestManager`]：基于**曲率遮挡**的兴趣管理（AoI）。行星是球体，远处的玩家
//!   会被行星本身挡住——这正是最廉价、最物理的视野裁剪。
//!
//! 设计红线：
//!
//! 1. 所有进入权威层的数据**先 `sanitize` 再校验**（NaN / 超长向量一律先规整）；
//! 2. 校验失败的默认处置是**纠正**（下发权威状态）而不是踢人——网络抖动是常态；
//! 3. AoI 的结果必须**确定性**（排序后再截断），否则同一次查询在不同机器上返回
//!    different 集合会破坏回放一致性。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use planet_coords::{CurvatureOcclusion, SphereGeometry};
use protocol::{ClientInput, PlayerState, RejectReason};
use rayon::prelude::*;

/// 校验结论。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValidationOutcome {
    /// 通过。
    Accepted,
    /// 需要纠正：以服务端状态为准。
    Corrected {
        /// 服务端认可的最大速度（米/秒）。
        max_speed_m_s: f64,
    },
    /// 拒绝：结构非法或明显越权。
    Rejected {
        /// 原因。
        reason: RejectReason,
    },
}

/// 移动校验器。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementValidator {
    /// 允许的最大水平速度（米/秒）。冲刺 + 下坡加速度应留出余量。
    pub max_speed_m_s: f64,
    /// 判定为「瞬移」的倍率：超过 `max_speed × 该倍率 × dt` 的位移直接拒绝。
    pub teleport_factor: f64,
    /// 允许的时间步长下限（秒）：`dt` 过小会让速度检查失去意义。
    pub min_dt_s: f64,
    /// 允许的时间步长上限（秒）。
    pub max_dt_s: f64,
}

impl MovementValidator {
    /// 以最大速度构造（默认瞬移倍率 4.0，dt 区间 `[1 ms, 1 s]`）。
    pub const fn new(max_speed_m_s: f64) -> Self {
        Self {
            max_speed_m_s,
            teleport_factor: 4.0,
            min_dt_s: 0.001,
            max_dt_s: 1.0,
        }
    }

    /// 校验一次输入的结构合法性。
    ///
    /// 返回 `None` 表示输入不可信，应回 [`RejectReason::Malformed`]。
    pub fn sanitize_input(&self, input: ClientInput) -> Option<ClientInput> {
        let mut sanitized = input;
        sanitized.sanitize();
        if !sanitized.is_finite() {
            return None;
        }
        Some(sanitized)
    }

    /// 校验从 `previous` 到 `claimed` 的位移是否在物理允许范围内。
    pub fn validate_movement(
        &self,
        previous: &PlayerState,
        claimed: &PlayerState,
        delta_s: f64,
    ) -> ValidationOutcome {
        if !delta_s.is_finite() || delta_s < self.min_dt_s {
            // dt 过小/非法：无法做速度推理，直接以服务端状态纠正。
            return ValidationOutcome::Corrected {
                max_speed_m_s: self.max_speed_m_s,
            };
        }
        let dt = delta_s.min(self.max_dt_s);
        let distance = previous.distance_to(claimed);
        if !distance.is_finite() {
            return ValidationOutcome::Rejected {
                reason: RejectReason::Malformed,
            };
        }
        let speed = distance / dt;
        if speed > self.max_speed_m_s * self.teleport_factor {
            return ValidationOutcome::Rejected {
                reason: RejectReason::Unauthorized,
            };
        }
        if speed > self.max_speed_m_s {
            return ValidationOutcome::Corrected {
                max_speed_m_s: self.max_speed_m_s,
            };
        }
        ValidationOutcome::Accepted
    }

    /// 校验一次方块修改请求：目标必须在交互距离内。
    pub fn validate_edit_distance(
        &self,
        player: &PlayerState,
        target: [f64; 3],
        max_reach_m: f64,
    ) -> ValidationOutcome {
        let dx = player.position[0] - target[0];
        let dy = player.position[1] - target[1];
        let dz = player.position[2] - target[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        if !distance.is_finite() {
            return ValidationOutcome::Rejected {
                reason: RejectReason::Malformed,
            };
        }
        if distance > max_reach_m {
            return ValidationOutcome::Rejected {
                reason: RejectReason::OutOfRange,
            };
        }
        ValidationOutcome::Accepted
    }
}

impl Default for MovementValidator {
    fn default() -> Self {
        Self::new(12.0)
    }
}

/// 基于曲率遮挡的兴趣管理（AoI）。
///
/// 判定链路：地平线可达性（双方高度决定各自能看到多远）→ 地形余量 → 目标数截断。
/// 相比「半径裁剪」，它天然处理了「行星另一侧」「山谷对面」等几何不可见的情形，
/// 而且计算量只有几次三角函数。
#[derive(Clone, Copy, Debug)]
pub struct InterestManager {
    geometry: SphereGeometry,
    occlusion: CurvatureOcclusion,
    /// 地形起伏余量（米）：允许「隔着一道山脊也能看到」的宽容度。
    clearance_m: f64,
    /// 单次快照最多下发多少目标。
    max_targets: usize,
}

impl InterestManager {
    /// 以行星半径、地形余量与目标上限构造。
    pub fn new(radius_m: f64, clearance_m: f64, max_targets: usize) -> Self {
        Self {
            geometry: SphereGeometry::new(radius_m),
            occlusion: CurvatureOcclusion::new(radius_m),
            clearance_m: clearance_m.max(0.0),
            max_targets: max_targets.max(1),
        }
    }

    /// 地形余量。
    #[inline]
    pub fn clearance_m(&self) -> f64 {
        self.clearance_m
    }

    /// 目标上限。
    #[inline]
    pub fn max_targets(&self) -> usize {
        self.max_targets
    }

    /// 两个世界坐标之间的相互可见性（自动由半径推出各自的高度）。
    pub fn is_visible(&self, observer: [f64; 3], target: [f64; 3]) -> bool {
        let Some((observer_dir, observer_height)) = self.split(observer) else {
            return false;
        };
        let Some((target_dir, target_height)) = self.split(target) else {
            return false;
        };
        self.occlusion.mutually_visible(
            observer_dir,
            observer_height,
            target_dir,
            target_height,
            self.clearance_m,
        )
    }

    /// 由世界坐标算出 `(方向, 高度)`；坐标非法时返回 `None`。
    fn split(&self, position: [f64; 3]) -> Option<(glam::DVec3, f64)> {
        let vector = glam::DVec3::new(position[0], position[1], position[2]);
        let radius = vector.length();
        if !radius.is_finite() || radius <= f64::MIN_POSITIVE {
            return None;
        }
        Some((vector / radius, radius - self.geometry.radius_m))
    }

    /// 过滤出对 `observer` 可见的候选者 ID（升序，最多 `max_targets` 个）。
    ///
    /// 排序保证确定性：并行过滤不会影响结果集合的**顺序**语义。
    pub fn filter_visible(&self, observer: [f64; 3], candidates: &[PlayerState]) -> Vec<u32> {
        let mut visible: Vec<u32> = candidates
            .par_iter()
            .filter(|player| player.id != 0 && self.is_visible(observer, player.position))
            .map(|player| player.id)
            .collect();
        visible.sort_unstable();
        visible.dedup();
        visible.truncate(self.max_targets);
        visible
    }

    /// 两个表面点之间的可达距离（米），供调试与压测估算。
    pub fn reachable_arc_m(&self, observer_height_m: f64, target_height_m: f64) -> f64 {
        let reach_angle = self.occlusion.horizon_angle(observer_height_m)
            + self.occlusion.horizon_angle(target_height_m);
        self.geometry.radius_m * reach_angle + self.clearance_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{Buttons, ClientInput};

    const EARTH_R: f64 = 6_371_000.0;

    fn at_height(height_m: f64, direction: glam::DVec3) -> [f64; 3] {
        let position = direction.normalize() * (EARTH_R + height_m);
        [position.x, position.y, position.z]
    }

    #[test]
    fn legal_movement_is_accepted() {
        let validator = MovementValidator::new(12.0);
        let previous = PlayerState::at(1, [0.0, 0.0, 0.0]);
        let mut claimed = previous;
        claimed.position = [0.6, 0.0, 0.0]; // 0.6 m / 0.05 s = 12 m/s
        assert_eq!(
            validator.validate_movement(&previous, &claimed, 0.05),
            ValidationOutcome::Accepted
        );
    }

    #[test]
    fn speed_hack_is_corrected_and_teleport_is_rejected() {
        let validator = MovementValidator::new(12.0);
        let previous = PlayerState::at(1, [0.0, 0.0, 0.0]);

        let mut fast = previous;
        fast.position = [2.0, 0.0, 0.0]; // 40 m/s：超速但未触及 4 倍瞬移阈值
        assert!(matches!(
            validator.validate_movement(&previous, &fast, 0.05),
            ValidationOutcome::Corrected { .. }
        ));

        let mut teleport = previous;
        teleport.position = [1_000.0, 0.0, 0.0];
        assert_eq!(
            validator.validate_movement(&previous, &teleport, 0.05),
            ValidationOutcome::Rejected {
                reason: RejectReason::Unauthorized
            }
        );
    }

    #[test]
    fn degenerate_time_steps_are_corrected_not_accepted() {
        let validator = MovementValidator::new(12.0);
        let previous = PlayerState::at(1, [0.0; 3]);
        let mut claimed = previous;
        claimed.position = [1.0, 0.0, 0.0];
        // dt = 0 会让速度趋于无穷：必须走纠正分支而不是拒绝/放行。
        assert!(matches!(
            validator.validate_movement(&previous, &claimed, 0.0),
            ValidationOutcome::Corrected { .. }
        ));
        assert!(matches!(
            validator.validate_movement(&previous, &claimed, f64::NAN),
            ValidationOutcome::Corrected { .. }
        ));
    }

    #[test]
    fn hostile_input_is_sanitized_before_validation() {
        let validator = MovementValidator::new(12.0);
        assert!(validator.sanitize_input(ClientInput::IDLE).is_some());

        let hostile = ClientInput {
            move_axis: [f32::NAN, 0.0, 0.0],
            buttons: u32::MAX,
            ..ClientInput::IDLE
        };
        let sanitized = validator.sanitize_input(hostile).expect("规整后应当可用");
        assert_eq!(sanitized.buttons, Buttons::ALL);
        assert_eq!(sanitized.move_axis, [0.0; 3]);
    }

    #[test]
    fn edit_reach_is_enforced() {
        let validator = MovementValidator::new(12.0);
        let player = PlayerState::at(1, [0.0, 0.0, 0.0]);
        assert_eq!(
            validator.validate_edit_distance(&player, [3.0, 4.0, 0.0], 6.0),
            ValidationOutcome::Accepted
        );
        assert_eq!(
            validator.validate_edit_distance(&player, [30.0, 40.0, 0.0], 6.0),
            ValidationOutcome::Rejected {
                reason: RejectReason::OutOfRange
            }
        );
    }

    #[test]
    fn nearby_players_are_visible_but_the_far_side_is_not() {
        let aoi = InterestManager::new(EARTH_R, 0.0, 64);
        let eye = glam::DVec3::X;
        let nearby = glam::DVec3::new(1.0, 0.0003, 0.0).normalize();
        let antipode = -glam::DVec3::X;

        assert!(
            aoi.is_visible(at_height(2.0, eye), at_height(2.0, nearby)),
            "2 公里内的两个地面玩家应当互相可见"
        );
        assert!(
            !aoi.is_visible(at_height(2.0, eye), at_height(2.0, antipode)),
            "行星另一侧绝不可见"
        );
        assert!(
            !aoi.is_visible([0.0, 0.0, 0.0], at_height(2.0, eye)),
            "非法坐标必须安全返回 false"
        );
    }

    #[test]
    fn towers_extend_visibility_beyond_the_ground_horizon() {
        let aoi = InterestManager::new(EARTH_R, 0.0, 64);
        let eye = glam::DVec3::X;
        // 30 公里外：平地上看不见，30 米高的塔互相可以看见（视线约 39 公里）。
        let distant = glam::DVec3::new(1.0, 30_000.0 / EARTH_R, 0.0).normalize();
        assert!(!aoi.is_visible(at_height(2.0, eye), at_height(2.0, distant)));
        assert!(aoi.is_visible(at_height(30.0, eye), at_height(30.0, distant)));
        assert!(aoi.reachable_arc_m(30.0, 30.0) > 30_000.0);
        assert!(
            !aoi.is_visible(at_height(30.0, eye), at_height(30.0, -glam::DVec3::X)),
            "再高的塔也看不到行星另一侧"
        );
    }

    #[test]
    fn filter_visible_is_deterministic_and_bounded() {
        let aoi = InterestManager::new(EARTH_R, 0.0, 3);
        let observer = at_height(2.0, glam::DVec3::X);
        let mut candidates = Vec::new();
        for id in 1..=8u32 {
            let offset = f64::from(id) * 0.0002;
            let direction = glam::DVec3::new(1.0, offset, 0.0).normalize();
            candidates.push(PlayerState::at(id, at_height(2.0, direction)));
        }
        // 行星另一侧的玩家（不可见）。
        candidates.push(PlayerState::at(99, at_height(2.0, -glam::DVec3::X)));

        let visible = aoi.filter_visible(observer, &candidates);
        assert!(visible.len() <= 3, "必须受 max_targets 约束");
        assert!(!visible.contains(&99), "不可见目标不得出现在结果中");
        assert_eq!(
            visible,
            aoi.filter_visible(observer, &candidates),
            "必须确定性"
        );
        assert!(visible.windows(2).all(|w| w[0] < w[1]), "结果必须升序");
    }
}
