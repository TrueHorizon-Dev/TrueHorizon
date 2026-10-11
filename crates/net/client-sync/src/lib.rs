//! # client-sync —— 客户端预测与和解
//!
//! 网络延迟不可消除，但可以被**隐藏**：
//!
//! ```text
//! 输入 ──► [本地立即应用] ──► 预测状态入队（带 sequence）
//!              │                        │
//!              │                  服务端快照（携带 last_acked_input）
//!              ▼                        ▼
//!         渲染（无延迟手感）      [和解]：误差超阈值 → 回滚到权威状态 + 重放未确认输入
//! ```
//!
//! 另一半是**插值**：远端玩家按固定的 `delay_s`（通常等于一个快照间隔）延迟播放，
//! 在两个已知快照之间插值，从而把网络抖动转成平滑运动。
//!
//! 本 crate 只做纯计算（不含 socket、不含线程），因此可以被单元测试与回放器直接驱动。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::collections::VecDeque;

use protocol::{PlayerState, ServerSnapshot};

/// 一条本地预测状态。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictedState {
    /// 对应的输入序号。
    pub sequence: u32,
    /// 预测发生的 Tick。
    pub tick: u64,
    /// 预测的世界坐标（米）。
    pub position: [f64; 3],
    /// 预测的偏航。
    pub yaw_rad: f32,
}

/// 预测历史（环形缓冲）。
///
/// 容量固定：一条未确认的输入最多保留 `capacity` 个 Tick，超出即认为该输入永远不会
/// 被确认（丢包），直接丢弃——**无限增长是内存泄漏的常见来源**。
#[derive(Clone, Debug)]
pub struct PredictionBuffer {
    entries: VecDeque<PredictedState>,
    capacity: usize,
    last_acked: u32,
}

impl PredictionBuffer {
    /// 以容量构造。
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            last_acked: 0,
        }
    }

    /// 记录一次预测。
    pub fn push(&mut self, state: PredictedState) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(state);
    }

    /// 服务端确认到 `sequence`：丢弃所有已确认的预测，返回丢弃条数。
    pub fn acknowledge(&mut self, sequence: u32) -> usize {
        let mut dropped = 0;
        while let Some(front) = self.entries.front() {
            if front.sequence > sequence {
                break;
            }
            self.entries.pop_front();
            dropped += 1;
        }
        self.last_acked = self.last_acked.max(sequence);
        dropped
    }

    /// 已确认的最大输入序号。
    #[inline]
    pub fn last_acked(&self) -> u32 {
        self.last_acked
    }

    /// 尚未确认的预测（按序号升序）。
    pub fn pending(&self) -> impl Iterator<Item = &PredictedState> {
        self.entries.iter()
    }

    /// 未确认条数。
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否没有未确认预测。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 最近一次预测。
    pub fn latest(&self) -> Option<&PredictedState> {
        self.entries.back()
    }
}

/// 和解结论。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconcileReport {
    /// 权威位置与本地预测位置的偏差（米）。
    pub error_m: f64,
    /// 是否需要回滚重放。
    pub needs_replay: bool,
    /// 重放起始序号（`needs_replay` 为真时有效）。
    pub replay_from_sequence: u32,
}

/// 比较权威状态与本地预测，判定是否需要回滚重放。
///
/// 关键设计：**只有超过容差的偏差才触发重放**。对每个快照都无条件重放会让角色
/// 在网络抖动下持续「抽搐」，体感比不修正更差。
pub fn reconcile(
    local: &PredictionBuffer,
    authoritative: &PlayerState,
    tolerance_m: f64,
) -> ReconcileReport {
    // 找与权威确认序号对应的那帧预测作为比较基准。
    let reference = local
        .pending()
        .find(|state| state.sequence == authoritative.last_acked_input)
        .or_else(|| local.pending().next())
        .or_else(|| local.latest());

    let Some(reference) = reference else {
        return ReconcileReport {
            error_m: 0.0,
            needs_replay: false,
            replay_from_sequence: 0,
        };
    };

    let dx = reference.position[0] - authoritative.position[0];
    let dy = reference.position[1] - authoritative.position[1];
    let dz = reference.position[2] - authoritative.position[2];
    let error_m = (dx * dx + dy * dy + dz * dz).sqrt();
    let tolerance = tolerance_m.max(0.0);

    ReconcileReport {
        error_m,
        needs_replay: error_m > tolerance,
        replay_from_sequence: reference.sequence,
    }
}

/// 插值后的状态。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolatedState {
    /// 所在 Tick（插值起点）。
    pub tick: u64,
    /// 插值因子 `[0, 1]`。
    pub alpha: f64,
    /// 快照原点（世界坐标，米）。
    pub origin: [f64; 3],
    /// 相对原点的位置（米，`f32`）。
    pub position: [f32; 3],
    /// 偏航。
    pub yaw_rad: f32,
    /// 俯仰。
    pub pitch_rad: f32,
}

/// 快照插值器。
#[derive(Clone, Debug)]
pub struct SnapshotInterpolator {
    buffer: VecDeque<ServerSnapshot>,
    delay_s: f64,
    capacity: usize,
}

impl SnapshotInterpolator {
    /// 以延迟与容量构造。`delay_s` 通常取一个快照间隔（20 Hz → 0.05 s）再加余量。
    pub fn new(delay_s: f64, capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.max(2)),
            delay_s: delay_s.max(0.0),
            capacity: capacity.max(2),
        }
    }

    /// 推入一个快照。
    ///
    /// 返回 `false` 表示该快照被丢弃（乱序或重复的 tick）——**绝不接受倒退的时间线**，
    /// 否则插值会在两个方向上反复横跳。
    pub fn push(&mut self, snapshot: ServerSnapshot) -> bool {
        if let Some(last) = self.buffer.back() {
            if snapshot.tick <= last.tick {
                return false;
            }
        }
        self.buffer.push_back(snapshot);
        while self.buffer.len() > self.capacity {
            self.buffer.pop_front();
        }
        true
    }

    /// 延迟量（秒）。
    #[inline]
    pub fn delay_s(&self) -> f64 {
        self.delay_s
    }

    /// 缓冲区中的快照数。
    #[inline]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// 由最新的服务端时间算出本次渲染应采样的时间点。
    pub fn render_time_s(&self, latest_server_time_s: f64) -> f64 {
        latest_server_time_s - self.delay_s
    }

    /// 在缓冲区中插值出某个玩家在 `render_time_s` 时刻的状态。
    ///
    /// * 早于最早快照 → 钳制到最早（不预测过去）；
    /// * 晚于最新快照 → 钳制到最新（**不做外推**：外推会导致穿墙与抖动）。
    pub fn interpolate_player(
        &self,
        player_id: u32,
        render_time_s: f64,
    ) -> Option<InterpolatedState> {
        let first = self.buffer.front()?;
        let last = self.buffer.back()?;

        if render_time_s <= first.server_time_s {
            return state_from_snapshot(first, last, player_id, 0.0);
        }
        if render_time_s >= last.server_time_s {
            return state_from_snapshot(last, last, player_id, 1.0);
        }

        let mut previous: Option<&ServerSnapshot> = None;
        for snapshot in &self.buffer {
            if let Some(prev) = previous {
                if render_time_s >= prev.server_time_s && render_time_s <= snapshot.server_time_s {
                    let span = snapshot.server_time_s - prev.server_time_s;
                    let alpha = if span > f64::MIN_POSITIVE {
                        (render_time_s - prev.server_time_s) / span
                    } else {
                        0.0
                    };
                    return state_from_snapshot(prev, snapshot, player_id, alpha);
                }
            }
            previous = Some(snapshot);
        }
        None
    }
}

/// 在两个快照之间为指定玩家插值。
fn state_from_snapshot(
    older: &ServerSnapshot,
    newer: &ServerSnapshot,
    player_id: u32,
    alpha: f64,
) -> Option<InterpolatedState> {
    let find = |snapshot: &ServerSnapshot| {
        snapshot
            .players
            .iter()
            .find(|player| player.id == player_id)
            .copied()
    };
    let a = find(older);
    let b = find(newer).or(a);
    let a = a.or(b)?;
    let b = b?;

    // 快照原点可能不同（浮动原点），必须先归一到 newer 的原点再插值。
    let relative = |player: &PlayerState, origin: [f64; 3]| -> [f64; 3] {
        [
            player.position[0] - origin[0],
            player.position[1] - origin[1],
            player.position[2] - origin[2],
        ]
    };
    let pa = relative(&a, newer.origin);
    let pb = relative(&b, newer.origin);
    let lerp = |x: f64, y: f64| (x + (y - x) * alpha) as f32;
    let lerp_angle = |x: f32, y: f32| {
        // 角度插值走最短弧，避免 359° → 1° 时转一整圈。
        let mut delta = y - x;
        while delta > std::f32::consts::PI {
            delta -= std::f32::consts::TAU;
        }
        while delta < -std::f32::consts::PI {
            delta += std::f32::consts::TAU;
        }
        x + delta * alpha as f32
    };

    Some(InterpolatedState {
        tick: newer.tick,
        alpha,
        origin: newer.origin,
        position: [lerp(pa[0], pb[0]), lerp(pa[1], pb[1]), lerp(pa[2], pb[2])],
        yaw_rad: lerp_angle(a.yaw_rad, b.yaw_rad),
        pitch_rad: lerp_angle(a.pitch_rad, b.pitch_rad),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::ServerSnapshot;

    fn snapshot(tick: u64, time_s: f64, x: f64) -> ServerSnapshot {
        ServerSnapshot {
            tick,
            server_time_s: time_s,
            origin: [0.0; 3],
            players: vec![PlayerState::at(7, [x, 0.0, 0.0])],
            entities: Vec::new(),
        }
    }

    #[test]
    fn prediction_buffer_drops_acknowledged_entries() {
        let mut buffer = PredictionBuffer::new(8);
        for sequence in 0..5 {
            buffer.push(PredictedState {
                sequence,
                tick: u64::from(sequence),
                position: [f64::from(sequence), 0.0, 0.0],
                yaw_rad: 0.0,
            });
        }
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.acknowledge(2), 3);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.last_acked(), 2);
        assert_eq!(buffer.pending().next().unwrap().sequence, 3);
    }

    #[test]
    fn prediction_buffer_is_bounded() {
        let mut buffer = PredictionBuffer::new(3);
        for sequence in 0..10 {
            buffer.push(PredictedState {
                sequence,
                tick: u64::from(sequence),
                position: [0.0; 3],
                yaw_rad: 0.0,
            });
        }
        assert_eq!(buffer.len(), 3, "容量必须硬性约束");
        assert_eq!(buffer.latest().unwrap().sequence, 9);
    }

    #[test]
    fn reconcile_only_triggers_beyond_tolerance() {
        let mut buffer = PredictionBuffer::new(8);
        buffer.push(PredictedState {
            sequence: 5,
            tick: 5,
            position: [100.0, 0.0, 0.0],
            yaw_rad: 0.0,
        });
        let mut authoritative = PlayerState::at(1, [100.05, 0.0, 0.0]);
        authoritative.last_acked_input = 5;

        let small = reconcile(&buffer, &authoritative, 0.2);
        assert!(!small.needs_replay);
        assert!((small.error_m - 0.05).abs() < 1e-9);

        authoritative.position = [140.0, 0.0, 0.0];
        let large = reconcile(&buffer, &authoritative, 0.2);
        assert!(large.needs_replay);
        assert_eq!(large.replay_from_sequence, 5);
        assert!((large.error_m - 40.0).abs() < 1e-9);
    }

    #[test]
    fn reconcile_without_history_is_a_no_op() {
        let buffer = PredictionBuffer::new(4);
        let authoritative = PlayerState::at(1, [1.0, 2.0, 3.0]);
        let report = reconcile(&buffer, &authoritative, 0.1);
        assert_eq!(report.error_m, 0.0);
        assert!(!report.needs_replay);
    }

    #[test]
    fn interpolation_between_two_snapshots_is_linear() {
        let mut interpolator = SnapshotInterpolator::new(0.0, 8);
        assert!(interpolator.push(snapshot(1, 0.0, 0.0)));
        assert!(interpolator.push(snapshot(2, 0.1, 10.0)));
        assert_eq!(interpolator.buffered(), 2);

        let middle = interpolator.interpolate_player(7, 0.05).unwrap();
        assert_eq!(middle.tick, 2);
        assert!((middle.alpha - 0.5).abs() < 1e-9);
        assert!(
            (middle.position[0] - 5.0).abs() < 1e-4,
            "位置 {}",
            middle.position[0]
        );

        // 钳制而不是外推。
        let future = interpolator.interpolate_player(7, 9.0).unwrap();
        assert!((future.position[0] - 10.0).abs() < 1e-4);
        let past = interpolator.interpolate_player(7, -9.0).unwrap();
        assert!(past.position[0].abs() < 1e-4);
        assert!(interpolator.interpolate_player(999, 0.05).is_none());
    }

    #[test]
    fn out_of_order_snapshots_are_rejected() {
        let mut interpolator = SnapshotInterpolator::new(0.05, 4);
        assert!(interpolator.push(snapshot(10, 0.50, 0.0)));
        assert!(
            !interpolator.push(snapshot(9, 0.45, 5.0)),
            "不得接受倒退的时间线"
        );
        assert!(
            !interpolator.push(snapshot(10, 0.50, 1.0)),
            "重复 tick 必须丢弃"
        );
        assert_eq!(interpolator.buffered(), 1);
        assert!((interpolator.render_time_s(1.0) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn buffer_capacity_evicts_oldest_snapshots() {
        let mut interpolator = SnapshotInterpolator::new(0.0, 2);
        for tick in 0..5 {
            interpolator.push(snapshot(tick, tick as f64 * 0.05, tick as f64));
        }
        assert_eq!(interpolator.buffered(), 2);
    }
}
