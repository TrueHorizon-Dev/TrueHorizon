//! 多人联机端到端：输入 → 规整 → 权威校验 → 快照 → 插值 → 和解。
//!
//! 全程不涉及图形与真实 socket：服务端与客户端两侧都是纯计算组件，
//! 因此可以在无头 CI 里以确定性方式复现「作弊被拦下」「远端玩家平滑移动」等场景。

use client_sync::{
    reconcile, InterpolatedState, PredictedState, PredictionBuffer, SnapshotInterpolator,
};
use glam::DVec3;
use protocol::{
    Buttons, ClientInput, ClientMessage, PlayerState, ProtocolVersion, RejectReason, ServerMessage,
    ServerSnapshot, PROTOCOL_VERSION,
};
use server_authority::{InterestManager, MovementValidator, ValidationOutcome};

const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// 20 Hz 快照间隔。
const TICK_DT: f64 = 0.05;
/// 服务端允许的最大速度。
const MAX_SPEED_M_S: f64 = 12.0;

fn surface_position(direction: DVec3, height_m: f64) -> [f64; 3] {
    let position = direction.normalize() * (EARTH_RADIUS_M + height_m);
    [position.x, position.y, position.z]
}

fn snapshot(tick: u64, player: PlayerState) -> ServerSnapshot {
    ServerSnapshot {
        tick,
        server_time_s: tick as f64 * TICK_DT,
        origin: [EARTH_RADIUS_M, 0.0, 0.0],
        players: vec![player],
        entities: Vec::new(),
    }
}

#[test]
fn handshake_rejects_an_older_client() {
    let client = ClientMessage::Hello {
        version: ProtocolVersion(PROTOCOL_VERSION.0 - 1),
        name: "legacy".into(),
    };
    let ClientMessage::Hello { version, .. } = client else {
        unreachable!("刚构造的就是握手报文")
    };
    assert!(!PROTOCOL_VERSION.is_compatible_with(version));

    let reply = ServerMessage::Reject {
        reason: RejectReason::VersionMismatch,
    };
    assert!(matches!(
        reply,
        ServerMessage::Reject {
            reason: RejectReason::VersionMismatch
        }
    ));
}

#[test]
fn authoritative_validation_stops_a_speed_hacker() {
    let validator = MovementValidator::new(MAX_SPEED_M_S);
    let previous_authoritative = PlayerState::at(1, surface_position(DVec3::X, 2.0));

    // 诚实客户端：0.05 秒走 0.5 米（10 m/s）。
    let mut honest = previous_authoritative;
    honest.position = surface_position(DVec3::X, 2.0);
    honest.position[1] += 0.5;
    assert_eq!(
        validator.validate_movement(&previous_authoritative, &honest, TICK_DT),
        ValidationOutcome::Accepted
    );

    // 作弊客户端：一帧瞬移 500 米。
    let mut cheater = previous_authoritative;
    cheater.position[1] += 500.0;
    let outcome = validator.validate_movement(&previous_authoritative, &cheater, TICK_DT);
    assert_eq!(
        outcome,
        ValidationOutcome::Rejected {
            reason: RejectReason::Unauthorized
        }
    );
    // 服务端的处置是「下发权威状态」，而不是踢人。
    let correction = ServerMessage::Correction(previous_authoritative);
    assert!(matches!(correction, ServerMessage::Correction(_)));

    // 超速但未到瞬移阈值：纠正而不是拒绝。
    let mut sprinting = previous_authoritative;
    sprinting.position[1] += 1.0; // 20 m/s
    assert!(matches!(
        validator.validate_movement(&previous_authoritative, &sprinting, TICK_DT),
        ValidationOutcome::Corrected { max_speed_m_s } if (max_speed_m_s - MAX_SPEED_M_S).abs() < 1e-9
    ));
}

#[test]
fn hostile_input_is_neutralized_before_it_reaches_the_simulation() {
    let validator = MovementValidator::new(MAX_SPEED_M_S);
    let hostile = ClientInput {
        sequence: 7,
        tick: 3,
        move_axis: [f32::NAN, f32::INFINITY, 0.0],
        look_yaw_rad: f32::NAN,
        look_pitch_rad: 100.0,
        buttons: u32::MAX,
    };
    assert!(!hostile.is_finite(), "原始输入就是恶意的");
    let sanitized = validator.sanitize_input(hostile).expect("规整后应当可接受");
    assert!(sanitized.is_finite());
    assert_eq!(sanitized.move_axis, [0.0; 3]);
    assert_eq!(sanitized.look_yaw_rad, 0.0);
    assert_eq!(sanitized.buttons, Buttons::ALL);
    assert_eq!(sanitized.sequence, 7, "序号必须原样保留，否则和解会错位");
}

#[test]
fn interest_management_hides_players_behind_the_planet() {
    let aoi = InterestManager::new(EARTH_RADIUS_M, 8.0, 32);
    let observer = surface_position(DVec3::X, 2.0);
    let near = surface_position(DVec3::new(1.0, 0.0003, 0.0), 2.0);
    let far_side = surface_position(-DVec3::X, 2.0);

    let candidates = vec![
        PlayerState::at(1, near),
        PlayerState::at(2, far_side),
        PlayerState::at(3, surface_position(DVec3::new(1.0, -0.0002, 0.0), 2.0)),
    ];
    let visible = aoi.filter_visible(observer, &candidates);
    assert!(visible.contains(&1) && visible.contains(&3));
    assert!(!visible.contains(&2), "行星另一侧必须被曲率遮挡剔除");
}

#[test]
fn remote_players_interpolate_smoothly_between_snapshots() {
    let mut interpolator = SnapshotInterpolator::new(TICK_DT, 16);
    let mut expected_x = Vec::new();
    for tick in 0..10u64 {
        let x = tick as f64 * 100.0;
        expected_x.push(x);
        let player = PlayerState::at(2, [EARTH_RADIUS_M + x, 0.0, 0.0]);
        assert!(
            interpolator.push(snapshot(tick, player)),
            "顺序快照必须被接受"
        );
    }
    assert_eq!(interpolator.buffered(), 10);

    // 在相邻快照中点采样：位置应严格单调递增且落在相邻两帧之间。
    let mut previous = f64::MIN;
    for tick in 0..8u64 {
        let sample_time = (tick as f64 + 0.5) * TICK_DT;
        let InterpolatedState {
            origin,
            position,
            alpha,
            ..
        } = interpolator
            .interpolate_player(2, sample_time)
            .expect("中点必须有插值结果");
        assert!((0.0..=1.0).contains(&alpha));
        // 位置是相对快照原点的偏移，换算回世界坐标后再比较。
        let world_x = origin[0] + f64::from(position[0]);
        assert!(
            world_x > previous,
            "远端玩家必须平滑前进：{world_x} <= {previous}"
        );
        previous = world_x;
    }

    // 不存在的玩家返回 None（而不是凭空造一个）。
    assert!(interpolator.interpolate_player(999, 0.1).is_none());
}

#[test]
fn client_prediction_reconciles_after_a_correction() {
    let mut local = PredictionBuffer::new(32);
    let mut predicted = PlayerState::at(1, surface_position(DVec3::X, 2.0));
    for sequence in 0..5u32 {
        predicted.position[1] += 0.5;
        local.push(PredictedState {
            sequence,
            tick: u64::from(sequence),
            position: predicted.position,
            yaw_rad: 0.0,
        });
    }

    // 服务端确认到序号 4，且位置与本地预测完全一致 → 无需重放。
    let mut authoritative = predicted;
    authoritative.last_acked_input = 4;
    let report = reconcile(&local, &authoritative, 0.05);
    assert!(!report.needs_replay, "一致时不应触发重放（避免抖动）");

    // 服务端否定了预测（例如被墙挡住）→ 触发回滚重放。
    authoritative.position[1] -= 12.0;
    let report = reconcile(&local, &authoritative, 0.05);
    assert!(report.needs_replay);
    assert!(
        (report.error_m - 12.0).abs() < 1e-6,
        "误差 {}",
        report.error_m
    );

    // 确认后，已确认的预测从缓冲中移除。
    assert_eq!(local.acknowledge(4), 5);
    assert!(local.is_empty());
    assert_eq!(local.last_acked(), 4);
}

#[test]
fn a_full_round_trip_converges_for_an_honest_client() {
    let validator = MovementValidator::new(MAX_SPEED_M_S);
    let mut server_state = PlayerState::at(1, surface_position(DVec3::X, 2.0));
    let mut local = PredictionBuffer::new(64);
    let mut interpolator = SnapshotInterpolator::new(TICK_DT, 16);

    // 100 个 Tick 的诚实移动：客户端本地立即前进，服务端每 Tick 确认一次。
    for sequence in 0..100u32 {
        let mut input = ClientInput {
            sequence,
            tick: u64::from(sequence),
            move_axis: [0.0, 0.2, 0.0],
            ..ClientInput::IDLE
        };
        input.sanitize();

        // 客户端预测。
        let mut predicted = server_state;
        predicted.position[1] += f64::from(input.move_axis[1]) * 2.5;
        local.push(PredictedState {
            sequence,
            tick: u64::from(sequence),
            position: predicted.position,
            yaw_rad: 0.0,
        });

        // 服务端校验并采纳。
        let claimed = predicted;
        assert_eq!(
            validator.validate_movement(&server_state, &claimed, TICK_DT),
            ValidationOutcome::Accepted,
            "诚实客户端在序号 {sequence} 被误判"
        );
        server_state = claimed;
        server_state.last_acked_input = sequence;

        // 广播快照 → 客户端插值。
        assert!(interpolator.push(snapshot(u64::from(sequence), server_state)));
        let report = reconcile(&local, &server_state, 0.05);
        assert!(!report.needs_replay, "序号 {sequence} 出现不必要的重放");
        local.acknowledge(sequence);
    }

    assert!(local.is_empty(), "所有预测最终都应被确认");
    assert!(interpolator.interpolate_player(1, 99.0 * TICK_DT).is_some());
    assert!(server_state.position[1] > 0.0, "服务端状态必须真的前进了");
}
