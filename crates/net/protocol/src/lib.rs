//! # protocol —— 网络协议 DTO
//!
//! 只放**数据定义**：带版本的客户端预测输入与服务端事件。不含传输、不含序列化实现、
//! 不含任何引擎类型——因此这个 crate 可以被工具链、回放器、压测脚本独立复用。
//!
//! 版本策略：
//!
//! * [`ProtocolVersion`] 只接受**完全相等**的版本（拒绝半兼容，避免出现难以复现的
//!   状态分歧）；
//! * 报文结构变更必须同时提升 [`PROTOCOL_VERSION`]，并在 `docs/` 记录变更点。
//!
//! 模块：
//!
//! * [`messages`]：全部报文与字段类型。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod messages;

pub use messages::{
    BlockEdit, Buttons, ChunkDeltaMessage, ClientInput, ClientMessage, EntitySnapshot, PlayerState,
    ProtocolVersion, RejectReason, ServerMessage, ServerSnapshot, MAX_INPUT_RATE_HZ,
    MAX_PACKET_BYTES, PROTOCOL_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compatibility_requires_exact_match() {
        assert!(PROTOCOL_VERSION.is_compatible_with(ProtocolVersion(1)));
        assert!(!PROTOCOL_VERSION.is_compatible_with(ProtocolVersion(2)));
        assert_eq!(PROTOCOL_VERSION.to_string(), "v1");
    }

    #[test]
    fn sanitize_clamps_hostile_input() {
        let mut input = ClientInput {
            move_axis: [3.0, 4.0, 0.0],
            look_pitch_rad: 9.0,
            look_yaw_rad: f32::NAN,
            buttons: u32::MAX,
            ..ClientInput::IDLE
        };
        input.sanitize();
        let length: f32 = input.move_axis.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-5, "移动轴必须被截断到单位长度");
        assert!(input.look_pitch_rad <= std::f32::consts::FRAC_PI_2);
        assert_eq!(input.look_yaw_rad, 0.0, "NaN 必须被清零");
        assert_eq!(input.buttons, Buttons::ALL, "未知按钮位必须被清掉");

        let mut hostile = ClientInput {
            move_axis: [f32::INFINITY, 0.0, 0.0],
            ..ClientInput::IDLE
        };
        assert!(!hostile.is_finite());
        hostile.sanitize();
        assert!(hostile.is_finite());
        assert_eq!(hostile.move_axis, [0.0; 3]);
    }

    #[test]
    fn button_bits_round_trip_the_pressed_set() {
        let input = ClientInput {
            buttons: Buttons::JUMP | Buttons::PRIMARY,
            ..ClientInput::IDLE
        };
        let bits = input.button_bits();
        assert_eq!(bits.len(), 8);
        assert!(bits[0] && bits[2]);
        assert!(!bits[1] && !bits[3]);
        assert_eq!(bits.count_ones(), 2);
    }

    #[test]
    fn player_state_reports_radius_and_distance() {
        let a = PlayerState::at(1, [6_371_000.0, 0.0, 0.0]);
        let b = PlayerState::at(2, [6_371_000.0, 3.0, 4.0]);
        assert!((a.radius_m() - 6_371_000.0).abs() < 1e-6);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn reject_reasons_are_human_readable() {
        for reason in [
            RejectReason::VersionMismatch,
            RejectReason::RateLimited,
            RejectReason::Malformed,
            RejectReason::Unauthorized,
            RejectReason::OutOfRange,
        ] {
            assert!(!reason.as_str().is_empty());
        }
    }

    #[test]
    fn packet_budget_is_mtu_safe() {
        assert_eq!(MAX_PACKET_BYTES, 1_200);
        let rate = MAX_INPUT_RATE_HZ;
        assert!(
            (20..=240).contains(&rate),
            "输入频率上限必须落在 20..=240 Hz 的合理区间，实为 {rate}"
        );
        // 一个满编队快照的粗略体积必须能被拆包策略容纳。
        let snapshot = ServerSnapshot {
            tick: 1,
            server_time_s: 0.0,
            origin: [0.0; 3],
            players: (0..32).map(|id| PlayerState::at(id, [0.0; 3])).collect(),
            entities: Vec::new(),
        };
        assert_eq!(snapshot.player_count(), 32);
    }
}
