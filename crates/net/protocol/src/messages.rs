//! 网络报文 DTO。
//!
//! **纯数据**：不持有任何引擎类型，不使用 `glam`（避免 `serde` 特性耦合），
//! 一律用定长数组表示向量，使报文在带宽、对齐与跨语言实现上都最省事。
//!
//! 编码约定（待接入 codec 时生效）：
//!
//! * 字节序：小端；
//! * 浮点：`f32` 用于渲染/速度，`f64` 用于世界坐标（行星尺度下 `f32` 会丢精度）；
//! * 单包上限 [`MAX_PACKET_BYTES`]（1200 字节，UDP 安全 MTU，避免 IP 分片）。

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

/// 当前协议版本。
pub const PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion(1);

/// 单个数据报的安全上限（字节）。超过必须拆分——IP 分片会显著提升丢包代价。
pub const MAX_PACKET_BYTES: usize = 1_200;

/// 客户端输入的下发频率上限（Hz），服务端据此限流。
pub const MAX_INPUT_RATE_HZ: u32 = 60;

/// 协议版本。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct ProtocolVersion(pub u16);

impl ProtocolVersion {
    /// 是否与对端兼容。**只接受完全相同的主版本**：宁可拒绝连接，也不做半吊子兼容。
    #[inline]
    pub fn is_compatible_with(self, other: ProtocolVersion) -> bool {
        self.0 == other.0
    }
}

impl core::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// 输入按钮位（`ClientInput::buttons`）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Buttons;

impl Buttons {
    /// 跳跃。
    pub const JUMP: u32 = 1 << 0;
    /// 下蹲。
    pub const CROUCH: u32 = 1 << 1;
    /// 主交互（挖掘）。
    pub const PRIMARY: u32 = 1 << 2;
    /// 次交互（放置）。
    pub const SECONDARY: u32 = 1 << 3;
    /// 冲刺。
    pub const SPRINT: u32 = 1 << 4;
    /// 全部已定义按钮。
    pub const ALL: u32 = Self::JUMP | Self::CROUCH | Self::PRIMARY | Self::SECONDARY | Self::SPRINT;
}

/// 客户端输入（每 Tick 一个，带序号的预测基准）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClientInput {
    /// 输入序号（单调递增；服务端快照会回带 `last_acked_input` 以驱动和解）。
    pub sequence: u32,
    /// 客户端认为的仿真 Tick。
    pub tick: u64,
    /// 期望移动方向（本地切线标架，长度 ≤ 1）。
    pub move_axis: [f32; 3],
    /// 视角偏航（弧度）。
    pub look_yaw_rad: f32,
    /// 视角俯仰（弧度）。
    pub look_pitch_rad: f32,
    /// 按钮位，见 [`Buttons`]。
    pub buttons: u32,
}

impl ClientInput {
    /// 空输入。
    pub const IDLE: Self = Self {
        sequence: 0,
        tick: 0,
        move_axis: [0.0; 3],
        look_yaw_rad: 0.0,
        look_pitch_rad: 0.0,
        buttons: 0,
    };

    /// 全部字段均为有限值。
    pub fn is_finite(&self) -> bool {
        self.move_axis.iter().all(|value| value.is_finite())
            && self.look_yaw_rad.is_finite()
            && self.look_pitch_rad.is_finite()
    }

    /// 就地规整：移动轴截断到单位长度、俯仰夹到 `±π/2`、未知按钮位清零。
    ///
    /// 服务端**必须先规整再校验**，否则作弊客户端可以用 NaN 或超大向量绕过速度检查。
    pub fn sanitize(&mut self) {
        let length_squared: f32 = self.move_axis.iter().map(|value| value * value).sum();
        if !length_squared.is_finite() {
            self.move_axis = [0.0; 3];
        } else if length_squared > 1.0 {
            let scale = length_squared.sqrt();
            for axis in &mut self.move_axis {
                *axis /= scale;
            }
        }
        if !self.look_yaw_rad.is_finite() {
            self.look_yaw_rad = 0.0;
        }
        if !self.look_pitch_rad.is_finite() {
            self.look_pitch_rad = 0.0;
        }
        let limit = std::f32::consts::FRAC_PI_2;
        self.look_pitch_rad = self.look_pitch_rad.clamp(-limit, limit);
        self.buttons &= Buttons::ALL;
    }

    /// 按钮位图（紧凑编码用）。
    pub fn button_bits(&self) -> BitVec {
        let mut bits = BitVec::repeat(false, 8);
        for index in 0..8 {
            bits.set(index, self.buttons & (1 << index) != 0);
        }
        bits
    }
}

impl Default for ClientInput {
    fn default() -> Self {
        Self::IDLE
    }
}

/// 玩家状态（同时用于客户端上报与服务端快照）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    /// 玩家 ID。
    pub id: u32,
    /// 世界坐标（米，`f64`：行星尺度下 `f32` 无法表示米级精度）。
    pub position: [f64; 3],
    /// 速度（米/秒）。
    pub velocity: [f32; 3],
    /// 偏航。
    pub yaw_rad: f32,
    /// 俯仰。
    pub pitch_rad: f32,
    /// 服务端已确认的最后输入序号。
    pub last_acked_input: u32,
}

impl PlayerState {
    /// 构造原点处的玩家。
    pub fn at(id: u32, position: [f64; 3]) -> Self {
        Self {
            id,
            position,
            velocity: [0.0; 3],
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            last_acked_input: 0,
        }
    }

    /// 到另一点的距离（米）。
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 距行星质心的半径（米）。
    pub fn radius_m(&self) -> f64 {
        let [x, y, z] = self.position;
        (x * x + y * y + z * z).sqrt()
    }
}

/// 实体快照（非玩家实体：掉落物、生物、载具）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// 实体 ID。
    pub entity: u32,
    /// 位置（相对快照原点的 `f32`，原点由 `ServerSnapshot::origin` 给出）。
    pub position: [f32; 3],
    /// 实体类型。
    pub kind: u16,
}

/// 服务端快照。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServerSnapshot {
    /// 服务端 Tick。
    pub tick: u64,
    /// 服务端时间（秒），客户端据此做插值时钟对齐。
    pub server_time_s: f64,
    /// 快照原点（世界坐标，米）：实体位置相对它用 `f32` 表达。
    pub origin: [f64; 3],
    /// 玩家状态。
    pub players: Vec<PlayerState>,
    /// 实体。
    pub entities: Vec<EntitySnapshot>,
}

impl ServerSnapshot {
    /// 玩家数量。
    pub fn player_count(&self) -> usize {
        self.players.len()
    }
}

/// 一次方块修改。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockEdit {
    /// Chunk 局部线性下标（`0..32768`）。
    pub local_index: u16,
    /// 修改后的方块 ID。
    pub block: u16,
}

/// Chunk 差量报文。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkDeltaMessage {
    /// Chunk 坐标。
    pub coord: [i64; 3],
    /// 差量基于的版本号。
    pub base_version: u32,
    /// 修改列表。
    pub edits: Vec<BlockEdit>,
}

/// 拒绝原因。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum RejectReason {
    /// 协议版本不匹配。
    VersionMismatch,
    /// 触发限流。
    RateLimited,
    /// 报文结构非法。
    Malformed,
    /// 未通过权威校验（作弊嫌疑）。
    Unauthorized,
    /// 目标不可交互（越权修改、距离过远等）。
    OutOfRange,
}

impl RejectReason {
    /// 面向玩家的默认提示文本。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VersionMismatch => "客户端与服务器版本不一致",
            Self::RateLimited => "请求过于频繁",
            Self::Malformed => "报文格式非法",
            Self::Unauthorized => "操作未通过权威校验",
            Self::OutOfRange => "目标超出可交互范围",
        }
    }
}

/// 客户端 → 服务端。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMessage {
    /// 握手。
    Hello {
        /// 客户端协议版本。
        version: ProtocolVersion,
        /// 玩家昵称。
        name: String,
    },
    /// 输入。
    Input(ClientInput),
    /// 修改请求（服务端必须校验权限与距离）。
    EditRequest {
        /// Chunk 坐标。
        coord: [i64; 3],
        /// 修改列表。
        edits: Vec<BlockEdit>,
    },
    /// 保活。
    KeepAlive {
        /// 客户端 Tick。
        tick: u64,
    },
}

/// 服务端 → 客户端。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerMessage {
    /// 握手成功。
    Welcome {
        /// 服务端协议版本。
        version: ProtocolVersion,
        /// 分配给玩家的 ID。
        player_id: u32,
        /// 服务端 Tick 频率。
        tick_rate_hz: u16,
    },
    /// 世界快照。
    Snapshot(ServerSnapshot),
    /// Chunk 差量。
    ChunkDelta(ChunkDeltaMessage),
    /// 状态纠正（和解失败时下发）。
    Correction(PlayerState),
    /// 拒绝。
    Reject {
        /// 原因。
        reason: RejectReason,
    },
    /// 保活。
    KeepAlive {
        /// 服务端 Tick。
        tick: u64,
    },
}
