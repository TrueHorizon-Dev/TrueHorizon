//! 线格式编解码：手写小端二进制（逐字段判界）。
//!
//! 为什么手写而不是 `serde` 派生格式：
//!
//! * 报文要能被**不链接引擎**的工具（回放器、压测脚本、抓包脚本）解析，格式必须显式；
//! * 解码路径面对的是不可信输入。**「先按对端声明的长度分配，再读取」是典型的远程 DoS 入口**，
//!   这里因此坚持两步：先校验「剩余字节是否够读下一条」，再分配内存；
//! * `serde` 仍留在 DTO 上（便于将来接 JSON/自描述调试格式），但线上格式不依赖它。
//!
//! 传输层由 `renet` 负责（可靠/不可靠通道、分片重组），本模块只负责**负载字节**。

use crate::messages::{
    BlockEdit, ChunkDeltaMessage, ClientInput, ClientMessage, EntitySnapshot, PlayerState,
    ProtocolVersion, RejectReason, ServerMessage, ServerSnapshot,
};

/// 昵称的最大字节数。
pub const MAX_NAME_BYTES: usize = 64;
/// 单条快照的玩家数上限。
pub const MAX_SNAPSHOT_PLAYERS: usize = 256;
/// 单条快照的实体数上限。
pub const MAX_SNAPSHOT_ENTITIES: usize = 4_096;
/// 单条差量报文的修改数上限。
pub const MAX_EDITS_PER_MESSAGE: usize = 4_096;

const TAG_CLIENT_HELLO: u8 = 1;
const TAG_CLIENT_INPUT: u8 = 2;
const TAG_CLIENT_EDIT_REQUEST: u8 = 3;
const TAG_CLIENT_KEEP_ALIVE: u8 = 4;

const TAG_SERVER_WELCOME: u8 = 1;
const TAG_SERVER_SNAPSHOT: u8 = 2;
const TAG_SERVER_CHUNK_DELTA: u8 = 3;
const TAG_SERVER_CORRECTION: u8 = 4;
const TAG_SERVER_REJECT: u8 = 5;
const TAG_SERVER_KEEP_ALIVE: u8 = 6;

/// 编解码错误。全部是「拒绝这条报文」，没有任何路径允许 panic。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// 负载在读完之前就结束了。
    Truncated {
        /// 还需要的字节数。
        needed: usize,
        /// 实际剩余字节数。
        remaining: usize,
    },
    /// 报文类型标签未知（对端比本端新）。
    UnknownTag(u8),
    /// 对端声明的元素个数超出上限。
    TooLarge {
        /// 字段名。
        field: &'static str,
        /// 声明的数量。
        value: usize,
        /// 允许的上限。
        max: usize,
    },
    /// 字符串不是合法 UTF-8。
    InvalidUtf8,
    /// 尾部有多余字节（说明两端格式不一致）。
    TrailingBytes(usize),
    /// 枚举判别值非法。
    InvalidValue {
        /// 字段名。
        field: &'static str,
        /// 非法取值。
        value: u32,
    },
}

impl core::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { needed, remaining } => {
                write!(f, "报文被截断：还差 {needed} 字节，只剩 {remaining} 字节")
            },
            Self::UnknownTag(tag) => write!(f, "未知报文类型: {tag}"),
            Self::TooLarge { field, value, max } => {
                write!(f, "字段 {field} 声明 {value} 项，超出上限 {max}")
            },
            Self::InvalidUtf8 => write!(f, "字符串不是合法 UTF-8"),
            Self::TrailingBytes(count) => write!(f, "报文尾部有 {count} 个多余字节"),
            Self::InvalidValue { field, value } => write!(f, "字段 {field} 取值非法: {value}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// 编码客户端 → 服务端报文。
pub fn encode_client(message: &ClientMessage) -> Vec<u8> {
    let mut writer = Writer::new();
    match message {
        ClientMessage::Hello { version, name } => {
            writer.u8(TAG_CLIENT_HELLO);
            writer.version(*version);
            writer.string(name);
        },
        ClientMessage::Input(input) => {
            writer.u8(TAG_CLIENT_INPUT);
            writer.input(input);
        },
        ClientMessage::EditRequest { coord, edits } => {
            writer.u8(TAG_CLIENT_EDIT_REQUEST);
            for axis in coord {
                writer.i64(*axis);
            }
            writer.u32(edits.len() as u32);
            for edit in edits {
                writer.u16(edit.local_index);
                writer.u16(edit.block);
            }
        },
        ClientMessage::KeepAlive { tick } => {
            writer.u8(TAG_CLIENT_KEEP_ALIVE);
            writer.u64(*tick);
        },
    }
    writer.finish()
}

/// 解码客户端 → 服务端报文。
pub fn decode_client(bytes: &[u8]) -> Result<ClientMessage, ProtocolError> {
    let mut reader = Reader::new(bytes);
    let tag = reader.u8()?;
    let message = match tag {
        TAG_CLIENT_HELLO => ClientMessage::Hello {
            version: reader.version()?,
            name: reader.string(MAX_NAME_BYTES)?,
        },
        TAG_CLIENT_INPUT => ClientMessage::Input(reader.input()?),
        TAG_CLIENT_EDIT_REQUEST => {
            let coord = [reader.i64()?, reader.i64()?, reader.i64()?];
            let edits = reader.edits(MAX_EDITS_PER_MESSAGE)?;
            ClientMessage::EditRequest { coord, edits }
        },
        TAG_CLIENT_KEEP_ALIVE => ClientMessage::KeepAlive {
            tick: reader.u64()?,
        },
        other => return Err(ProtocolError::UnknownTag(other)),
    };
    reader.expect_end()?;
    Ok(message)
}

/// 编码服务端 → 客户端报文。
pub fn encode_server(message: &ServerMessage) -> Vec<u8> {
    let mut writer = Writer::new();
    match message {
        ServerMessage::Welcome {
            version,
            player_id,
            tick_rate_hz,
        } => {
            writer.u8(TAG_SERVER_WELCOME);
            writer.version(*version);
            writer.u32(*player_id);
            writer.u16(*tick_rate_hz);
        },
        ServerMessage::Snapshot(snapshot) => {
            writer.u8(TAG_SERVER_SNAPSHOT);
            writer.u64(snapshot.tick);
            writer.f64(snapshot.server_time_s);
            for axis in snapshot.origin {
                writer.f64(axis);
            }
            writer.u32(snapshot.players.len() as u32);
            for player in &snapshot.players {
                writer.player(player);
            }
            writer.u32(snapshot.entities.len() as u32);
            for entity in &snapshot.entities {
                writer.u32(entity.entity);
                for axis in entity.position {
                    writer.f32(axis);
                }
                writer.u16(entity.kind);
            }
        },
        ServerMessage::ChunkDelta(delta) => {
            writer.u8(TAG_SERVER_CHUNK_DELTA);
            for axis in delta.coord {
                writer.i64(axis);
            }
            writer.u32(delta.base_version);
            writer.u32(delta.edits.len() as u32);
            for edit in &delta.edits {
                writer.u16(edit.local_index);
                writer.u16(edit.block);
            }
        },
        ServerMessage::Correction(player) => {
            writer.u8(TAG_SERVER_CORRECTION);
            writer.player(player);
        },
        ServerMessage::Reject { reason } => {
            writer.u8(TAG_SERVER_REJECT);
            writer.u8(reason_tag(*reason));
        },
        ServerMessage::KeepAlive { tick } => {
            writer.u8(TAG_SERVER_KEEP_ALIVE);
            writer.u64(*tick);
        },
    }
    writer.finish()
}

/// 解码服务端 → 客户端报文。
pub fn decode_server(bytes: &[u8]) -> Result<ServerMessage, ProtocolError> {
    let mut reader = Reader::new(bytes);
    let tag = reader.u8()?;
    let message = match tag {
        TAG_SERVER_WELCOME => ServerMessage::Welcome {
            version: reader.version()?,
            player_id: reader.u32()?,
            tick_rate_hz: reader.u16()?,
        },
        TAG_SERVER_SNAPSHOT => ServerMessage::Snapshot(ServerSnapshot {
            tick: reader.u64()?,
            server_time_s: reader.f64()?,
            origin: [reader.f64()?, reader.f64()?, reader.f64()?],
            players: reader.players(MAX_SNAPSHOT_PLAYERS)?,
            entities: reader.entities(MAX_SNAPSHOT_ENTITIES)?,
        }),
        TAG_SERVER_CHUNK_DELTA => ServerMessage::ChunkDelta(ChunkDeltaMessage {
            coord: [reader.i64()?, reader.i64()?, reader.i64()?],
            base_version: reader.u32()?,
            edits: reader.edits(MAX_EDITS_PER_MESSAGE)?,
        }),
        TAG_SERVER_CORRECTION => ServerMessage::Correction(reader.player()?),
        TAG_SERVER_REJECT => ServerMessage::Reject {
            reason: reject_reason(reader.u8()?)?,
        },
        TAG_SERVER_KEEP_ALIVE => ServerMessage::KeepAlive {
            tick: reader.u64()?,
        },
        other => return Err(ProtocolError::UnknownTag(other)),
    };
    reader.expect_end()?;
    Ok(message)
}

/// 报文是否落在单数据报预算内（超出则由 `renet` 分片重组）。
pub fn fits_single_datagram(bytes: &[u8]) -> bool {
    bytes.len() <= crate::messages::MAX_PACKET_BYTES
}

fn reason_tag(reason: RejectReason) -> u8 {
    match reason {
        RejectReason::VersionMismatch => 1,
        RejectReason::RateLimited => 2,
        RejectReason::Malformed => 3,
        RejectReason::Unauthorized => 4,
        RejectReason::OutOfRange => 5,
    }
}

fn reject_reason(tag: u8) -> Result<RejectReason, ProtocolError> {
    Ok(match tag {
        1 => RejectReason::VersionMismatch,
        2 => RejectReason::RateLimited,
        3 => RejectReason::Malformed,
        4 => RejectReason::Unauthorized,
        5 => RejectReason::OutOfRange,
        other => {
            return Err(ProtocolError::InvalidValue {
                field: "RejectReason",
                value: u32::from(other),
            })
        },
    })
}

/// 写入端。
#[derive(Default)]
struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { out: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.out.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn version(&mut self, version: ProtocolVersion) {
        self.u16(version.0);
    }

    fn string(&mut self, value: &str) {
        let bytes = value.as_bytes();
        let length = bytes.len().min(MAX_NAME_BYTES);
        self.u16(length as u16);
        self.out.extend_from_slice(&bytes[..length]);
    }

    fn input(&mut self, input: &ClientInput) {
        self.u32(input.sequence);
        self.u64(input.tick);
        for axis in input.move_axis {
            self.f32(axis);
        }
        self.f32(input.look_yaw_rad);
        self.f32(input.look_pitch_rad);
        self.u32(input.buttons);
    }

    fn player(&mut self, player: &PlayerState) {
        self.u32(player.id);
        for axis in player.position {
            self.f64(axis);
        }
        for axis in player.velocity {
            self.f32(axis);
        }
        self.f32(player.yaw_rad);
        self.f32(player.pitch_rad);
        self.u32(player.last_acked_input);
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// 读取端：每次读取前都确认剩余字节足够。
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ProtocolError> {
        if self.remaining() < count {
            return Err(ProtocolError::Truncated {
                needed: count,
                remaining: self.remaining(),
            });
        }
        let slice = &self.bytes[self.position..self.position + count];
        self.position += count;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ProtocolError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ProtocolError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, ProtocolError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ProtocolError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, ProtocolError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(i64::from_le_bytes(bytes))
    }

    fn f32(&mut self) -> Result<f32, ProtocolError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(f32::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, ProtocolError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(f64::from_le_bytes(bytes))
    }

    fn version(&mut self) -> Result<ProtocolVersion, ProtocolError> {
        Ok(ProtocolVersion(self.u16()?))
    }

    fn string(&mut self, max: usize) -> Result<String, ProtocolError> {
        let length = usize::from(self.u16()?);
        if length > max {
            return Err(ProtocolError::TooLarge {
                field: "name",
                value: length,
                max,
            });
        }
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ProtocolError::InvalidUtf8)
    }

    fn input(&mut self) -> Result<ClientInput, ProtocolError> {
        Ok(ClientInput {
            sequence: self.u32()?,
            tick: self.u64()?,
            move_axis: [self.f32()?, self.f32()?, self.f32()?],
            look_yaw_rad: self.f32()?,
            look_pitch_rad: self.f32()?,
            buttons: self.u32()?,
        })
    }

    fn player(&mut self) -> Result<PlayerState, ProtocolError> {
        Ok(PlayerState {
            id: self.u32()?,
            position: [self.f64()?, self.f64()?, self.f64()?],
            velocity: [self.f32()?, self.f32()?, self.f32()?],
            yaw_rad: self.f32()?,
            pitch_rad: self.f32()?,
            last_acked_input: self.u32()?,
        })
    }

    /// 读取定长元素列表：**先校验剩余字节，再分配**。
    fn list<T>(
        &mut self,
        field: &'static str,
        max: usize,
        element_size: usize,
        mut read: impl FnMut(&mut Self) -> Result<T, ProtocolError>,
    ) -> Result<Vec<T>, ProtocolError> {
        let count = self.u32()? as usize;
        if count > max {
            return Err(ProtocolError::TooLarge {
                field,
                value: count,
                max,
            });
        }
        if self.remaining() < count.saturating_mul(element_size) {
            return Err(ProtocolError::Truncated {
                needed: count.saturating_mul(element_size),
                remaining: self.remaining(),
            });
        }
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(read(self)?);
        }
        Ok(items)
    }

    fn players(&mut self, max: usize) -> Result<Vec<PlayerState>, ProtocolError> {
        self.list("players", max, 52, |reader| reader.player())
    }

    fn entities(&mut self, max: usize) -> Result<Vec<EntitySnapshot>, ProtocolError> {
        self.list("entities", max, 18, |reader| {
            Ok(EntitySnapshot {
                entity: reader.u32()?,
                position: [reader.f32()?, reader.f32()?, reader.f32()?],
                kind: reader.u16()?,
            })
        })
    }

    fn edits(&mut self, max: usize) -> Result<Vec<BlockEdit>, ProtocolError> {
        self.list("edits", max, 4, |reader| {
            Ok(BlockEdit {
                local_index: reader.u16()?,
                block: reader.u16()?,
            })
        })
    }

    fn expect_end(&self) -> Result<(), ProtocolError> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(ProtocolError::TrailingBytes(self.remaining()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> ServerSnapshot {
        ServerSnapshot {
            tick: 9001,
            server_time_s: 12.5,
            origin: [6_371_000.0, -12.0, 33.0],
            players: vec![
                PlayerState {
                    id: 7,
                    position: [6_371_000.0, 1.0, 2.0],
                    velocity: [1.0, -2.0, 0.5],
                    yaw_rad: 0.25,
                    pitch_rad: -0.5,
                    last_acked_input: 42,
                },
                PlayerState::at(9, [6_371_001.0, 0.0, 0.0]),
            ],
            entities: vec![EntitySnapshot {
                entity: 3,
                position: [1.0, 2.0, 3.0],
                kind: 77,
            }],
        }
    }

    #[test]
    fn client_messages_round_trip() {
        let messages = [
            ClientMessage::Hello {
                version: ProtocolVersion(1),
                name: "挖矿的史蒂夫".to_string(),
            },
            ClientMessage::Input(ClientInput {
                sequence: 5,
                tick: 6,
                move_axis: [0.5, -0.25, 0.0],
                look_yaw_rad: 1.5,
                look_pitch_rad: -1.5,
                buttons: 0b1_0101,
            }),
            ClientMessage::EditRequest {
                coord: [-1, 2, -3],
                edits: vec![
                    BlockEdit {
                        local_index: 0,
                        block: 1,
                    },
                    BlockEdit {
                        local_index: 32_767,
                        block: 5,
                    },
                ],
            },
            ClientMessage::KeepAlive { tick: u64::MAX },
        ];
        for message in messages {
            let bytes = encode_client(&message);
            assert_eq!(decode_client(&bytes).unwrap(), message);
        }
    }

    #[test]
    fn server_messages_round_trip() {
        let messages = [
            ServerMessage::Welcome {
                version: ProtocolVersion(1),
                player_id: 3,
                tick_rate_hz: 60,
            },
            ServerMessage::Snapshot(sample_snapshot()),
            ServerMessage::ChunkDelta(ChunkDeltaMessage {
                coord: [4, 5, 6],
                base_version: 9,
                edits: vec![BlockEdit {
                    local_index: 7,
                    block: 2,
                }],
            }),
            ServerMessage::Correction(PlayerState::at(1, [1.0, 2.0, 3.0])),
            ServerMessage::Reject {
                reason: RejectReason::RateLimited,
            },
            ServerMessage::KeepAlive { tick: 1 },
        ];
        for message in messages {
            let bytes = encode_server(&message);
            assert_eq!(decode_server(&bytes).unwrap(), message);
        }
    }

    #[test]
    fn every_reject_reason_has_a_stable_tag() {
        for reason in [
            RejectReason::VersionMismatch,
            RejectReason::RateLimited,
            RejectReason::Malformed,
            RejectReason::Unauthorized,
            RejectReason::OutOfRange,
        ] {
            let bytes = encode_server(&ServerMessage::Reject { reason });
            assert_eq!(
                decode_server(&bytes).unwrap(),
                ServerMessage::Reject { reason }
            );
        }
        let error = decode_server(&[TAG_SERVER_REJECT, 99]).unwrap_err();
        assert!(matches!(
            error,
            ProtocolError::InvalidValue {
                field: "RejectReason",
                ..
            }
        ));
    }

    #[test]
    fn truncated_payloads_are_rejected_field_by_field() {
        let bytes = encode_server(&ServerMessage::Snapshot(sample_snapshot()));
        for cut in 0..bytes.len() {
            // 任何前缀都只能是错误或（恰好完整时的）合法报文，绝不允许 panic。
            if let Ok(message) = decode_server(&bytes[..cut]) {
                assert_eq!(cut, bytes.len(), "只有完整报文才允许解码成功");
                assert_eq!(message, ServerMessage::Snapshot(sample_snapshot()));
            }
        }
    }

    #[test]
    fn hostile_counts_are_refused_before_allocating() {
        // 声明 100 万个玩家，但只给了几个字节：必须在分配前拒绝。
        let mut bytes = vec![TAG_SERVER_SNAPSHOT];
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tick
        bytes.extend_from_slice(&0f64.to_le_bytes()); // server_time_s
        for _ in 0..3 {
            bytes.extend_from_slice(&0f64.to_le_bytes()); // origin
        }
        bytes.extend_from_slice(&1_000_000u32.to_le_bytes()); // players 数量
        let error = decode_server(&bytes).unwrap_err();
        assert!(
            matches!(
                error,
                ProtocolError::TooLarge {
                    field: "players",
                    ..
                }
            ),
            "超大计数必须在分配前被拒，实际: {error}"
        );

        // 数量在合法上限内、但负载不够：报截断而不是越界读取。
        let mut bytes = vec![TAG_SERVER_SNAPSHOT];
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        for _ in 0..3 {
            bytes.extend_from_slice(&0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let error = decode_server(&bytes).unwrap_err();
        assert!(matches!(error, ProtocolError::Truncated { .. }));
    }

    #[test]
    fn trailing_bytes_and_unknown_tags_are_reported() {
        let mut bytes = encode_client(&ClientMessage::KeepAlive { tick: 1 });
        bytes.push(0);
        assert!(matches!(
            decode_client(&bytes).unwrap_err(),
            ProtocolError::TrailingBytes(1)
        ));
        assert_eq!(
            decode_client(&[200]).unwrap_err(),
            ProtocolError::UnknownTag(200)
        );
        assert_eq!(
            decode_server(&[]).unwrap_err(),
            ProtocolError::Truncated {
                needed: 1,
                remaining: 0
            }
        );
    }

    #[test]
    fn overlong_names_are_refused_not_truncated_silently() {
        let name = "x".repeat(MAX_NAME_BYTES + 1);
        // 编码端截断到上限（防御本端 bug），解码端拒绝超长声明（防御对端）。
        let bytes = encode_client(&ClientMessage::Hello {
            version: ProtocolVersion(1),
            name: name.clone(),
        });
        assert_eq!(
            decode_client(&bytes).unwrap(),
            ClientMessage::Hello {
                version: ProtocolVersion(1),
                name: "x".repeat(MAX_NAME_BYTES)
            }
        );

        let mut hostile = vec![TAG_CLIENT_HELLO];
        hostile.extend_from_slice(&1u16.to_le_bytes());
        hostile.extend_from_slice(&(MAX_NAME_BYTES as u16 + 1).to_le_bytes());
        hostile.extend_from_slice("x".repeat(MAX_NAME_BYTES + 1).as_bytes());
        assert!(matches!(
            decode_client(&hostile).unwrap_err(),
            ProtocolError::TooLarge { field: "name", .. }
        ));
    }

    #[test]
    fn input_messages_fit_a_single_datagram() {
        let bytes = encode_client(&ClientMessage::Input(ClientInput::IDLE));
        assert!(fits_single_datagram(&bytes));
        assert_eq!(bytes.len(), 1 + 36, "输入报文是定长小报文");

        // 满编快照会超过单数据报预算——这正是交给 renet 分片而不是自己截断的原因。
        let mut snapshot = sample_snapshot();
        snapshot.players = (0..64).map(|id| PlayerState::at(id, [0.0; 3])).collect();
        let bytes = encode_server(&ServerMessage::Snapshot(snapshot));
        assert!(!fits_single_datagram(&bytes));
    }
}
