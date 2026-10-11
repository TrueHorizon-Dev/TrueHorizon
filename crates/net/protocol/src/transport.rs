//! `renet` 2.0 传输层接线：通道定义、连接配置与传输构造。
//!
//! 分工明确，互不渗透：
//!
//! * **`renet` 负责「字节怎么送到」**——握手、加密、重传、分片与乱序重组；
//! * **负载格式由 [`crate::codec`] 定义**——定点小端、逐字段判界。
//!
//! `renet` 的通道是**单向**的，客户端与服务端各自持有一张通道表，因此这里用两个
//! 强类型枚举（[`ClientChannel`] / [`ServerChannel`]）而不是裸 `u8`，
//! 避免「把快照发到输入通道」这类只能在联调时才发现的错误。

use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use renet::{ChannelConfig, ConnectionConfig, SendType};
use renet_netcode::{
    ClientAuthentication, NetcodeClientTransport, NetcodeServerTransport, ServerAuthentication,
    ServerConfig,
};

/// 默认 netcode 协议号：客户端与服务端必须一致，否则握手报文会被直接丢弃。
pub const DEFAULT_PROTOCOL_ID: u64 = 0x7A0E_2026;
/// 可靠通道的重传间隔。
pub const RESEND_TIME: Duration = Duration::from_millis(300);
/// 单个通道未确认消息的内存上限。
pub const CHANNEL_MEMORY_BYTES: usize = 2 * 1024 * 1024;
/// 默认每 Tick 发送预算（字节）：60 Hz 下约 28.8 Mbps。
pub const DEFAULT_BYTES_PER_TICK: u64 = 60_000;

/// 客户端 → 服务端通道。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientChannel {
    /// 输入。**不可靠**：丢一帧只损失一次采样，重传反而把延迟叠加到后续帧上。
    Input,
    /// 方块修改请求。可靠无序：顺序由服务端权威重排，先到先校验。
    EditRequest,
    /// 握手与保活。可靠有序。
    Control,
}

impl From<ClientChannel> for u8 {
    fn from(channel: ClientChannel) -> Self {
        match channel {
            ClientChannel::Input => 0,
            ClientChannel::EditRequest => 1,
            ClientChannel::Control => 2,
        }
    }
}

/// 服务端 → 客户端通道。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerChannel {
    /// 世界快照。**不可靠**：新快照本身就是旧快照的替代品，重传只会加重拥塞。
    Snapshot,
    /// Chunk 差量。可靠有序：建筑改动丢一条就会与世界产生永久分歧。
    ChunkDelta,
    /// 握手结果、状态纠正与拒绝。可靠有序。
    Control,
}

impl From<ServerChannel> for u8 {
    fn from(channel: ServerChannel) -> Self {
        match channel {
            ServerChannel::Snapshot => 0,
            ServerChannel::ChunkDelta => 1,
            ServerChannel::Control => 2,
        }
    }
}

/// 传输层配置。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportConfig {
    /// netcode 协议号。
    pub protocol_id: u64,
    /// 服务端最大连接数。
    pub max_clients: usize,
    /// 每 Tick 发送预算（字节）。
    pub bytes_per_tick: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            protocol_id: DEFAULT_PROTOCOL_ID,
            max_clients: 64,
            bytes_per_tick: DEFAULT_BYTES_PER_TICK,
        }
    }
}

impl TransportConfig {
    /// 按协议号与最大连接数构造。
    pub fn new(protocol_id: u64, max_clients: usize) -> Self {
        Self {
            protocol_id,
            max_clients,
            bytes_per_tick: DEFAULT_BYTES_PER_TICK,
        }
    }

    /// `renet` 连接配置（两张通道表）。
    pub fn connection_config(&self) -> ConnectionConfig {
        ConnectionConfig {
            available_bytes_per_tick: self.bytes_per_tick,
            server_channels_config: server_channels(),
            client_channels_config: client_channels(),
        }
    }

    /// 服务端：绑定端口并建立 netcode 传输。
    ///
    /// `public_addresses` 是**对客户端可见**的地址（NAT 后与本地绑定地址不同）。
    /// 留空时若绑定在通配地址（`0.0.0.0`）上，会自动退化成回环地址——
    /// 这只适用于本机联调；生产环境必须显式给出公网/内网可达地址，
    /// 否则客户端会把连接令牌发给一个不可路由的地址。
    pub fn server_transport(
        &self,
        port: u16,
        public_addresses: Vec<SocketAddr>,
    ) -> Result<(NetcodeServerTransport, SocketAddr), TransportError> {
        let socket = bind_server(port)?;
        let local_addr = socket.local_addr()?;
        let advertised = if local_addr.ip().is_unspecified() {
            SocketAddr::new(
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                local_addr.port(),
            )
        } else {
            local_addr
        };
        let addresses = if public_addresses.is_empty() {
            vec![advertised]
        } else {
            public_addresses
        };
        let config = ServerConfig {
            current_time: current_time(),
            max_clients: self.max_clients,
            protocol_id: self.protocol_id,
            public_addresses: addresses,
            authentication: ServerAuthentication::Unsecure,
        };
        let transport = NetcodeServerTransport::new(config, socket)?;
        Ok((transport, advertised))
    }

    /// 客户端：绑定随机本地端口并建立 netcode 传输。
    pub fn client_transport(
        &self,
        server_addr: SocketAddr,
        client_id: u64,
    ) -> Result<NetcodeClientTransport, TransportError> {
        let socket = bind_client()?;
        self.client_transport_on(socket, server_addr, client_id)
    }

    /// 客户端：在已绑定的 socket 上建立 netcode 传输（测试需要固定端口时使用）。
    pub fn client_transport_on(
        &self,
        socket: UdpSocket,
        server_addr: SocketAddr,
        client_id: u64,
    ) -> Result<NetcodeClientTransport, TransportError> {
        let authentication = ClientAuthentication::Unsecure {
            server_addr,
            client_id,
            user_data: None,
            protocol_id: self.protocol_id,
        };
        NetcodeClientTransport::new(current_time(), authentication, socket)
            .map_err(|error| TransportError::Netcode(error.to_string()))
    }
}

/// 服务端 → 客户端的通道表。
fn server_channels() -> Vec<ChannelConfig> {
    vec![
        ChannelConfig {
            channel_id: ServerChannel::Snapshot.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::Unreliable,
        },
        ChannelConfig {
            channel_id: ServerChannel::ChunkDelta.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::ReliableOrdered {
                resend_time: RESEND_TIME,
            },
        },
        ChannelConfig {
            channel_id: ServerChannel::Control.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::ReliableOrdered {
                resend_time: RESEND_TIME,
            },
        },
    ]
}

/// 客户端 → 服务端的通道表。输入放在最前：预算是按顺序消耗的，输入优先于建筑改动。
fn client_channels() -> Vec<ChannelConfig> {
    vec![
        ChannelConfig {
            channel_id: ClientChannel::Input.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::Unreliable,
        },
        ChannelConfig {
            channel_id: ClientChannel::EditRequest.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::ReliableUnordered {
                resend_time: RESEND_TIME,
            },
        },
        ChannelConfig {
            channel_id: ClientChannel::Control.into(),
            max_memory_usage_bytes: CHANNEL_MEMORY_BYTES,
            send_type: SendType::ReliableOrdered {
                resend_time: RESEND_TIME,
            },
        },
    ]
}

/// 绑定服务端 UDP 端口（`0` 表示由系统分配）。
pub fn bind_server(port: u16) -> Result<UdpSocket, TransportError> {
    Ok(UdpSocket::bind(("0.0.0.0", port))?)
}

/// 绑定客户端 UDP socket（随机本地端口）。
pub fn bind_client() -> Result<UdpSocket, TransportError> {
    Ok(UdpSocket::bind("0.0.0.0:0")?)
}

/// 距 UNIX 纪元的时间。netcode 用它做重放保护与超时判定，**必须单调可信**。
pub fn current_time() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
}

/// 传输层错误。
#[derive(Debug)]
pub enum TransportError {
    /// UDP 绑定/收发失败。
    Io(std::io::Error),
    /// netcode 层拒绝（协议号、令牌、地址等）。
    Netcode(String),
}

impl core::fmt::Display for TransportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "UDP 传输失败: {error}"),
            Self::Netcode(message) => write!(f, "netcode 传输失败: {message}"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Netcode(_) => None,
        }
    }
}

impl From<std::io::Error> for TransportError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_are_unique_within_each_direction() {
        let client_ids: Vec<u8> = [
            ClientChannel::Input,
            ClientChannel::EditRequest,
            ClientChannel::Control,
        ]
        .iter()
        .map(|channel| u8::from(*channel))
        .collect();
        let mut sorted = client_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), client_ids.len(), "同一方向的通道 ID 必须唯一");

        let server_ids: Vec<u8> = [
            ServerChannel::Snapshot,
            ServerChannel::ChunkDelta,
            ServerChannel::Control,
        ]
        .iter()
        .map(|channel| u8::from(*channel))
        .collect();
        let mut sorted = server_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), server_ids.len());
    }

    #[test]
    fn transport_shape_matches_the_documented_guarantees() {
        let config = TransportConfig::default();
        let connection = config.connection_config();

        let input = &connection.client_channels_config[0];
        assert_eq!(input.channel_id, u8::from(ClientChannel::Input));
        assert!(matches!(input.send_type, SendType::Unreliable));

        let snapshot = &connection.server_channels_config[0];
        assert_eq!(snapshot.channel_id, u8::from(ServerChannel::Snapshot));
        assert!(matches!(snapshot.send_type, SendType::Unreliable));

        // 建筑改动必须可靠：丢一条差量就是永久的客户/服务端分歧。
        let delta = &connection.server_channels_config[1];
        assert!(matches!(delta.send_type, SendType::ReliableOrdered { .. }));
        assert_eq!(
            connection.server_channels_config.len(),
            3,
            "服务端通道表必须与 ServerChannel 枚举一一对应"
        );
    }

    #[test]
    fn server_and_client_transports_bind_loopback() {
        let config = TransportConfig::default();
        let (transport, addr) = config
            .server_transport(0, Vec::new())
            .expect("绑定服务端端口");
        assert_eq!(transport.connected_clients(), 0);
        assert!(addr.port() > 0, "端口 0 必须被系统分配成真实端口");

        let client = config.client_transport(addr, 1).expect("建立客户端传输");
        assert_eq!(client.client_id(), 1);
        assert!(config.client_transport(addr, 2).is_ok());
    }

    #[test]
    fn protocol_id_is_part_of_the_handshake() {
        let config = TransportConfig::new(DEFAULT_PROTOCOL_ID + 1, 4);
        let (transport, _addr) = config.server_transport(0, Vec::new()).unwrap();
        assert_eq!(transport.max_clients(), 4);
        assert_eq!(config.protocol_id, DEFAULT_PROTOCOL_ID + 1);
    }
}
