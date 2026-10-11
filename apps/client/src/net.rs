//! 客户端网络会话：`renet` 客户端 + netcode 传输，输入上行 / 快照下行。
//!
//! 本模块只做**传输与解码**，不做预测与和解——那是 `client-sync` 的职责。
//! 之所以仍把收到的报文原样交回上层，是为了让预测/和解的接入点保持显式（而不是藏在这里）。

use std::error::Error;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use protocol::{
    decode_server, encode_client, ClientChannel, ClientInput, ClientMessage, ServerChannel,
    ServerMessage, ServerSnapshot, TransportConfig, MAX_INPUT_RATE_HZ,
};
use renet::RenetClient;
use renet_netcode::NetcodeClientTransport;

/// 一次客户端网络会话。
pub struct NetSession {
    client: RenetClient,
    transport: NetcodeClientTransport,
    server: SocketAddr,
    sequence: u32,
    last_input_sent: Option<Instant>,
    received: Vec<ServerMessage>,
}

impl NetSession {
    /// 连接服务端；`server` 形如 `127.0.0.1:27100`。
    pub fn connect(server: &str, client_id: u64) -> Result<Self, Box<dyn Error>> {
        let server: SocketAddr = server
            .parse()
            .map_err(|_| format!("服务端地址必须是 IP:PORT 形式，实际: {server}"))?;
        let config = TransportConfig::default();
        Ok(Self {
            client: RenetClient::new(config.connection_config()),
            transport: config.client_transport(server, client_id)?,
            server,
            sequence: 0,
            last_input_sent: None,
            received: Vec::new(),
        })
    }

    /// 目标服务端地址。
    pub fn server(&self) -> SocketAddr {
        self.server
    }

    /// 是否已完成 netcode 握手。
    pub fn is_connected(&self) -> bool {
        self.client.is_connected()
    }

    /// 主动断开（进程退出时调用，让服务端立刻看到断开而不是等超时）。
    pub fn disconnect(&mut self) {
        self.transport.disconnect();
        // 断开报文要立刻发出去；发送失败只意味着对端也走了，没有补救动作。
        let _ = self.transport.send_packets(&mut self.client);
    }

    /// 推进一帧：收包 → 解码 → 按 60 Hz 上限上行输入 → 冲刷发送队列。
    ///
    /// `dt` 是真实帧间隔：传输层用它做重传与超时判定，必须传真实值。
    pub fn update(
        &mut self,
        dt: Duration,
        move_axis: [f32; 3],
        look: [f32; 2],
    ) -> Result<&[ServerMessage], Box<dyn Error>> {
        self.client.update(dt);
        self.transport.update(dt, &mut self.client)?;

        // 先抽干通道再解码：`receive_message` 需要 &mut self.client，不能与解码交错。
        let mut payloads = Vec::new();
        while let Some(bytes) = self.client.receive_message(ServerChannel::Control) {
            payloads.push(bytes);
        }
        while let Some(bytes) = self.client.receive_message(ServerChannel::Snapshot) {
            payloads.push(bytes);
        }
        while let Some(bytes) = self.client.receive_message(ServerChannel::ChunkDelta) {
            payloads.push(bytes);
        }
        self.received.clear();
        for bytes in payloads {
            match decode_server(&bytes) {
                Ok(message) => self.received.push(message),
                Err(error) => eprintln!("[client] 下行报文非法，已丢弃: {error}"),
            }
        }

        if self.client.is_connected() && self.should_send_input() {
            self.sequence = self.sequence.wrapping_add(1);
            let input = ClientInput {
                sequence: self.sequence,
                // TODO(client): 用固定 Tick 的仿真时钟，而不是帧序号。
                tick: u64::from(self.sequence),
                move_axis,
                look_yaw_rad: look[0],
                look_pitch_rad: look[1],
                // TODO(input): 由键盘状态填充按钮位（当前只上行移动轴与视角）。
                buttons: 0,
            };
            self.client.send_message(
                ClientChannel::Input,
                encode_client(&ClientMessage::Input(input)),
            );
            self.last_input_sent = Some(Instant::now());
        }

        if self.client.is_connected() {
            self.transport.send_packets(&mut self.client)?;
        }
        Ok(&self.received)
    }

    /// 本帧收到的最新快照。
    pub fn latest_snapshot(&self) -> Option<&ServerSnapshot> {
        self.received
            .iter()
            .rev()
            .find_map(|message| match message {
                ServerMessage::Snapshot(snapshot) => Some(snapshot),
                _ => None,
            })
    }

    /// 是否到了下一个输入上行时刻（`MAX_INPUT_RATE_HZ` 上限）。
    fn should_send_input(&self) -> bool {
        let interval = Duration::from_millis(1_000 / u64::from(MAX_INPUT_RATE_HZ.max(1)));
        match self.last_input_sent {
            Some(sent) => sent.elapsed() >= interval,
            None => true,
        }
    }
}
