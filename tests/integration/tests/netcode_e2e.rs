//! renet 端到端：**真实 UDP 回环**上的握手、输入上行、快照下行、畸形报文与断开。
//!
//! 这些测试刻意不用任何 mock 传输：客户端与服务端各自跑真实的
//! `RenetClient`/`RenetServer` + netcode 传输，报文经真实 socket 收发。
//! 因此它们能覆盖「通道语义是否配对」「协议号是否真的生效」「畸形负载会不会拖垮连接」
//! 这类只有真跑一遍才会暴露的问题。
//!
//! 抗抖动原则：**每一条断言都轮询到期限**（默认 10 s），不做 `sleep(小常数) 然后断言`。

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use protocol::{
    decode_client, decode_server, encode_client, encode_server, ClientChannel, ClientMessage,
    PlayerState, ServerChannel, ServerMessage, TransportConfig, DEFAULT_PROTOCOL_ID,
};
use renet::{ClientId, RenetClient, RenetServer, ServerEvent};
use renet_netcode::{NetcodeClientTransport, NetcodeTransportError};
use server_authority::MovementValidator;

/// 轮询步长（约 100 Hz）。
const STEP: Duration = Duration::from_millis(10);
/// 单条断言的默认期限。
const DEADLINE: Duration = Duration::from_secs(10);

/// 服务端harness 上报给测试线程的事件。
#[derive(Clone, Debug, PartialEq)]
enum ServerEventReport {
    /// 有新客户端接入。
    Connected(ClientId),
    /// 收到并解码出握手。
    Hello { client_id: ClientId, name: String },
    /// 收到一条通过规整的输入。
    Input { client_id: ClientId, sequence: u32 },
    /// 收到无法解码的负载。
    Malformed { client_id: ClientId },
    /// 客户端断开。
    Disconnected(ClientId),
}

/// 测试用的服务端：真实 renet 服务端 + 真实 UDP socket，逻辑与 `dedicated-server` 同构。
struct ServerHarness {
    addr: SocketAddr,
    events: Receiver<ServerEventReport>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ServerHarness {
    fn spawn(protocol_id: u64) -> Self {
        let (addr_tx, addr_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            let config = TransportConfig::new(protocol_id, 8);
            let (mut transport, addr) = config
                .server_transport(0, Vec::new())
                .expect("绑定服务端端口");
            let mut server = RenetServer::new(config.connection_config());
            let mut players: HashMap<ClientId, PlayerState> = HashMap::new();
            let validator = MovementValidator::new(12.0);
            addr_tx.send(addr).expect("回传监听地址");

            let mut tick = 0u64;
            while !stop_flag.load(Ordering::Relaxed) {
                server.update(STEP);
                if let Err(error) = transport.update(STEP, &mut server) {
                    // 单个客户端的传输错误不该终止服务端。
                    eprintln!("harness transport error: {error}");
                }

                while let Some(event) = server.get_event() {
                    match event {
                        ServerEvent::ClientConnected { client_id } => {
                            players.insert(
                                client_id,
                                PlayerState::at(client_id as u32, [6_371_000.0, 0.0, 0.0]),
                            );
                            let _ = event_tx.send(ServerEventReport::Connected(client_id));
                        },
                        ServerEvent::ClientDisconnected { client_id, .. } => {
                            players.remove(&client_id);
                            let _ = event_tx.send(ServerEventReport::Disconnected(client_id));
                        },
                    }
                }

                for client_id in server.clients_id() {
                    while let Some(bytes) =
                        server.receive_message(client_id, ClientChannel::Control)
                    {
                        match decode_client(&bytes) {
                            Ok(ClientMessage::Hello { name, .. }) => {
                                let _ = event_tx.send(ServerEventReport::Hello {
                                    client_id,
                                    name: name.clone(),
                                });
                                let welcome = ServerMessage::Welcome {
                                    version: protocol::PROTOCOL_VERSION,
                                    player_id: client_id as u32,
                                    tick_rate_hz: 20,
                                };
                                server.send_message(
                                    client_id,
                                    ServerChannel::Control,
                                    encode_server(&welcome),
                                );
                            },
                            Ok(_) => {},
                            Err(_) => {
                                // 畸形负载：回拒绝，但不踢连接。
                                let _ = event_tx.send(ServerEventReport::Malformed { client_id });
                                let reject = ServerMessage::Reject {
                                    reason: protocol::RejectReason::Malformed,
                                };
                                server.send_message(
                                    client_id,
                                    ServerChannel::Control,
                                    encode_server(&reject),
                                );
                            },
                        }
                    }

                    while let Some(bytes) = server.receive_message(client_id, ClientChannel::Input)
                    {
                        let Ok(ClientMessage::Input(input)) = decode_client(&bytes) else {
                            continue;
                        };
                        let Some(sanitized) = validator.sanitize_input(input) else {
                            continue;
                        };
                        if let Some(player) = players.get_mut(&client_id) {
                            player.last_acked_input = sanitized.sequence;
                            player.yaw_rad = sanitized.look_yaw_rad;
                        }
                        let _ = event_tx.send(ServerEventReport::Input {
                            client_id,
                            sequence: sanitized.sequence,
                        });
                    }
                }

                // 20 Hz 快照；测试只关心「能不能收到并解出一致内容」。
                if tick % 2 == 0 {
                    for client_id in server.clients_id() {
                        let snapshot = ServerMessage::Snapshot(protocol::ServerSnapshot {
                            tick,
                            server_time_s: tick as f64 * 0.05,
                            origin: [6_371_000.0, 0.0, 0.0],
                            players: players.values().copied().collect(),
                            entities: Vec::new(),
                        });
                        server.send_message(
                            client_id,
                            ServerChannel::Snapshot,
                            encode_server(&snapshot),
                        );
                    }
                }

                transport.send_packets(&mut server);
                tick += 1;
                thread::sleep(STEP);
            }
        });

        let addr = addr_rx
            .recv_timeout(DEADLINE)
            .expect("服务端应在一个期限内报告监听地址");
        Self {
            addr,
            events: event_rx,
            stop,
            handle: Some(handle),
        }
    }

    /// 在期限内等待某个事件。
    fn wait_for(
        &self,
        predicate: impl Fn(&ServerEventReport) -> bool,
    ) -> Option<ServerEventReport> {
        let deadline = Instant::now() + DEADLINE;
        while Instant::now() < deadline {
            match self.events.recv_timeout(Duration::from_millis(100)) {
                Ok(event) => {
                    if predicate(&event) {
                        return Some(event);
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
        None
    }

    /// 期限内是否**没有**出现满足条件的事件。
    fn expect_absent(
        &self,
        duration: Duration,
        predicate: impl Fn(&ServerEventReport) -> bool,
    ) -> bool {
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            match self.events.try_recv() {
                Ok(event) => {
                    if predicate(&event) {
                        return false;
                    }
                },
                Err(TryRecvError::Empty) => thread::sleep(STEP),
                Err(TryRecvError::Disconnected) => break,
            }
        }
        true
    }
}

impl Drop for ServerHarness {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// 测试客户端：真实 renet 客户端 + 已解出的下行报文。
struct TestClient {
    client: RenetClient,
    transport: NetcodeClientTransport,
    control: Vec<ServerMessage>,
    snapshots: Vec<ServerMessage>,
    /// 最近一次传输错误（诊断用，不吞掉原因）。
    last_error: Option<String>,
}

impl TestClient {
    fn connect(server_addr: SocketAddr, protocol_id: u64, client_id: ClientId) -> Self {
        let config = TransportConfig::new(protocol_id, 8);
        Self {
            client: RenetClient::new(config.connection_config()),
            transport: config
                .client_transport(server_addr, client_id)
                .expect("建立客户端传输"),
            control: Vec::new(),
            snapshots: Vec::new(),
            last_error: None,
        }
    }

    /// 推进一帧并抽干下行通道。
    fn step(&mut self) -> Result<(), NetcodeTransportError> {
        self.client.update(STEP);
        self.transport.update(STEP, &mut self.client)?;

        let mut control = Vec::new();
        while let Some(bytes) = self.client.receive_message(ServerChannel::Control) {
            control.push(bytes);
        }
        for bytes in control {
            if let Ok(message) = decode_server(&bytes) {
                self.control.push(message);
            }
        }
        let mut snapshots = Vec::new();
        while let Some(bytes) = self.client.receive_message(ServerChannel::Snapshot) {
            snapshots.push(bytes);
        }
        for bytes in snapshots {
            if let Ok(message) = decode_server(&bytes) {
                self.snapshots.push(message);
            }
        }

        self.transport.send_packets(&mut self.client)?;
        thread::sleep(STEP);
        Ok(())
    }

    /// 轮询到条件成立或超时。
    ///
    /// 传输错误会被记进 [`TestClient::last_error`] 并立刻结束轮询——否则测试只会
    /// 报一句「没连上」，把真正的原因（协议号？地址？端口？）藏起来。
    fn run_until(&mut self, timeout: Duration, mut predicate: impl FnMut(&Self) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate(self) {
                return true;
            }
            if let Err(error) = self.step() {
                self.last_error = Some(error.to_string());
                return false;
            }
        }
        predicate(self)
    }

    fn send(
        &mut self,
        channel: ClientChannel,
        message: &ClientMessage,
    ) -> Result<(), NetcodeTransportError> {
        self.client.send_message(channel, encode_client(message));
        self.transport.send_packets(&mut self.client)
    }

    /// 直接发原始字节（用于畸形负载测试）。
    fn send_raw(
        &mut self,
        channel: ClientChannel,
        bytes: Vec<u8>,
    ) -> Result<(), NetcodeTransportError> {
        self.client.send_message(channel, bytes);
        self.transport.send_packets(&mut self.client)
    }

    fn welcome_seen(&self) -> bool {
        self.control
            .iter()
            .any(|message| matches!(message, ServerMessage::Welcome { .. }))
    }

    fn last_snapshot(&self) -> Option<&ServerSnapshot> {
        self.snapshots
            .iter()
            .rev()
            .find_map(|message| match message {
                ServerMessage::Snapshot(snapshot) => Some(snapshot),
                _ => None,
            })
    }
}

use protocol::ServerSnapshot;

#[test]
fn client_and_server_complete_a_real_session_over_udp() {
    let harness = ServerHarness::spawn(DEFAULT_PROTOCOL_ID);
    let mut client = TestClient::connect(harness.addr, DEFAULT_PROTOCOL_ID, 1);

    // 1) 握手：客户端必须真的连上，服务端必须收到 Hello，客户端必须收到 Welcome。
    assert!(
        client.run_until(DEADLINE, |client| client.client.is_connected()),
        "客户端必须完成 netcode 握手（最近错误: {:?}）",
        client.last_error
    );
    assert!(
        harness
            .wait_for(|event| matches!(event, ServerEventReport::Connected(_)))
            .is_some(),
        "服务端必须观察到客户端接入"
    );

    client
        .send(
            ClientChannel::Control,
            &ClientMessage::Hello {
                version: protocol::PROTOCOL_VERSION,
                name: "联调机器人".to_string(),
            },
        )
        .expect("发送握手");
    let hello = harness.wait_for(|event| matches!(event, ServerEventReport::Hello { .. }));
    match hello {
        Some(ServerEventReport::Hello { name, .. }) => assert_eq!(name, "联调机器人"),
        other => panic!("服务端必须收到握手，实际: {other:?}"),
    }
    assert!(
        client.run_until(DEADLINE, |client| client.welcome_seen()),
        "客户端必须收到 Welcome"
    );

    // 2) 输入上行：不可靠通道也必须到达（回环上没有丢包）。
    client
        .send(
            ClientChannel::Input,
            &ClientMessage::Input(protocol::ClientInput {
                sequence: 7,
                tick: 3,
                move_axis: [1.0, 0.0, 0.0],
                look_yaw_rad: 0.5,
                look_pitch_rad: 0.0,
                buttons: 0,
            }),
        )
        .expect("发送输入");
    let input = harness.wait_for(|event| matches!(event, ServerEventReport::Input { .. }));
    match input {
        Some(ServerEventReport::Input { sequence, .. }) => assert_eq!(sequence, 7),
        other => panic!("服务端必须收到输入，实际: {other:?}"),
    }

    // 3) 快照下行：客户端必须收到并解出快照，且**最终**要看到服务端确认了那次输入
    //    （输入上行与快照下行是两条独立通道，因此这里断言的是收敛而不是同一帧）。
    assert!(
        client.run_until(DEADLINE, |client| client
            .last_snapshot()
            .map(|snapshot| snapshot
                .players
                .iter()
                .any(|player| player.last_acked_input == 7))
            .unwrap_or(false)),
        "快照必须最终反映服务端已确认的输入序号 7，实际: {:?}（最近错误: {:?}）",
        client
            .last_snapshot()
            .map(|snapshot| snapshot.players.clone()),
        client.last_error
    );
    let snapshot = client.last_snapshot().expect("快照").clone();
    assert_eq!(snapshot.origin, [6_371_000.0, 0.0, 0.0]);

    // 4) 断开：服务端必须观察到（netcode 会发出断开报文）。
    client.transport.disconnect();
    let disconnected =
        harness.wait_for(|event| matches!(event, ServerEventReport::Disconnected(_)));
    assert!(disconnected.is_some(), "服务端必须观察到断开");
}

#[test]
fn mismatched_protocol_id_never_connects() {
    let harness = ServerHarness::spawn(DEFAULT_PROTOCOL_ID);
    // 客户端用另一个协议号：netcode 必须直接丢弃它的报文。
    let mut client = TestClient::connect(harness.addr, DEFAULT_PROTOCOL_ID + 1, 9);
    let connected = client.run_until(Duration::from_secs(3), |client| {
        client.client.is_connected()
    });
    assert!(!connected, "协议号不一致时绝不允许连上");
    assert!(
        harness.expect_absent(Duration::from_secs(1), |event| matches!(
            event,
            ServerEventReport::Connected(_)
        )),
        "协议号不一致时服务端不应出现接入事件"
    );
}

#[test]
fn malformed_payload_gets_a_reject_not_a_disconnect() {
    let harness = ServerHarness::spawn(DEFAULT_PROTOCOL_ID);
    let mut client = TestClient::connect(harness.addr, DEFAULT_PROTOCOL_ID, 2);
    assert!(
        client.run_until(DEADLINE, |client| client.client.is_connected()),
        "客户端必须完成 netcode 握手（最近错误: {:?}）",
        client.last_error
    );

    // 未知报文类型标签：服务端必须回 Reject(Malformed)，并且连接保持。
    client
        .send_raw(ClientChannel::Control, vec![0xEE, 0x01, 0x02])
        .expect("发送畸形负载");
    assert!(
        harness
            .wait_for(|event| matches!(event, ServerEventReport::Malformed { .. }))
            .is_some(),
        "服务端必须识别出畸形负载"
    );
    assert!(
        client.run_until(DEADLINE, |client| client.control.iter().any(|message| {
            matches!(
                message,
                ServerMessage::Reject {
                    reason: protocol::RejectReason::Malformed
                }
            )
        })),
        "客户端必须收到 Malformed 拒绝，实际: {:?}",
        client.control
    );
    assert!(
        client.client.is_connected(),
        "畸形负载只应被拒绝，不应断开连接"
    );
}
