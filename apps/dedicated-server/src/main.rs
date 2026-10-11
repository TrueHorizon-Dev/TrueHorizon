//! # TrueHorizon 专用服务器
//!
//! 无图形界面的权威世界循环：**renet 传输** → 输入校验 → 固定 Tick 仿真 → 兴趣管理 → 快照广播。
//!
//! ```text
//! loop {
//!   renet.update()  ──► 解码 ClientMessage ──► 规整 + 权威校验 ──► 玩家状态
//!   固定 Tick 仿真  ──► 脏 Chunk ──► 流式任务（生成 / 网格）
//!   20 Hz 快照      ──► 按 AoI 逐客户端过滤 ──► 编码 ──► 广播
//!   transport.send_packets()
//! }
//! ```
//!
//! 传输层是 `renet` 2.0 + `netcode`（握手、加密、重传、分片由它负责）；
//! 负载由 `protocol::codec` 编解码。本程序只做**权威逻辑**。
//!
//! 用法：
//!
//! ```text
//! dedicated-server [--port 27100] [--seed 42] [--ticks 0]
//! ```
//!
//! `--ticks N`（N > 0）时跑满 N 个 Tick 后正常退出，供 CI 做无头冒烟测试。

use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};

use foundation::Profiler;
use geology::{GeologyConfig, GeologyGenerator};
use protocol::{
    decode_client, encode_server, ClientChannel, ClientInput, ClientMessage, PlayerState,
    ServerChannel, ServerMessage, ServerSnapshot, TransportConfig, PROTOCOL_VERSION,
};
use renet::{ClientId, RenetServer, ServerEvent};
use renet_netcode::NetcodeServerTransport;
use server_authority::{InterestManager, MovementValidator, ValidationOutcome};
use simulation::{Rule, RuleAccess, RuleContext, Simulation, TickRate};
use streaming::{plan_near_field, StreamTask, StreamingConfig, TaskKind, TaskQueue};
use world_contracts::{ChunkCoord, StratumSampler};

/// 步行速度（米/秒）。权威仿真用它推进玩家，校验器用它做上限。
const WALK_SPEED_M_S: f64 = 6.0;
/// 快照广播频率。
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(50);

/// 服务器启动参数。
#[derive(Clone, Copy, Debug, PartialEq)]
struct ServerConfig {
    port: u16,
    seed: u64,
    /// 跑满多少个 Tick 后退出；0 表示常驻。
    max_ticks: u64,
    planet_radius_m: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 27_100,
            seed: 0x7A0E_2026,
            max_ticks: 0,
            planet_radius_m: 6_371_000.0,
        }
    }
}

impl ServerConfig {
    /// 从命令行解析（只支持 `--key value`，参数错误直接报错而不是静默用默认值）。
    fn from_args() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args.next().ok_or_else(|| format!("参数 {flag} 缺少取值"))?;
            match flag.as_str() {
                "--port" => {
                    config.port = value
                        .parse()
                        .map_err(|_| "端口必须是 0..65535".to_string())?
                },
                "--seed" => {
                    config.seed = value.parse().map_err(|_| "种子必须是 u64".to_string())?
                },
                "--ticks" => {
                    config.max_ticks = value.parse().map_err(|_| "ticks 必须是 u64".to_string())?
                },
                other => return Err(format!("未知参数: {other}")),
            }
        }
        Ok(config)
    }
}

/// 示例规则：每 20 Hz 检查一次世界是否应该长草/结冰。
///
/// 它的存在是为了证明「规则注册 → 固定 Tick 调度 → 脏 Chunk 产出」这条链路是通的。
struct WeatherRule {
    name: &'static str,
    ticks_run: u64,
    dirty: Vec<ChunkCoord>,
}

impl WeatherRule {
    fn new() -> Self {
        Self {
            name: "rule.weather",
            ticks_run: 0,
            dirty: Vec::new(),
        }
    }
}

impl Rule for WeatherRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cadence(&self) -> TickRate {
        TickRate::Hz20
    }

    fn access(&self) -> RuleAccess {
        RuleAccess {
            reads: &["climate"],
            writes: &["blocks"],
        }
    }

    fn apply(&mut self, ctx: &mut RuleContext) {
        self.ticks_run += 1;
        // 每 20 Tick（约 1 秒）在原点附近标记一个 Chunk 为脏。
        if ctx.tick % 20 == 0 {
            let roll = ctx.next_random() % 64;
            let coord = ChunkCoord::new(roll as i64 - 32, 0, 0);
            ctx.mark_dirty(coord);
            self.dirty.push(coord);
        }
    }
}

/// 网络与仿真统计（退出时打印，同时是冒烟测试的断言依据）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Stats {
    ticks: u64,
    packets_in: u64,
    inputs_accepted: u64,
    inputs_corrected: u64,
    inputs_rejected: u64,
    edits_requested: u64,
    snapshots_sent: u64,
    clients_connected: u64,
    clients_disconnected: u64,
}

/// 权威世界服务器。
struct GameServer {
    config: ServerConfig,
    server: RenetServer,
    transport: NetcodeServerTransport,
    players: HashMap<ClientId, PlayerState>,
    validator: MovementValidator,
    aoi: InterestManager,
    simulation: Simulation,
    tasks: TaskQueue,
    profiler: Profiler,
    geology: GeologyGenerator,
    streaming: StreamingConfig,
    stats: Stats,
}

impl GameServer {
    fn new(config: ServerConfig) -> Result<Self, Box<dyn Error>> {
        let transport_config = TransportConfig::default();
        let (transport, local_addr) = transport_config.server_transport(config.port, Vec::new())?;
        println!(
            "[server] TrueHorizon 专用服务器 协议 {} | 监听 {} | 种子 {} | Tick {} Hz",
            PROTOCOL_VERSION,
            local_addr,
            config.seed,
            TickRate::Hz20.hz()
        );

        let geology = GeologyGenerator::new(GeologyConfig {
            seed: config.seed,
            ..GeologyConfig::default()
        });
        let mut simulation = Simulation::new(TickRate::Hz20, config.seed);
        if !simulation.register(Box::new(WeatherRule::new())) {
            return Err("规则注册失败：名称重复".into());
        }
        if !simulation.conflicts().is_empty() {
            return Err("规则的写集存在冲突，拒绝启动".into());
        }

        Ok(Self {
            config,
            server: RenetServer::new(transport_config.connection_config()),
            transport,
            players: HashMap::new(),
            // 校验器上限留出冲刺余量：步行 6 m/s，允许 12 m/s。
            validator: MovementValidator::new(WALK_SPEED_M_S * 2.0),
            aoi: InterestManager::new(config.planet_radius_m, 24.0, 128),
            simulation,
            tasks: TaskQueue::new(4096),
            profiler: Profiler::with_capacity(4096),
            geology,
            streaming: StreamingConfig::default(),
            stats: Stats::default(),
        })
    }

    /// 出生点：赤道方向的海平面上方 2 m。
    fn spawn_position(&self) -> [f64; 3] {
        let height = self
            .geology
            .surface_height_m(glam::DVec3::new(1.0, 0.0, 0.0));
        [self.config.planet_radius_m + height + 2.0, 0.0, 0.0]
    }

    /// 处理连接/断开事件。
    fn handle_events(&mut self) {
        while let Some(event) = self.server.get_event() {
            match event {
                ServerEvent::ClientConnected { client_id } => {
                    let spawn = self.spawn_position();
                    self.players
                        .insert(client_id, PlayerState::at(client_id as u32, spawn));
                    self.stats.clients_connected += 1;
                    let welcome = ServerMessage::Welcome {
                        version: PROTOCOL_VERSION,
                        player_id: client_id as u32,
                        tick_rate_hz: TickRate::Hz20.hz() as u16,
                    };
                    self.server.send_message(
                        client_id,
                        ServerChannel::Control,
                        encode_server(&welcome),
                    );
                    println!("[server] 客户端 {client_id} 已连接（出生点 {spawn:?}）");
                },
                ServerEvent::ClientDisconnected { client_id, reason } => {
                    self.players.remove(&client_id);
                    self.stats.clients_disconnected += 1;
                    println!("[server] 客户端 {client_id} 断开：{reason}");
                },
            }
        }
    }

    /// 收取并处理所有客户端报文。
    fn receive_messages(&mut self) {
        for client_id in self.server.clients_id() {
            while let Some(bytes) = self.server.receive_message(client_id, ClientChannel::Input) {
                self.stats.packets_in += 1;
                self.handle_client_payload(client_id, &bytes, 1.0 / 20.0);
            }
            while let Some(bytes) = self
                .server
                .receive_message(client_id, ClientChannel::EditRequest)
            {
                self.stats.packets_in += 1;
                self.handle_client_payload(client_id, &bytes, 1.0 / 20.0);
            }
            while let Some(bytes) = self
                .server
                .receive_message(client_id, ClientChannel::Control)
            {
                self.stats.packets_in += 1;
                self.handle_client_payload(client_id, &bytes, 1.0 / 20.0);
            }
        }
    }

    /// 解码一条客户端负载并按类型分派。
    fn handle_client_payload(&mut self, client_id: ClientId, bytes: &[u8], delta_s: f64) {
        let message = match decode_client(bytes) {
            Ok(message) => message,
            Err(error) => {
                // 解码失败必须是**拒绝**而不是断开：畸形报文只说明对端有 bug 或被篡改。
                println!("[server] 客户端 {client_id} 报文非法（{error}），已拒绝");
                let reject = ServerMessage::Reject {
                    reason: protocol::RejectReason::Malformed,
                };
                self.server
                    .send_message(client_id, ServerChannel::Control, encode_server(&reject));
                return;
            },
        };

        match message {
            ClientMessage::Hello { version, name } => {
                println!("[server] 客户端 {client_id} 握手：{name}（协议 {version}）");
            },
            ClientMessage::Input(input) => self.apply_input(client_id, input, delta_s),
            ClientMessage::EditRequest { coord, edits } => {
                self.stats.edits_requested += edits.len() as u64;
                // TODO(authority): 逐条按交互距离校验 → 写入 Chunk → 广播 ChunkDelta。
                // 目前只做规模统计与日志，避免半成品写入污染权威世界。
                let _ = coord;
            },
            ClientMessage::KeepAlive { .. } => {},
        }
    }

    /// 权威移动：规整 → 预测 → 校验 → 接受/纠正/拒绝。
    fn apply_input(&mut self, client_id: ClientId, input: ClientInput, delta_s: f64) {
        let Some(sanitized) = self.validator.sanitize_input(input) else {
            self.stats.inputs_rejected += 1;
            let reject = ServerMessage::Reject {
                reason: protocol::RejectReason::Malformed,
            };
            self.server
                .send_message(client_id, ServerChannel::Control, encode_server(&reject));
            return;
        };
        let Some(player) = self.players.get_mut(&client_id) else {
            return;
        };
        let previous = *player;
        let mut claimed = previous;
        claimed.last_acked_input = sanitized.sequence;
        claimed.yaw_rad = sanitized.look_yaw_rad;
        claimed.pitch_rad = sanitized.look_pitch_rad;
        claimed.position = step_on_sphere(previous.position, sanitized.move_axis, delta_s);

        match self
            .validator
            .validate_movement(&previous, &claimed, delta_s)
        {
            ValidationOutcome::Accepted => {
                *player = claimed;
                self.stats.inputs_accepted += 1;
            },
            ValidationOutcome::Corrected { max_speed_m_s } => {
                // 服务端说了算：按允许的最大速度重算，并下发纠正。
                let speed = (max_speed_m_s / WALK_SPEED_M_S).clamp(0.0, 1.0);
                let scaled = [
                    sanitized.move_axis[0] * speed as f32,
                    sanitized.move_axis[1] * speed as f32,
                    sanitized.move_axis[2] * speed as f32,
                ];
                claimed.position = step_on_sphere(previous.position, scaled, delta_s);
                *player = claimed;
                self.stats.inputs_corrected += 1;
                let correction = ServerMessage::Correction(claimed);
                self.server.send_message(
                    client_id,
                    ServerChannel::Control,
                    encode_server(&correction),
                );
            },
            ValidationOutcome::Rejected { reason } => {
                self.stats.inputs_rejected += 1;
                let reject = ServerMessage::Reject { reason };
                self.server
                    .send_message(client_id, ServerChannel::Control, encode_server(&reject));
                // 以服务端状态纠正，避免客户端在本地越走越远。
                let correction = ServerMessage::Correction(previous);
                self.server.send_message(
                    client_id,
                    ServerChannel::Control,
                    encode_server(&correction),
                );
            },
        }
    }

    /// 按 AoI 逐客户端过滤后广播快照。
    fn broadcast_snapshots(&mut self, server_time_s: f64) {
        let players: Vec<PlayerState> = self.players.values().copied().collect();
        for client_id in self.server.clients_id() {
            let Some(observer) = self.players.get(&client_id).copied() else {
                continue;
            };
            let visible = self.aoi.filter_visible(observer.position, &players);
            let snapshot = ServerSnapshot {
                tick: self.simulation.clock().tick(),
                server_time_s,
                origin: observer.position,
                players: players
                    .iter()
                    .filter(|player| visible.contains(&player.id))
                    .copied()
                    .collect(),
                entities: Vec::new(),
            };
            let payload = encode_server(&ServerMessage::Snapshot(snapshot));
            self.server
                .send_message(client_id, ServerChannel::Snapshot, payload);
            self.stats.snapshots_sent += 1;
        }
    }

    /// 跑起来；返回退出时的统计。
    fn run(&mut self) -> Result<Stats, Box<dyn Error>> {
        let start = Instant::now();
        let mut last = start;
        let mut last_snapshot = start;

        loop {
            let now = Instant::now();
            let delta_s = (now - last).as_secs_f64();
            last = now;

            // 1) 推进仿真（固定 Tick，内部按 TickRate 累积）。
            let ticks = self.simulation.advance(delta_s);
            self.stats.ticks += u64::from(ticks);
            for coord in self.simulation.dirty_chunks() {
                // TODO(persistence): 收集差量并写入 WAL。
                // TODO(net): 对 AoI 内的客户端广播 ChunkDelta。
                let _ = self
                    .tasks
                    .submit(StreamTask::new(*coord, TaskKind::Mesh, 0.0));
            }

            // 2) 网络：先收后发，顺序不能颠倒（先把本 Tick 的输入吃干净再产出快照）。
            {
                let _span = self.profiler.span("net.receive");
                self.server.update(Duration::from_secs_f64(delta_s));
                self.transport
                    .update(Duration::from_secs_f64(delta_s), &mut self.server)?;
            }
            self.handle_events();
            self.receive_messages();

            // 3) 流式加载：以原点附近的 Chunk 为中心规划。
            if ticks > 0 {
                let center = ChunkCoord::default();
                let planned = plan_near_field(center, &self.streaming, |_| false);
                for coord in planned.iter().take(self.streaming.max_tasks_per_frame) {
                    self.tasks
                        .submit(StreamTask::new(*coord, TaskKind::Generate { lod: 0 }, 0.0));
                }
                if !self.tasks.is_empty() {
                    let _span = self.profiler.span("streaming.dispatch");
                    let _ =
                        TaskQueue::pop_batch(&mut self.tasks, self.streaming.max_tasks_per_frame);
                }
            }

            // 4) 20 Hz 快照广播。
            if now.duration_since(last_snapshot) >= SNAPSHOT_INTERVAL {
                last_snapshot = now;
                let section_start = Instant::now();
                self.broadcast_snapshots(start.elapsed().as_secs_f64());
                self.profiler
                    .record("net.snapshot", section_start.elapsed());
            }

            // 5) 发送（包含可靠通道的重传）。
            {
                let _span = self.profiler.span("net.send");
                self.transport.send_packets(&mut self.server);
            }

            // 6) 退出条件（CI 冒烟）。
            if self.config.max_ticks > 0 && self.simulation.clock().tick() >= self.config.max_ticks
            {
                self.transport.disconnect_all(&mut self.server);
                self.transport.send_packets(&mut self.server);
                return Ok(self.stats);
            }

            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// 退出时的统计与 profiler 输出。
    fn report(&self) {
        let stats = self.stats;
        println!(
            "[server] 已跑满 {} 个 Tick：输入 {}/{}/{}（接受/纠正/拒绝），编辑请求 {}，快照 {}，连接 {}/{}，收包 {}",
            stats.ticks,
            stats.inputs_accepted,
            stats.inputs_corrected,
            stats.inputs_rejected,
            stats.edits_requested,
            stats.snapshots_sent,
            stats.clients_connected,
            stats.clients_disconnected,
            stats.packets_in
        );
        println!(
            "[server] 队列 {}，脏块 {}",
            self.tasks.len(),
            self.simulation.dirty_chunks().len()
        );
        for stat in self.profiler.stats() {
            println!(
                "[server]   profiler {} calls={} total={}µs max={}µs",
                stat.name, stat.calls, stat.total_micros, stat.max_micros
            );
        }
    }
}

/// 在球面上走一步：把本地切线标架里的移动轴换成世界坐标位移，再投影回球面。
///
/// 简化之处（有计划地留白）：不含重力、碰撞与地形贴地——那些属于
/// `crates/world/physics`，在接入权威移动之前这里只保证「位移在切平面内、半径不变」。
fn step_on_sphere(position: [f64; 3], move_axis: [f32; 3], delta_s: f64) -> [f64; 3] {
    let position = glam::DVec3::new(position[0], position[1], position[2]);
    let radius = position.length();
    if radius <= f64::EPSILON {
        return [0.0; 3];
    }
    let up = position / radius;
    // 选一个与 up 不平行的参考轴，构造切平面基。
    let reference = if up.x.abs() < 0.9 {
        glam::DVec3::X
    } else {
        glam::DVec3::Y
    };
    let east = up.cross(reference).normalize();
    let north = up.cross(east).normalize();
    let step = (east * f64::from(move_axis[0]) + north * f64::from(move_axis[2]))
        * WALK_SPEED_M_S
        * delta_s;
    // 沿切平面走一步后重新投影回同半径球面，避免长跑之后漂出球壳。
    let moved = position + step;
    let moved_radius = moved.length();
    if moved_radius <= f64::EPSILON {
        return [position.x, position.y, position.z];
    }
    let projected = moved / moved_radius * radius;
    [projected.x, projected.y, projected.z]
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = ServerConfig::from_args()?;
    let mut server = GameServer::new(config)?;
    server.run()?;
    server.report();
    Ok(())
}
