//! # TrueHorizon 专用服务器
//!
//! 无图形界面的权威世界循环：固定 Tick 推进 + 输入校验 + 兴趣管理 + 流式加载调度。
//!
//! ```text
//! loop {
//!   收包（非阻塞） ──► 规整 + 校验 ──► 入队
//!   仿真固定 Tick ──► 脏 Chunk ──► 落盘 / 广播差量
//!   流式调度 ──► 生成任务
//!   定期广播快照（按 AoI 过滤）
//! }
//! ```
//!
//! 用法：
//!
//! ```text
//! dedicated-server [--port 27100] [--seed 42] [--ticks 0]
//! ```
//!
//! `--ticks N`（N > 0）时跑满 N 个 Tick 后正常退出，供 CI 做无头冒烟测试。

use std::error::Error;
use std::net::UdpSocket;
use std::time::{Duration, Instant};

use foundation::Profiler;
use geology::{GeologyConfig, GeologyGenerator};
use protocol::{PlayerState, ServerMessage, ServerSnapshot, PROTOCOL_VERSION};
use server_authority::{InterestManager, MovementValidator};
use simulation::{Rule, RuleAccess, RuleContext, Simulation, TickRate};
use streaming::{plan_near_field, StreamTask, StreamingConfig, TaskKind, TaskQueue};
use world_contracts::{ChunkCoord, StratumSampler};

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
                }
                "--seed" => {
                    config.seed = value.parse().map_err(|_| "种子必须是 u64".to_string())?
                }
                "--ticks" => {
                    config.max_ticks = value.parse().map_err(|_| "ticks 必须是 u64".to_string())?
                }
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

fn main() -> Result<(), Box<dyn Error>> {
    let config = ServerConfig::from_args()?;
    println!(
        "[server] TrueHorizon 专用服务器 协议 {} | 端口 {} | 种子 {} | Tick {} Hz",
        PROTOCOL_VERSION,
        config.port,
        config.seed,
        TickRate::Hz20.hz()
    );

    // 非阻塞 socket：主循环绝不能因为等包而丢 Tick。
    let socket = UdpSocket::bind(("0.0.0.0", config.port))?;
    socket.set_nonblocking(true)?;

    let geology: GeologyGenerator = GeologyGenerator::new(GeologyConfig {
        seed: config.seed,
        ..GeologyConfig::default()
    });
    let validator = MovementValidator::new(12.0);
    let aoi = InterestManager::new(config.planet_radius_m, 24.0, 128);
    let streaming = StreamingConfig::default();
    let mut tasks = TaskQueue::new(4096);
    let mut profiler = Profiler::with_capacity(4096);
    let mut simulation = Simulation::new(TickRate::Hz20, config.seed);
    if !simulation.register(Box::new(WeatherRule::new())) {
        return Err("规则注册失败：名称重复".into());
    }
    if !simulation.conflicts().is_empty() {
        return Err("规则的写集存在冲突，拒绝启动".into());
    }

    let mut buffer = [0u8; 2_048];
    let start = Instant::now();
    let mut last = start;
    let mut last_snapshot = start;
    let mut tick_total = 0u64;
    let mut packets_in = 0u64;

    loop {
        let now = Instant::now();
        let delta_s = (now - last).as_secs_f64();
        last = now;

        // 1) 收包：只做规整、计数与（TODO）解码校验，绝不阻塞。
        loop {
            match socket.recv_from(&mut buffer) {
                Ok((length, _peer)) => {
                    packets_in += 1;
                    if length == 0 {
                        continue;
                    }
                    // TODO(net): 解码 `ClientMessage` → `validator.sanitize_input` →
                    //            权威校验 → 入队输入。
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        let _ = validator.sanitize_input(protocol::ClientInput::IDLE);

        // 2) 推进仿真。
        let ticks = simulation.advance(delta_s);
        tick_total += u64::from(ticks);
        for coord in simulation.dirty_chunks() {
            // TODO(persistence): 收集差量并写入 WAL。
            // TODO(net): 对 AoI 内的客户端广播 ChunkDelta。
            let _ = tasks.submit(StreamTask::new(*coord, TaskKind::Mesh, 0.0));
        }

        // 3) 流式加载：以 Origin 附近的 Chunk 为中心规划。
        if ticks > 0 {
            let center = ChunkCoord::default();
            let planned = plan_near_field(center, &streaming, |_| false);
            for coord in planned.iter().take(streaming.max_tasks_per_frame) {
                tasks.submit(StreamTask::new(*coord, TaskKind::Generate { lod: 0 }, 0.0));
            }
            let batch = tasks.pop_batch(streaming.max_tasks_per_frame);
            if !batch.is_empty() {
                let _span = profiler.span("streaming.dispatch");
                let _ = batch.len();
            }
        }

        // 4) 定期广播快照（20 Hz）。
        if now.duration_since(last_snapshot) >= Duration::from_millis(50) {
            last_snapshot = now;
            let surface_radius_m =
                config.planet_radius_m + geology.surface_height_m(glam::DVec3::new(1.0, 0.0, 0.0));
            let _ = aoi.filter_visible([config.planet_radius_m, 0.0, 0.0], &[]);
            let snapshot = ServerSnapshot {
                tick: simulation.clock().tick(),
                server_time_s: start.elapsed().as_secs_f64(),
                origin: [surface_radius_m, 0.0, 0.0],
                players: Vec::<PlayerState>::new(),
                entities: Vec::new(),
            };
            let message = ServerMessage::Snapshot(snapshot);
            // TODO(net): 编码并按 AoI 逐客户端发送。
            std::hint::black_box(&message);
        }

        // 5) 退出条件（CI 冒烟）。
        if config.max_ticks > 0 && simulation.clock().tick() >= config.max_ticks {
            println!(
                "[server] 已跑满 {} 个 Tick：共 {} Tick，收包 {}，队列 {}，脏块 {}",
                config.max_ticks,
                tick_total,
                packets_in,
                tasks.len(),
                simulation.dirty_chunks().len()
            );
            for stat in profiler.stats() {
                println!(
                    "[server]   profiler {} calls={} total={}µs max={}µs",
                    stat.name, stat.calls, stat.total_micros, stat.max_micros
                );
            }
            return Ok(());
        }

        std::thread::sleep(Duration::from_millis(2));
    }
}
