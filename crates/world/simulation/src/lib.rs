//! # simulation —— 规则仿真与 Mod 宿主
//!
//! 世界状态按**固定 Tick** 推进，规则以「频率 + 读写集」声明自己的节奏：
//!
//! ```text
//! 每帧 dt ──► FixedClock ──► n × Tick(20Hz / 60Hz)
//!                              │
//!                              ├─ 规则 A（20Hz，写：流体）
//!                              ├─ 规则 B（60Hz，写：玩家物理）
//!                              └─ Mod 规则（沙箱内，只能通过宿主函数读写）
//! ```
//!
//! 设计约束：
//!
//! * 规则**不得**依赖帧率或真实时间，只能读 `ctx.tick`；
//! * 写集必须声明，冲突的规则不得并行（骨架阶段仍串行执行，声明是为后续并行化留口）；
//! * 任何随机性都从 `ctx.rng` 取，保证重放一致（回放录像、服务端复算依赖此点）。
//!
//! 模块：
//!
//! * [`tick`]：固定步长时钟与插值因子。
//! * [`mods`]：`wasmtime` 沙箱宿主与 Mod 权限模型。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod mods;
pub mod tick;

use foundation::Profiler;
use world_contracts::ChunkCoord;

pub use mods::{ModError, ModHost, ModManifest, ModModule, ModPermissions, MOD_ABI_VERSION};
pub use tick::{FixedClock, TickRate};

/// 规则对世界数据的访问范围。
///
/// 声明写集的目的是**把并行化的边界显式化**：写集不相交的规则可以被调度到不同线程。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RuleAccess {
    /// 只读的子系统名（例如 `"blocks"`、`"players"`）。
    pub reads: &'static [&'static str],
    /// 写入的子系统名。
    pub writes: &'static [&'static str],
}

impl RuleAccess {
    /// 无访问（纯计算规则）。
    pub const NONE: Self = Self {
        reads: &[],
        writes: &[],
    };

    /// 两组写集是否冲突。
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.writes
            .iter()
            .any(|write| other.writes.contains(write) || other.reads.contains(write))
            || other.writes.iter().any(|write| self.reads.contains(write))
    }
}

/// 规则执行上下文。
///
/// 只暴露「读时钟 + 写脏块列表」这两件必要的事：规则拿不到 `WorldView`，
/// 想访问世界必须走显式注入的接口，从而保证依赖关系可审计。
#[derive(Debug)]
pub struct RuleContext {
    /// 当前 Tick 号。
    pub tick: u64,
    /// 本 Tick 由规则标记为脏的 Chunk（交给 `persistence` 落盘、`streaming` 重传）。
    pub dirty: Vec<ChunkCoord>,
    /// 随机数状态（splitmix64 状态字，规则可用 [`RuleContext::next_random`] 推进）。
    pub rng: u64,
}

impl RuleContext {
    /// 构造。
    pub fn new(tick: u64, rng: u64) -> Self {
        Self {
            tick,
            dirty: Vec::new(),
            rng,
        }
    }

    /// 取下一个确定性随机数并推进状态。
    pub fn next_random(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 标记一个 Chunk 为脏。
    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        if !self.dirty.contains(&coord) {
            self.dirty.push(coord);
        }
    }
}

/// 一条仿真规则。
pub trait Rule: Send {
    /// 规则名（用于性能剖析与冲突报告，必须唯一）。
    fn name(&self) -> &'static str;

    /// 执行频率。
    fn cadence(&self) -> TickRate;

    /// 读写集声明。
    fn access(&self) -> RuleAccess;

    /// 执行一次。
    fn apply(&mut self, ctx: &mut RuleContext);
}

/// 仿真调度器。
pub struct Simulation {
    clock: FixedClock,
    rules: Vec<Box<dyn Rule>>,
    context: RuleContext,
    profiler: Profiler,
    ticks_executed: u64,
}

impl Simulation {
    /// 以主频构造（交互物理频率）。默认 60 Hz、最多补跑 8 Tick。
    pub fn new(rate: TickRate, seed: u64) -> Self {
        Self {
            clock: FixedClock::new(rate, 8),
            rules: Vec::new(),
            context: RuleContext::new(0, seed),
            profiler: Profiler::with_capacity(256),
            ticks_executed: 0,
        }
    }

    /// 注册规则。同名规则会被拒绝（返回 `false`），避免剖析数据被污染。
    pub fn register(&mut self, rule: Box<dyn Rule>) -> bool {
        if self
            .rules
            .iter()
            .any(|existing| existing.name() == rule.name())
        {
            return false;
        }
        self.rules.push(rule);
        true
    }

    /// 已注册规则数。
    #[inline]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// 已执行的 Tick 总数。
    #[inline]
    pub fn ticks_executed(&self) -> u64 {
        self.ticks_executed
    }

    /// 时钟（供渲染取插值因子）。
    #[inline]
    pub fn clock(&self) -> &FixedClock {
        &self.clock
    }

    /// 规则间的写集冲突对（诊断用，正常应为空）。
    pub fn conflicts(&self) -> Vec<(&'static str, &'static str)> {
        let mut found = Vec::new();
        for (index, left) in self.rules.iter().enumerate() {
            for right in self.rules.iter().skip(index + 1) {
                if left.access().conflicts_with(&right.access()) {
                    found.push((left.name(), right.name()));
                }
            }
        }
        found
    }

    /// 按 `delta_s` 推进世界，返回执行的 Tick 数。
    pub fn advance(&mut self, delta_s: f64) -> u32 {
        let pending = self.clock.advance(delta_s);
        if pending == 0 {
            return 0;
        }
        // 时钟已经前进到最新的 tick，回溯出本批次第一个 tick 的编号，
        // 使规则看到的是**逐个递增**的 tick 号（而不是同一帧里的同一个值）。
        let first_tick = self.clock.tick() + 1 - u64::from(pending);
        for offset in 0..u64::from(pending) {
            self.run_tick(first_tick + offset);
        }
        pending
    }

    /// 手动执行一个 Tick（用于确定性回放与单元测试，不依赖真实时间）。
    pub fn step_once(&mut self) {
        let dt = self.clock.dt_s();
        self.clock.advance(dt);
        let tick = self.clock.tick();
        self.run_tick(tick);
    }

    /// 执行编号为 `tick` 的那一个 Tick。
    fn run_tick(&mut self, tick: u64) {
        self.context.tick = tick;
        self.context.dirty.clear();
        for rule in &mut self.rules {
            let period = (self.clock.rate().hz() / rule.cadence().hz()).max(1) as u64;
            if tick % period != 0 {
                continue;
            }
            // 每个规则单独计时：这是定位「哪条规则拖慢了世界」的唯一手段。
            let _span = self.profiler.span(rule.name());
            rule.apply(&mut self.context);
        }
        self.ticks_executed += 1;
    }

    /// 本 Tick 被标记为脏的 Chunk。
    #[inline]
    pub fn dirty_chunks(&self) -> &[ChunkCoord] {
        &self.context.dirty
    }

    /// 剖析器（按规则名聚合的耗时）。
    #[inline]
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }
}

impl core::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Simulation")
            .field("rate", &self.clock.rate())
            .field("tick", &self.clock.tick())
            .field("rules", &self.rules.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 记录调用次数的测试规则。
    struct CountingRule {
        name: &'static str,
        cadence: TickRate,
        calls: u32,
        writes: &'static [&'static str],
    }

    impl Rule for CountingRule {
        fn name(&self) -> &'static str {
            self.name
        }
        fn cadence(&self) -> TickRate {
            self.cadence
        }
        fn access(&self) -> RuleAccess {
            RuleAccess {
                reads: &[],
                writes: self.writes,
            }
        }
        fn apply(&mut self, ctx: &mut RuleContext) {
            self.calls += 1;
            ctx.mark_dirty(ChunkCoord::new(self.calls as i64, 0, 0));
        }
    }

    #[test]
    fn rules_run_at_their_declared_cadence() {
        let mut sim = Simulation::new(TickRate::Hz60, 1);
        assert!(sim.register(Box::new(CountingRule {
            name: "fast",
            cadence: TickRate::Hz60,
            calls: 0,
            writes: &["players"],
        })));
        assert!(sim.register(Box::new(CountingRule {
            name: "slow",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &["fluids"],
        })));

        // 0.05 秒 = 3 个 60 Hz Tick = 1 个 20 Hz Tick。
        assert_eq!(sim.advance(0.05), 3);
        assert_eq!(sim.ticks_executed(), 3);
        // 无法直接读规则的私有计数，故用脏列表间接验证：每 Tick 清理，
        // 20 Hz 规则只在 tick 0 运行，60 Hz 规则每 Tick 都运行。
        assert_eq!(sim.dirty_chunks().len(), 2);
    }

    #[test]
    fn duplicate_rule_names_are_refused() {
        let mut sim = Simulation::new(TickRate::Hz20, 1);
        assert!(sim.register(Box::new(CountingRule {
            name: "dup",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &[],
        })));
        assert!(!sim.register(Box::new(CountingRule {
            name: "dup",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &[],
        })));
        assert_eq!(sim.rule_count(), 1);
    }

    #[test]
    fn write_set_conflicts_are_detected() {
        let mut sim = Simulation::new(TickRate::Hz20, 1);
        sim.register(Box::new(CountingRule {
            name: "a",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &["blocks"],
        }));
        sim.register(Box::new(CountingRule {
            name: "b",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &["blocks"],
        }));
        sim.register(Box::new(CountingRule {
            name: "c",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &["entities"],
        }));
        assert_eq!(sim.conflicts(), vec![("a", "b")]);
    }

    #[test]
    fn profiler_records_per_rule_cost() {
        let mut sim = Simulation::new(TickRate::Hz20, 7);
        sim.register(Box::new(CountingRule {
            name: "profiled",
            cadence: TickRate::Hz20,
            calls: 0,
            writes: &[],
        }));
        sim.step_once();
        let stats = sim.profiler().stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "profiled");
        assert_eq!(stats[0].calls, 1);
    }

    #[test]
    fn rng_is_deterministic_for_the_same_seed() {
        let mut a = RuleContext::new(0, 12345);
        let mut b = RuleContext::new(0, 12345);
        let draws_a: Vec<u64> = (0..8).map(|_| a.next_random()).collect();
        let draws_b: Vec<u64> = (0..8).map(|_| b.next_random()).collect();
        assert_eq!(draws_a, draws_b);
        assert!(draws_a.windows(2).all(|w| w[0] != w[1]), "连续抽样不应重复");
    }

    #[test]
    fn dirty_marking_is_idempotent() {
        let mut ctx = RuleContext::new(0, 0);
        let coord = ChunkCoord::new(1, 2, 3);
        ctx.mark_dirty(coord);
        ctx.mark_dirty(coord);
        assert_eq!(ctx.dirty.len(), 1);
    }
}
