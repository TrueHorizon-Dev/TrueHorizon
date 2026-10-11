//! 固定步长时钟。
//!
//! 仿真必须以**固定 Tick** 推进（20 Hz 规则世界 + 60 Hz 交互物理），渲染则以可变
//! 帧率采样并做插值。二者之间只通过本模块握手：
//!
//! ```text
//! 帧 dt ──► [accumulator] ──► n 次固定 Tick ──► 剩余 alpha ∈ [0,1) 供渲染插值
//! ```
//!
//! 「死亡螺旋」防护：单帧最多补跑 `max_catch_up` 个 Tick，超出的时间被**丢弃**而不是
//! 排队——宁可让世界慢一拍，也不能让卡顿滚雪球。

/// 仿真频率。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TickRate {
    /// 20 Hz：世界规则（流体、生长、侵蚀）与网络快照。
    Hz20 = 20,
    /// 60 Hz：玩家与实体的交互物理。
    Hz60 = 60,
}

impl TickRate {
    /// 频率（Hz）。
    #[inline]
    pub const fn hz(self) -> u32 {
        self as u32
    }

    /// 单 Tick 时长（秒）。
    #[inline]
    pub fn dt_s(self) -> f64 {
        1.0 / self as u32 as f64
    }

    /// 单 Tick 时长（毫秒）。
    #[inline]
    pub fn dt_ms(self) -> f64 {
        1000.0 / self as u32 as f64
    }
}

/// 固定步长时钟。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedClock {
    rate: TickRate,
    accumulator_s: f64,
    tick: u64,
    max_catch_up: u32,
}

impl FixedClock {
    /// 构造。`max_catch_up` 会被夹到 `[1, 240]`。
    pub fn new(rate: TickRate, max_catch_up: u32) -> Self {
        Self {
            rate,
            accumulator_s: 0.0,
            tick: 0,
            max_catch_up: max_catch_up.clamp(1, 240),
        }
    }

    /// 当前频率。
    #[inline]
    pub fn rate(&self) -> TickRate {
        self.rate
    }

    /// 已推进的 Tick 数。
    #[inline]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// 单 Tick 时长（秒）。
    #[inline]
    pub fn dt_s(&self) -> f64 {
        self.rate.dt_s()
    }

    /// 渲染插值因子 `alpha = accumulator / dt`，恒在 `[0, 1)`。
    pub fn alpha(&self) -> f64 {
        (self.accumulator_s / self.dt_s()).clamp(0.0, 1.0)
    }

    /// 推进 `delta_s` 秒，返回本帧应执行的 Tick 数。
    ///
    /// 非有限或非正的 `delta_s` 一律视为 0（时间源异常不应让世界倒退或卡死）。
    pub fn advance(&mut self, delta_s: f64) -> u32 {
        if !delta_s.is_finite() || delta_s <= 0.0 {
            return 0;
        }
        let dt = self.dt_s();
        let budget = dt * f64::from(self.max_catch_up);
        self.accumulator_s += delta_s.min(budget);

        let mut ticks = 0;
        while self.accumulator_s + f64::EPSILON >= dt && ticks < self.max_catch_up {
            self.accumulator_s -= dt;
            self.tick += 1;
            ticks += 1;
        }
        if ticks == self.max_catch_up {
            // 补跑额度用尽：**丢弃全部积压**。哪怕只留一个 Tick 的欠账，
            // 下一帧也会继续补跑，卡顿就会滚成雪球（death spiral）。
            self.accumulator_s = 0.0;
        }
        ticks
    }

    /// 重置到 Tick 0。
    pub fn reset(&mut self) {
        self.accumulator_s = 0.0;
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_rates_map_to_expected_periods() {
        assert_eq!(TickRate::Hz20.hz(), 20);
        assert!((TickRate::Hz20.dt_s() - 0.05).abs() < 1e-12);
        assert!((TickRate::Hz60.dt_ms() - 16.666_666).abs() < 1e-4);
    }

    #[test]
    fn advance_accumulates_partial_frames() {
        let mut clock = FixedClock::new(TickRate::Hz60, 8);
        // 三次 1/180 秒 = 一次 60 Hz Tick。
        assert_eq!(clock.advance(1.0 / 180.0), 0);
        assert_eq!(clock.advance(1.0 / 180.0), 0);
        assert_eq!(clock.advance(1.0 / 180.0), 1);
        assert_eq!(clock.tick(), 1);
        assert!(clock.alpha() < 1.0);
    }

    #[test]
    fn twenty_hz_advances_once_per_fifty_milliseconds() {
        let mut clock = FixedClock::new(TickRate::Hz20, 8);
        assert_eq!(clock.advance(0.049), 0);
        assert_eq!(clock.advance(0.001), 1);
        assert_eq!(clock.advance(0.150), 3);
        assert_eq!(clock.tick(), 4);
    }

    #[test]
    fn catch_up_is_capped_to_avoid_death_spiral() {
        let mut clock = FixedClock::new(TickRate::Hz20, 3);
        // 模拟 5 秒的卡顿：只允许补跑 3 个 Tick，其余时间被丢弃。
        assert_eq!(clock.advance(5.0), 3);
        assert_eq!(clock.tick(), 3);
        assert_eq!(clock.alpha(), 0.0, "积压必须被完全丢弃");
        // 随后立刻回到正常节奏：不再补跑，也不再欠账。
        assert_eq!(clock.advance(0.04), 0, "不应残留积欠的 Tick");
        assert_eq!(clock.advance(0.05), 1, "恢复正常节奏");
    }

    #[test]
    fn invalid_time_sources_are_ignored() {
        let mut clock = FixedClock::new(TickRate::Hz20, 4);
        assert_eq!(clock.advance(0.0), 0);
        assert_eq!(clock.advance(-1.0), 0);
        assert_eq!(clock.advance(f64::NAN), 0);
        assert_eq!(clock.advance(f64::INFINITY), 0);
        assert_eq!(clock.tick(), 0);
        clock.reset();
        assert_eq!(clock.alpha(), 0.0);
    }
}
