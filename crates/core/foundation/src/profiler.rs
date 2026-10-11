//! 微秒级作用域采样器。

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 一次作用域采样的原始记录。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// 作用域名字（要求 `'static`，避免采样热路径上分配字符串）。
    pub name: &'static str,
    /// 该次耗时的微秒数。
    pub micros: u64,
}

/// 按名字聚合后的统计结果。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScopeStat {
    /// 作用域名字。
    pub name: &'static str,
    /// 调用次数。
    pub calls: u64,
    /// 累计微秒数。
    pub total_micros: u64,
    /// 单次最大微秒数（定位卡顿帧）。
    pub max_micros: u64,
}

impl ScopeStat {
    /// 平均微秒数。
    pub fn mean_micros(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_micros as f64 / self.calls as f64
        }
    }
}

/// 微秒级作用域采样器。
///
/// 用法：
///
/// ```
/// use foundation::Profiler;
///
/// let mut profiler = Profiler::new();
/// {
///     let _span = profiler.span("geology.strata");
///     // ... 被测代码
/// }
/// assert_eq!(profiler.stats()[0].name, "geology.strata");
/// ```
///
/// 约束：`SpanGuard` 独占借用 `Profiler`，因此**同一实例上不能嵌套 span**。
/// 需要嵌套或跨线程时，各线程/各阶段持有独立实例，结束后用 [`Profiler::merge`] 汇总。
pub struct Profiler {
    samples: Vec<Sample>,
}

impl Profiler {
    /// 创建空采样器。
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// 预分配 `capacity` 条采样记录。
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }

    /// 开启一个计时作用域；guard 析构时写入采样。
    ///
    /// ```
    /// # use foundation::Profiler;
    /// let mut profiler = Profiler::new();
    /// let guard = profiler.span("frame.simulate");
    /// drop(guard);
    /// assert_eq!(profiler.len(), 1);
    /// ```
    pub fn span(&mut self, name: &'static str) -> SpanGuard<'_> {
        SpanGuard {
            profiler: Some(self),
            name,
            start: Instant::now(),
        }
    }

    /// 直接写入一条手工测量结果（例如来自 GPU timestamp query）。
    pub fn record(&mut self, name: &'static str, elapsed: Duration) {
        self.samples.push(Sample {
            name,
            micros: elapsed.as_micros().min(u64::MAX as u128) as u64,
        });
    }

    /// 采样条数。
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// 是否没有任何采样。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 全部采样累计微秒数。
    pub fn total_micros(&self) -> u64 {
        self.samples.iter().map(|s| s.micros).sum()
    }

    /// 原始采样切片。
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// 按累计耗时降序聚合。
    pub fn stats(&self) -> Vec<ScopeStat> {
        let mut table: HashMap<&'static str, ScopeStat> = HashMap::new();
        for sample in &self.samples {
            let entry = table.entry(sample.name).or_insert(ScopeStat {
                name: sample.name,
                calls: 0,
                total_micros: 0,
                max_micros: 0,
            });
            entry.calls += 1;
            entry.total_micros += sample.micros;
            entry.max_micros = entry.max_micros.max(sample.micros);
        }
        let mut stats: Vec<ScopeStat> = table.into_values().collect();
        stats.sort_by(|a, b| {
            b.total_micros
                .cmp(&a.total_micros)
                .then_with(|| a.name.cmp(b.name))
        });
        stats
    }

    /// 合并另一个采样器（用于把工作线程的采样并入主线程）。
    pub fn merge(&mut self, other: Profiler) {
        self.samples.extend(other.samples);
    }

    /// 清空采样但保留已分配容量。
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for Profiler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Profiler")
            .field("samples", &self.samples.len())
            .field("total_micros", &self.total_micros())
            .finish()
    }
}

/// [`Profiler::span`] 返回的 RAII 计时器。
///
/// 内部用 `Option<&mut Profiler>` 持有借用，使 [`SpanGuard::finish`] 能主动结束计时，
/// 同时保证 `Drop` 不会重复记录（`take` 之后即为 `None`）。
pub struct SpanGuard<'a> {
    profiler: Option<&'a mut Profiler>,
    name: &'static str,
    start: Instant,
}

impl SpanGuard<'_> {
    /// 主动结束计时并返回本次微秒数（等价于提前 drop，但能拿到数值）。
    pub fn finish(mut self) -> u64 {
        let micros = self.start.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if let Some(profiler) = self.profiler.take() {
            profiler.samples.push(Sample {
                name: self.name,
                micros,
            });
        }
        micros
    }
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        if let Some(profiler) = self.profiler.take() {
            profiler.samples.push(Sample {
                name: self.name,
                micros: self.start.elapsed().as_micros().min(u64::MAX as u128) as u64,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_records_on_drop() {
        let mut profiler = Profiler::new();
        {
            let _span = profiler.span("test.scope");
        }
        assert_eq!(profiler.len(), 1);
        assert_eq!(profiler.stats()[0].name, "test.scope");
    }

    #[test]
    fn stats_aggregate_and_sort() {
        let mut profiler = Profiler::new();
        profiler.record("b", Duration::from_micros(10));
        profiler.record("a", Duration::from_micros(50));
        profiler.record("a", Duration::from_micros(70));
        let stats = profiler.stats();
        assert_eq!(stats[0].name, "a");
        assert_eq!(stats[0].calls, 2);
        assert_eq!(stats[0].total_micros, 120);
        assert_eq!(stats[0].max_micros, 70);
        assert_eq!(stats[1].name, "b");
        assert_eq!(profiler.total_micros(), 130);
    }

    #[test]
    fn merge_combines_worker_samples() {
        let mut main = Profiler::new();
        main.record("main", Duration::from_micros(1));
        let mut worker = Profiler::new();
        worker.record("worker", Duration::from_micros(2));
        main.merge(worker);
        assert_eq!(main.len(), 2);
        assert_eq!(main.total_micros(), 3);
    }
}
