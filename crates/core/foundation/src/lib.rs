//! # foundation —— 底层基底
//!
//! 与游戏语义完全无关的通用基础设施。本 crate 位于依赖图的最底层：
//! **它不依赖任何其它内部 crate，并且只允许依赖 `std`**（保留未来 `no_std` 迁移的可能）。
//!
//! 模块职责：
//!
//! * [`handle`]：世代索引 [`Handle`]，杜绝悬挂引用与 ABA 问题。
//! * [`arena`]：由世代句柄寻址的对象池 [`Arena`]。
//! * [`profiler`]：微秒级作用域采样器 [`Profiler`]，为 20Hz/60Hz 帧预算兜底。
//!
//! 设计红线：本 crate 不做 I/O、不做分配策略假设、不引入第三方依赖。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod arena;
pub mod handle;
pub mod profiler;

pub use arena::Arena;
pub use handle::Handle;
pub use profiler::{Profiler, Sample, ScopeStat, SpanGuard};
