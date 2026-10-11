//! # geology —— 行星地质仿真
//!
//! 目标：用**极小的状态**（一个 64 位种子 + 数十个板块站点）生成一颗始终自洽、
//! 可无限细分、无需存储的行星。任何方向的地层都能在 O(1) 时间内重新算出。
//!
//! 生成管线：
//!
//! ```text
//! seed ──► [板块 Voronoi 撒点] ──► [边界造山 / 洋中脊] ──► [分形细节]
//!                                                              │
//!                                             ┌────────────────┘
//!                                             ▼
//!                                 [连续地层剖面 StrataProfile]
//!                                             │
//!                          ┌──────────────────┴──────────────────┐
//!                          ▼                                     ▼
//!            [宏观 LOD / streaming]                [近景体素填充 world-meshing]
//! ```
//!
//! * [`noise`]：确定性哈希噪声（无第三方依赖，跨平台逐位一致）。
//! * [`plates`]：球面 Voronoi 板块场与漂移。
//! * [`strata`]：把板块场转成 `world-contracts::StratumSampler` 的连续剖面。
//!
//! 尚未实现（路线图）：水文 DAG 侵蚀与沉积物输运、岩性随深度变质、火山口与陨石坑叠加。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod noise;
pub mod plates;
pub mod strata;

pub use plates::{PlateField, PlateId, PlateSite};
pub use strata::{GeologyBlocks, GeologyConfig, GeologyGenerator};
