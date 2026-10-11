//! # world-contracts —— 世界层只读契约
//!
//! 本 crate 只包含 **trait 与值类型**，不含任何实现。作用是切断依赖环：
//! `geology`、`storage`、`streaming`、`render` 之间只通过这里的接口对话，
//! 因此任何一侧都可以被替换或 mock。
//!
//! * [`StratumSampler`]：地质侧向外提供「沿某个方向的地层剖面」。
//! * [`WorldView`]：体素世界的只读视图，网格生成、物理拾取、权威校验共用。
//! * [`ChunkCoord`]：32³ Chunk 的全局坐标格子。
//!
//! 契约约束：
//!
//! 1. 所有查询**必须无副作用**且可跨线程调用（`Send + Sync`）；
//! 2. 越界查询一律返回 [`BlockId::AIR`]，禁止 panic——边界是流式加载的常态而非异常；
//! 3. 单位一律为米，坐标为全局 `i64` 体素格子。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod sampler;
pub mod view;

pub use sampler::{StrataProfile, Stratum, StratumSampler, MAX_STRATA};
pub use view::{ChunkCoord, EmptyWorld, WorldView, CHUNK_EDGE, CHUNK_VOLUME};

pub use block_schema::{BlockId, BlockRegistry};
pub use planet_coords::{CubeFace, SphereGeometry, SurfacePoint, VoxelCoord};
