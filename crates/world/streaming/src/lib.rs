//! # streaming —— 流式加载与 LOD 调度
//!
//! 把「相机在哪」翻译成「本帧该生成什么、该换出什么」，并且**永不阻塞帧**。
//!
//! 两条独立流水：
//!
//! | 区域 | 表示 | 调度器 |
//! |------|------|--------|
//! | 近景 `0 ~ 256 m` | 32³ 体素 Chunk（可编辑、可物理碰撞） | [`ChunkPlanner`] |
//! | 中远景 | 四叉树 / Clipmap patch（不可编辑） | [`QuadtreeSelector`] |
//!
//! * [`generate_chunk`]：按地层剖面填充一个 Chunk 的体素（可并行、可复现）。
//! * [`TaskQueue`]：惰性实例化任务队列（`crossbeam-channel`），本帧只取预算内的量。
//!
//! 性能约定：单帧从队列取走的任务数受 [`StreamingConfig::max_tasks_per_frame`] 约束，
//! 保证「相机高速移动」不会演变成一次性生成上百个 Chunk 的卡顿。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod quadtree;

use block_schema::BlockId;
use crossbeam_channel::{unbounded, Receiver, Sender};
use geology::GeologyGenerator;
use planet_coords::{voxel_edge_m, VoxelCoord};
use rayon::prelude::*;
use world_contracts::{ChunkCoord, StratumSampler, CHUNK_EDGE, CHUNK_VOLUME};

pub use quadtree::{QuadNode, QuadtreeConfig, QuadtreeSelector};

/// 流式加载配置。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamingConfig {
    /// 近景在水平方向覆盖的 Chunk 半径。
    pub horizontal_radius_chunks: i32,
    /// 近景在垂直方向覆盖的 Chunk 半径（地表上下）。
    pub vertical_radius_chunks: i32,
    /// 单帧最多派发的任务数。
    pub max_tasks_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            horizontal_radius_chunks: 6,
            vertical_radius_chunks: 2,
            max_tasks_per_frame: 24,
        }
    }
}

/// 按程序化地层填充一个 Chunk 的全部体素。
///
/// 对每个体素：求出其中心方向与半径，与地表高程比较——低于地表则为实体，
/// 并按「地表以下深度」选出对应地层的方块。因此 Chunk 内容**完全由种子决定**，
/// 不需要任何存储。
///
/// 性能说明：朴素实现是每体素一次 `elevation_m`（约 24 次站点比较 + 5 倍频噪声）。
/// 32³ 约需数毫秒，适合烘焙与流式后台；运行时若要更快，应按「列」缓存高程
/// （同一 `(x, z)` 的所有 `y` 共享方向附近的高程），这是路线图上的第一项优化。
pub fn generate_chunk(geo: &GeologyGenerator, coord: ChunkCoord, radius_m: f64) -> Vec<BlockId> {
    let origin = coord.min_voxel();
    let mut voxels = vec![BlockId::AIR; CHUNK_VOLUME];
    for z in 0..CHUNK_EDGE {
        for y in 0..CHUNK_EDGE {
            for x in 0..CHUNK_EDGE {
                let voxel = VoxelCoord::new(
                    origin.x + i64::from(x),
                    origin.y + i64::from(y),
                    origin.z + i64::from(z),
                );
                let center = voxel.center_m();
                let distance = center.length();
                if distance <= f64::MIN_POSITIVE {
                    continue;
                }
                let dir = center / distance;
                let elevation = geo.surface_height_m(dir);
                let radius = radius_m + elevation;
                if distance >= radius {
                    continue;
                }
                let depth = radius - distance;
                let index = (x + y * CHUNK_EDGE + z * CHUNK_EDGE * CHUNK_EDGE) as usize;
                voxels[index] = geo.block_at_depth(dir, depth);
            }
        }
    }
    voxels
}

/// 并行填充一批 Chunk（顺序与输入一致）。
///
/// 这是 `world-tool bake` 与流式加载共用的入口：地质采样是纯函数且无共享状态，
/// 因此可以安全地铺满所有核心。
pub fn generate_chunks(
    geo: &GeologyGenerator,
    coords: &[ChunkCoord],
    radius_m: f64,
) -> Vec<(ChunkCoord, Vec<BlockId>)> {
    coords
        .par_iter()
        .map(|coord| (*coord, generate_chunk(geo, *coord, radius_m)))
        .collect()
}

/// 近景 Chunk 规划：找出半径内**尚未加载**的 Chunk，按距离由近到远排序。
pub fn plan_near_field(
    center: ChunkCoord,
    config: &StreamingConfig,
    is_loaded: impl Fn(ChunkCoord) -> bool,
) -> Vec<ChunkCoord> {
    let mut planned = Vec::new();
    for dz in -config.vertical_radius_chunks..=config.vertical_radius_chunks {
        for dy in -config.vertical_radius_chunks..=config.vertical_radius_chunks {
            for dx in -config.horizontal_radius_chunks..=config.horizontal_radius_chunks {
                // 圆柱形而不是立方体：垂直方向的视野需求远小于水平方向。
                if dx * dx + dz * dz > config.horizontal_radius_chunks.pow(2) {
                    continue;
                }
                let coord = ChunkCoord::new(
                    center.x + i64::from(dx),
                    center.y + i64::from(dy),
                    center.z + i64::from(dz),
                );
                if is_loaded(coord) {
                    continue;
                }
                planned.push(coord);
            }
        }
    }
    planned.sort_by_key(|coord| {
        let dx = coord.x - center.x;
        let dy = coord.y - center.y;
        let dz = coord.z - center.z;
        dx * dx + dy * dy + dz * dz
    });
    planned
}

/// Chunk 体素的球心半径对应的默认体素边长（米）。
pub fn default_voxel_edge_m() -> f64 {
    voxel_edge_m()
}

/// 任务类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    /// 生成体素（`lod` 决定是否降采样）。
    Generate {
        /// LOD 层级，0 = 全分辨率。
        lod: u32,
    },
    /// 生成网格。
    Mesh,
    /// 落盘。
    Persist,
    /// 换出内存。
    Evict,
}

/// 一个流式任务。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamTask {
    /// 目标 Chunk。
    pub coord: ChunkCoord,
    /// 任务类型。
    pub kind: TaskKind,
    /// 优先级（越小越先执行，通常取相机距离）。
    pub priority: f64,
}

impl StreamTask {
    /// 构造。
    pub fn new(coord: ChunkCoord, kind: TaskKind, priority: f64) -> Self {
        Self {
            coord,
            kind,
            priority,
        }
    }
}

/// 统计。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingStats {
    /// 已提交任务数。
    pub submitted: u64,
    /// 已派出任务数。
    pub dispatched: u64,
    /// 因队列已满被丢弃的任务数。
    pub dropped: u64,
}

/// 惰性实例化任务队列。
///
/// 生产侧（选址器）与消费侧（网格生成工作线程）通过无锁 MPMC 通道解耦：
/// 生产侧永不阻塞，消费侧按自己的节奏批量取任务。
pub struct TaskQueue {
    sender: Sender<StreamTask>,
    receiver: Receiver<StreamTask>,
    capacity: usize,
    stats: StreamingStats,
}

impl TaskQueue {
    /// 以容量构造（队列满时提交会被拒绝而非阻塞）。
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            sender,
            receiver,
            capacity: capacity.max(1),
            stats: StreamingStats::default(),
        }
    }

    /// 提交任务；队列已满时返回 `false`。
    pub fn submit(&mut self, task: StreamTask) -> bool {
        if self.sender.len() >= self.capacity {
            self.stats.dropped += 1;
            return false;
        }
        if self.sender.send(task).is_err() {
            self.stats.dropped += 1;
            return false;
        }
        self.stats.submitted += 1;
        true
    }

    /// 取出至多 `limit` 个任务。
    pub fn pop_batch(&mut self, limit: usize) -> Vec<StreamTask> {
        let mut batch = Vec::with_capacity(limit.min(64));
        for _ in 0..limit {
            match self.receiver.try_recv() {
                Ok(task) => batch.push(task),
                Err(_) => break,
            }
        }
        self.stats.dispatched += batch.len() as u64;
        batch
    }

    /// 当前排队数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.sender.len()
    }

    /// 队列是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sender.len() == 0
    }

    /// 容量。
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 统计快照。
    #[inline]
    pub fn stats(&self) -> StreamingStats {
        self.stats
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl core::fmt::Debug for TaskQueue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TaskQueue")
            .field("queued", &self.len())
            .field("capacity", &self.capacity)
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geology::{GeologyConfig, GeologyGenerator};

    const EARTH_R: f64 = 6_371_000.0;

    fn geology() -> GeologyGenerator {
        GeologyGenerator::new(GeologyConfig {
            seed: 2026,
            plate_count: 16,
            ..GeologyConfig::default()
        })
    }

    #[test]
    fn generated_chunk_is_deterministic_and_mixed() {
        let geo = geology();
        let dir = glam::DVec3::X;
        let surface_voxel = (EARTH_R + geo.surface_height_m(dir)).round() as i64;
        // 取正好跨越地表的那个 Chunk。
        let coord = ChunkCoord::from_voxel(VoxelCoord::new(surface_voxel, 0, 0));

        let first = generate_chunk(&geo, coord, EARTH_R);
        let second = generate_chunk(&geo, coord, EARTH_R);
        assert_eq!(first, second, "同一坐标必须生成完全相同的体素");

        let solid = first.iter().filter(|block| !block.is_air()).count();
        assert!(solid > 0, "跨越地表的 Chunk 必须有实体");
        assert!(solid < CHUNK_VOLUME, "Chunk 顶部必须是空气");
    }

    #[test]
    fn batch_generation_preserves_order() {
        let geo = geology();
        let coords = vec![
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(1, 0, 0),
            ChunkCoord::new(2, 0, 0),
        ];
        let generated = generate_chunks(&geo, &coords, EARTH_R);
        assert_eq!(generated.len(), 3);
        assert_eq!(generated[0].0, coords[0]);
        assert_eq!(generated[2].0, coords[2]);
        assert!(generated
            .iter()
            .all(|(_, voxels)| voxels.len() == CHUNK_VOLUME));
    }

    #[test]
    fn planner_skips_loaded_chunks_and_sorts_by_distance() {
        let config = StreamingConfig {
            horizontal_radius_chunks: 3,
            vertical_radius_chunks: 1,
            max_tasks_per_frame: 8,
        };
        let center = ChunkCoord::new(10, 0, 10);
        let loaded_center = center;
        let planned = plan_near_field(center, &config, |coord| coord == loaded_center);
        assert!(!planned.contains(&loaded_center), "起点自身不应重复加载");
        assert!(!planned.is_empty());

        let distance = |coord: ChunkCoord| {
            let dx = coord.x - center.x;
            let dy = coord.y - center.y;
            let dz = coord.z - center.z;
            dx * dx + dy * dy + dz * dz
        };
        assert!(
            planned.windows(2).all(|w| distance(w[0]) <= distance(w[1])),
            "必须按距离由近到远"
        );
        // 圆柱形裁剪：最远的水平距离不超过半径。
        assert!(planned.iter().all(|coord| {
            let dx = coord.x - center.x;
            let dz = coord.z - center.z;
            dx * dx + dz * dz <= 9
        }));
    }

    #[test]
    fn task_queue_batches_and_respects_capacity() {
        let mut queue = TaskQueue::new(2);
        assert_eq!(queue.capacity(), 2);
        assert!(queue.submit(StreamTask::new(
            ChunkCoord::new(0, 0, 0),
            TaskKind::Mesh,
            1.0
        )));
        assert!(queue.submit(StreamTask::new(
            ChunkCoord::new(1, 0, 0),
            TaskKind::Mesh,
            2.0
        )));
        assert!(!queue.submit(StreamTask::new(
            ChunkCoord::new(2, 0, 0),
            TaskKind::Mesh,
            3.0
        )));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.stats().dropped, 1);

        let batch = queue.pop_batch(1);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].coord, ChunkCoord::new(0, 0, 0), "FIFO 顺序");
        assert_eq!(queue.pop_batch(8).len(), 1);
        assert!(queue.is_empty());
        assert_eq!(queue.stats().submitted, 2);
        assert_eq!(queue.stats().dispatched, 2);
    }

    #[test]
    fn voxel_edge_default_matches_planet_coords() {
        assert!((default_voxel_edge_m() - 1.0).abs() < f64::EPSILON);
    }
}
