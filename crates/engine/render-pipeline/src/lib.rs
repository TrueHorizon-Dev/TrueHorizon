//! # render-pipeline —— GPU-Driven 渲染管线
//!
//! 消费 `render-contracts::SceneSnapshot`，产出帧。分层：
//!
//! ```text
//! SceneSnapshot ──► [实例缓冲上传] ──► [球面 Clipmap 选择] ──► [GPU-Driven 剔除]
//!                                                                     │
//!                        ┌──────────── Pass 顺序 ────────────────────┘
//!                        ▼
//!   DepthPrepass → OpaqueMeshlet → CutoutMeshlet → Translucent → PostFx
//! ```
//!
//! 骨架现状：**帧的状态推进、快照校验、Clipmap 选择、统计与射线反查已经可用**
//! （因此可以在无 GPU 的 CI 里跑通），真实命令录制与 mesh shader 派发入口保留在
//! [`SpherePipeline::record`] 中作为下一步。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod clipmap;

pub use clipmap::{PatchKey, SphereClipmap, SphereClipmapConfig};
use glam::DVec3;
use planet_coords::FloatingOrigin;
use render_contracts::{
    CameraSnapshot, DrawBucket, InstanceRaw, RenderContract, SceneError, SceneSnapshot,
};

/// 一帧内的渲染 Pass 顺序。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum PassKind {
    /// 深度预通道（顺带写入 Hi-Z，供 GPU 剔除使用）。
    DepthPrepass = 0,
    /// 不透明 meshlet。
    OpaqueMeshlet = 1,
    /// 镂空 meshlet。
    CutoutMeshlet = 2,
    /// 半透明（含水面）。
    Translucent = 3,
    /// 后处理（大气散射、色调映射、抗锯齿）。
    PostFx = 4,
}

impl PassKind {
    /// 全部 Pass，顺序即提交顺序。
    pub const ALL: [PassKind; 5] = [
        PassKind::DepthPrepass,
        PassKind::OpaqueMeshlet,
        PassKind::CutoutMeshlet,
        PassKind::Translucent,
        PassKind::PostFx,
    ];

    /// 对应的绘制桶（后处理没有绘制桶）。
    pub const fn bucket(self) -> Option<DrawBucket> {
        match self {
            Self::OpaqueMeshlet => Some(DrawBucket::Opaque),
            Self::CutoutMeshlet => Some(DrawBucket::Cutout),
            Self::Translucent => Some(DrawBucket::Translucent),
            Self::DepthPrepass | Self::PostFx => None,
        }
    }
}

/// 管线配置。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipelineConfig {
    /// 行星半径（米）。
    pub planet_radius_m: f64,
    /// Clipmap 配置。
    pub clipmap: SphereClipmapConfig,
    /// 单帧绘制命令上限，超出即视为上层逻辑异常（防雪崩）。
    pub max_draw_calls: u32,
    /// 是否走 mesh shader 路径；`false` 时退化为顶点拉取路径。
    pub use_mesh_shader: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            planet_radius_m: 6_371_000.0,
            clipmap: SphereClipmapConfig::default(),
            max_draw_calls: 65_536,
            use_mesh_shader: true,
        }
    }
}

/// 单帧统计，供 HUD、`world-tool diagnose` 与性能回归使用。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// 帧号。
    pub frame_index: u64,
    /// 绘制命令数。
    pub draw_calls: u32,
    /// 实例数。
    pub instance_count: u32,
    /// 本帧选中的 clipmap patch 数。
    pub clipmap_patches: u32,
    /// 本帧上传字节数。
    pub uploaded_bytes: u64,
    /// 浮动原点搬移累计次数。
    pub origin_shifts: u64,
}

/// 一个待录制的 Pass。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassRecord {
    /// Pass 类型。
    pub kind: PassKind,
    /// 该 Pass 的绘制命令数。
    pub draw_calls: u32,
}

/// 球面渲染管线。
///
/// 实现了 [`RenderContract`]，因此可以被 `apps/client` 直接当作渲染器使用，
/// 也可以被 `tests/integration` 用作「无 GPU 的假渲染器」跑端到端流程。
#[derive(Debug)]
pub struct SpherePipeline {
    config: PipelineConfig,
    clipmap: SphereClipmap,
    origin: FloatingOrigin,
    camera: CameraSnapshot,
    stats: FrameStats,
    passes: Vec<PassRecord>,
    instance_snapshot: Vec<InstanceRaw>,
}

impl SpherePipeline {
    /// 以配置构造。
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            clipmap: SphereClipmap::new(config.clipmap, config.planet_radius_m),
            config,
            origin: FloatingOrigin::default(),
            camera: CameraSnapshot::default(),
            stats: FrameStats::default(),
            passes: Vec::new(),
            instance_snapshot: Vec::new(),
        }
    }

    /// 配置。
    #[inline]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Clipmap 选择器。
    #[inline]
    pub fn clipmap(&self) -> &SphereClipmap {
        &self.clipmap
    }

    /// 最近一帧统计。
    #[inline]
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// 上一帧各 Pass 的命令分布。
    #[inline]
    pub fn passes(&self) -> &[PassRecord] {
        &self.passes
    }

    /// 当前浮动原点。
    #[inline]
    pub fn origin(&self) -> FloatingOrigin {
        self.origin
    }

    /// 当前相机快照。
    #[inline]
    pub fn camera(&self) -> CameraSnapshot {
        self.camera
    }

    /// 本帧需要绘制的 clipmap patch（按层升序）。
    pub fn select_patches(&self) -> Vec<PatchKey> {
        let horizon = self.clipmap.geometry().horizon_arc_m(self.camera.height_m);
        let view_dir = self.camera.orientation * DVec3::Z;
        self.clipmap.select(view_dir, horizon)
    }

    /// 录制命令缓冲。
    ///
    /// TODO(rhi): 真实实现需要 —— 实例缓冲环形区上传、mesh shader 派发参数写入、
    /// 时间戳查询与 Hi-Z 回读。骨架阶段只做状态推进与统计。
    pub fn record(&mut self, snapshot: &SceneSnapshot) -> Result<(), SceneError> {
        snapshot.validate()?;
        let draw_calls = snapshot.draw_call_count() as u32;
        if draw_calls > self.config.max_draw_calls {
            return Err(SceneError::InvalidMesh {
                bucket: DrawBucket::Opaque,
                index: draw_calls as usize,
            });
        }

        self.camera = snapshot.camera;
        self.origin = snapshot.origin;
        self.instance_snapshot.clear();
        self.instance_snapshot
            .extend_from_slice(&snapshot.instances);

        self.passes.clear();
        for kind in PassKind::ALL {
            let count = kind
                .bucket()
                .map(|bucket| snapshot.bucket(bucket).len() as u32)
                .unwrap_or(0);
            self.passes.push(PassRecord {
                kind,
                draw_calls: count,
            });
        }

        let patches = self.select_patches();
        self.stats = FrameStats {
            frame_index: snapshot.frame_index,
            draw_calls,
            instance_count: snapshot.instances.len() as u32,
            clipmap_patches: patches.len() as u32,
            uploaded_bytes: (snapshot.instances.len() * core::mem::size_of::<InstanceRaw>()
                + snapshot.draw_call_count() * core::mem::size_of::<render_contracts::DrawCall>())
                as u64,
            origin_shifts: self.stats.origin_shifts,
        };
        Ok(())
    }
}

impl RenderContract for SpherePipeline {
    fn submit(&mut self, snapshot: &SceneSnapshot) -> Result<(), SceneError> {
        self.record(snapshot)
    }

    fn on_origin_shifted(&mut self, origin: FloatingOrigin) {
        self.origin = origin;
        self.stats.origin_shifts += 1;
        // 原点搬移后，上一帧的实例缓冲内容已失效，必须整体重建。
        self.instance_snapshot.clear();
    }

    fn screen_to_ray(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Option<(DVec3, DVec3)> {
        if !aspect.is_finite() || aspect <= 0.0 {
            return None;
        }
        let tan_y = (self.camera.fov_y_rad * 0.5).tan();
        let tan_x = tan_y * aspect as f64;
        // Vulkan 的 NDC 为 Y 向下，因此世界系中的「上」对应 -ndc_y。
        let direction = (self.camera.orientation
            * DVec3::new(tan_x * ndc_x as f64, -tan_y * ndc_y as f64, 1.0))
        .normalize();
        let origin = self
            .origin
            .to_render(self.camera.position_world_m)
            .as_dvec3();
        Some((origin, direction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use render_contracts::{DrawCall, Material, MeshHandle};

    fn snapshot_with_one_instance() -> SceneSnapshot {
        let mut snapshot = SceneSnapshot::default();
        snapshot.instances.push(InstanceRaw::from_world(
            FloatingOrigin::default(),
            DVec3::new(10.0, 0.0, 0.0),
            0,
            Material::Rock,
        ));
        snapshot
            .opaque
            .push(DrawCall::new(MeshHandle(1), Material::Rock, 0, 1));
        snapshot
    }

    #[test]
    fn pass_order_matches_documented_pipeline() {
        let order: Vec<u8> = PassKind::ALL.iter().map(|p| *p as u8).collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            PassKind::Translucent.bucket(),
            Some(DrawBucket::Translucent)
        );
        assert_eq!(PassKind::PostFx.bucket(), None);
    }

    #[test]
    fn submit_advances_state_and_stats() {
        let mut pipeline = SpherePipeline::new(PipelineConfig::default());
        pipeline.submit(&snapshot_with_one_instance()).unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.draw_calls, 1);
        assert_eq!(stats.instance_count, 1);
        assert!(stats.uploaded_bytes >= 48 + 24);
        assert_eq!(pipeline.passes()[0].kind, PassKind::DepthPrepass);
        assert_eq!(pipeline.passes()[1].draw_calls, 1);
    }

    #[test]
    fn origin_shift_invalidates_cached_instances() {
        let mut pipeline = SpherePipeline::new(PipelineConfig::default());
        pipeline.submit(&snapshot_with_one_instance()).unwrap();
        pipeline.on_origin_shifted(FloatingOrigin::new(DVec3::new(1.0e6, 0.0, 0.0)));
        assert_eq!(pipeline.stats().origin_shifts, 1);
        assert!(pipeline.instance_snapshot.is_empty());
    }

    #[test]
    fn rejecting_oversized_frames() {
        let config = PipelineConfig {
            max_draw_calls: 0,
            ..PipelineConfig::default()
        };
        let mut pipeline = SpherePipeline::new(config);
        assert!(pipeline.submit(&snapshot_with_one_instance()).is_err());
    }

    #[test]
    fn screen_center_ray_points_along_view_direction() {
        let mut pipeline = SpherePipeline::new(PipelineConfig::default());
        let camera = CameraSnapshot {
            position_world_m: DVec3::new(0.0, 0.0, 6_371_000.0),
            orientation: glam::DQuat::IDENTITY,
            ..CameraSnapshot::default()
        };
        pipeline.camera = camera;
        pipeline.origin = FloatingOrigin::new(camera.position_world_m);
        let (origin, direction) = pipeline.screen_to_ray(0.0, 0.0, 16.0 / 9.0).unwrap();
        assert!(origin.length() < 1e-6, "原点即相机位置");
        assert!((direction - DVec3::Z).length() < 1e-9);
        assert!(pipeline.screen_to_ray(0.0, 0.0, 0.0).is_none());
    }
}
