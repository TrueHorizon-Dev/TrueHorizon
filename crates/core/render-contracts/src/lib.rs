//! # render-contracts —— 渲染契约
//!
//! 定义「世界/仿真」与「渲染器」之间的唯一接口。上层只构造 [`SceneSnapshot`]，
//! 下层只消费命令；双方不共享任何 GPU 类型，因此渲染后端可被完整替换。
//!
//! * [`SceneSnapshot`]：一帧的全部输入（相机、原点、绘制命令、实例数据）。
//! * [`DrawCall`] / [`InstanceRaw`]：**GPU 可直接消费**的 `Pod` 布局。
//! * [`RenderContract`]：后端需要实现的唯一 trait。
//!
//! 数据布局纪律：本 crate 中所有 `#[repr(C)] + Pod` 的类型都会零拷贝上传显存，
//! 因此**字段顺序与对齐即为 ABI**，改动必须同步 shader 侧定义并升级版本号。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use glam::{DQuat, DVec3, Mat4};
use planet_coords::{FloatingOrigin, SurfacePoint};

// `DrawCall::new` 以 `Material` 为参数，因此必须把它重新导出：
// 否则外部调用方无法命名这个类型（参数类型不可达的公开 API 是设计缺陷）。
pub use block_schema::Material;

/// 网格在显存池中的句柄（索引，由渲染后端分配）。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct MeshHandle(pub u32);

impl MeshHandle {
    /// 无效句柄。
    pub const INVALID: Self = Self(u32::MAX);

    /// 是否有效。
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

/// 绘制桶。提交顺序即渲染顺序，半透明必须在最后。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum DrawBucket {
    /// 不透明。
    Opaque = 0,
    /// 镂空。
    Cutout = 1,
    /// 半透明（按深度排序）。
    Translucent = 2,
}

impl DrawBucket {
    /// 全部桶，顺序即提交顺序。
    pub const ALL: [DrawBucket; 3] = [
        DrawBucket::Opaque,
        DrawBucket::Cutout,
        DrawBucket::Translucent,
    ];
}

/// 单条绘制命令，GPU 侧按 `storage buffer` 直接索引。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawCall {
    /// 网格句柄。
    pub mesh: u32,
    /// 实例缓冲中的起始下标。
    pub first_instance: u32,
    /// 实例数量。
    pub instance_count: u32,
    /// 材质槽位（[`Material::id`]）。
    pub material_id: u32,
    /// 管线标志位（背面剔除、双面、是否走 mesh shader 路径等）。
    pub flags: u32,
    /// 该批次选中的 LOD 层级。
    pub lod_level: u32,
}

impl DrawCall {
    /// 空命令。
    pub const EMPTY: Self = Self {
        mesh: MeshHandle::INVALID.0,
        first_instance: 0,
        instance_count: 0,
        material_id: 0,
        flags: 0,
        lod_level: 0,
    };

    /// 由材质与实例区间构造。
    pub fn new(
        mesh: MeshHandle,
        material: Material,
        first_instance: u32,
        instance_count: u32,
    ) -> Self {
        Self {
            mesh: mesh.0,
            first_instance,
            instance_count,
            material_id: material.id(),
            flags: 0,
            lod_level: 0,
        }
    }

    /// 该命令的网格句柄。
    #[inline]
    pub fn mesh_handle(&self) -> MeshHandle {
        MeshHandle(self.mesh)
    }

    /// 是否为空命令（无实例或无效网格）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0 || !self.mesh_handle().is_valid()
    }
}

/// 单个实例的变换与材质覆盖，`Pod` 布局直接上传。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// 相对浮动原点的位置（米，`f32`）。
    pub translation: [f32; 3],
    /// 网格槽位（同一缓冲内的子网格索引）。
    pub mesh_slot: u32,
    /// 旋转四元数 `(x, y, z, w)`。
    pub rotation: [f32; 4],
    /// 均匀缩放。
    pub scale: f32,
    /// 材质槽位。
    pub material_id: u32,
    /// 自定义标志位（如"刚被破坏，播放碎裂动画"）。
    pub flags: u32,
    /// 预留对齐字段，保持 48 字节步长。
    pub _reserved: u32,
}

impl InstanceRaw {
    /// 由世界坐标（双精度）构造，内部自动扣掉浮动原点。
    pub fn from_world(
        origin: FloatingOrigin,
        world_m: DVec3,
        mesh_slot: u32,
        material: Material,
    ) -> Self {
        let local = origin.to_render(world_m);
        Self {
            translation: [local.x, local.y, local.z],
            mesh_slot,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: 1.0,
            material_id: material.id(),
            flags: 0,
            _reserved: 0,
        }
    }
}

/// 相机快照。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraSnapshot {
    /// 世界坐标位置（米，`f64`）。
    pub position_world_m: DVec3,
    /// 朝向（世界系）。
    pub orientation: DQuat,
    /// 垂直视场角（弧度）。
    pub fov_y_rad: f64,
    /// 近裁剪面（米，相对相机；在浮动原点下可以小到 0.05）。
    pub near_m: f32,
    /// 远裁剪面（米，相对相机；配合曲率遮挡，地面视角通常 200 km 以内）。
    pub far_m: f32,
    /// 观察者所在高度（米，用于地平线裁剪）。
    pub height_m: f64,
}

impl Default for CameraSnapshot {
    fn default() -> Self {
        Self {
            position_world_m: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            fov_y_rad: 70f64.to_radians(),
            near_m: 0.05,
            far_m: 200_000.0,
            height_m: 0.0,
        }
    }
}

impl CameraSnapshot {
    /// 视图矩阵（相对浮动原点，`f32`）。
    pub fn view_matrix(&self, origin: FloatingOrigin) -> Mat4 {
        let eye = origin.to_render(self.position_world_m);
        let rotation = Mat4::from_quat(self.orientation.as_quat());
        Mat4::from_translation(-eye) * rotation
    }

    /// 投影矩阵（Vulkan 深度的 `[0, 1]` 约定，Y 轴朝下）。
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_rad as f32,
            aspect.max(1e-4),
            self.near_m.max(1e-4),
            self.far_m,
        )
    }
}

/// 一帧的完整渲染输入。
///
/// 生命周期：由世界/仿真线程构造 → 提交给渲染线程 → 渲染线程在帧结束前持有。
/// 构造后**不再修改**（`SceneSnapshot` 一旦提交即为不可变快照，便于并行编码）。
#[derive(Clone, Debug, Default)]
pub struct SceneSnapshot {
    /// 单调递增的帧号，用于与仿真的固定 Tick 对账。
    pub frame_index: u64,
    /// 本帧的浮动原点。
    pub origin: FloatingOrigin,
    /// 相机。
    pub camera: CameraSnapshot,
    /// 不透明命令。
    pub opaque: Vec<DrawCall>,
    /// 镂空命令。
    pub cutout: Vec<DrawCall>,
    /// 半透明命令。
    pub translucent: Vec<DrawCall>,
    /// 全部实例（三个桶共享一个实例缓冲，按 `first_instance/count` 切片）。
    pub instances: Vec<InstanceRaw>,
}

impl SceneSnapshot {
    /// 空帧。
    pub fn new(frame_index: u64, origin: FloatingOrigin, camera: CameraSnapshot) -> Self {
        Self {
            frame_index,
            origin,
            camera,
            ..Default::default()
        }
    }

    /// 按桶取得命令切片。
    pub fn bucket(&self, bucket: DrawBucket) -> &[DrawCall] {
        match bucket {
            DrawBucket::Opaque => &self.opaque,
            DrawBucket::Cutout => &self.cutout,
            DrawBucket::Translucent => &self.translucent,
        }
    }

    /// 按桶取得命令的可变切片。
    pub fn bucket_mut(&mut self, bucket: DrawBucket) -> &mut Vec<DrawCall> {
        match bucket {
            DrawBucket::Opaque => &mut self.opaque,
            DrawBucket::Cutout => &mut self.cutout,
            DrawBucket::Translucent => &mut self.translucent,
        }
    }

    /// 命令总数。
    pub fn draw_call_count(&self) -> usize {
        self.opaque.len() + self.cutout.len() + self.translucent.len()
    }

    /// 清空为下一帧复用（保留已分配容量）。
    pub fn clear(&mut self) {
        self.opaque.clear();
        self.cutout.clear();
        self.translucent.clear();
        self.instances.clear();
    }

    /// 校验不变量：实例区间必须在实例缓冲范围内。
    ///
    /// 提交前调用；返回 `Err` 说明上层构造逻辑有 bug（越界会让 GPU 读到垃圾数据或崩溃）。
    pub fn validate(&self) -> Result<(), SceneError> {
        let total = self.instances.len() as u64;
        for bucket in DrawBucket::ALL {
            for (index, call) in self.bucket(bucket).iter().enumerate() {
                let end = call.first_instance as u64 + call.instance_count as u64;
                if end > total {
                    return Err(SceneError::InstanceRangeOutOfBounds {
                        bucket,
                        index,
                        start: call.first_instance,
                        count: call.instance_count,
                        available: self.instances.len(),
                    });
                }
                if call.instance_count > 0 && !call.mesh_handle().is_valid() {
                    return Err(SceneError::InvalidMesh { bucket, index });
                }
            }
        }
        Ok(())
    }
}

/// 快照校验错误。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneError {
    /// 实例区间越界。
    InstanceRangeOutOfBounds {
        /// 出错的桶。
        bucket: DrawBucket,
        /// 桶内下标。
        index: usize,
        /// 起始实例下标。
        start: u32,
        /// 实例数量。
        count: u32,
        /// 实例缓冲实际长度。
        available: usize,
    },
    /// 命令引用了无效网格。
    InvalidMesh {
        /// 出错的桶。
        bucket: DrawBucket,
        /// 桶内下标。
        index: usize,
    },
}

impl core::fmt::Display for SceneError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InstanceRangeOutOfBounds {
                bucket,
                index,
                start,
                count,
                available,
            } => write!(
                f,
                "{bucket:?} 桶第 {index} 条命令的实例区间 [{start}, {} ) 越界（共 {available} 个实例）",
                *start as u64 + *count as u64
            ),
            Self::InvalidMesh { bucket, index } => {
                write!(f, "{bucket:?} 桶第 {index} 条命令引用了无效网格句柄")
            }
        }
    }
}

impl std::error::Error for SceneError {}

/// 渲染后端需要实现的唯一契约。
///
/// 实现方（`render-pipeline`）负责：上传实例缓冲、录制命令、按桶顺序提交。
pub trait RenderContract: Send {
    /// 提交一帧。
    fn submit(&mut self, snapshot: &SceneSnapshot) -> Result<(), SceneError>;

    /// 通知渲染器浮动原点已搬移，需要刷新所有缓存的渲染侧位置。
    fn on_origin_shifted(&mut self, origin: FloatingOrigin);

    /// 屏幕 NDC → 世界射线（光标拾取的反向查询）。
    ///
    /// 返回 `(起点, 单位方向)`，均为相对当前浮动原点的渲染坐标。
    fn screen_to_ray(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Option<(DVec3, DVec3)>;
}

/// 便利函数：表面点 → 世界坐标（米）。
pub fn surface_world_m(point: SurfacePoint, radius_m: f64, elevation_m: f64) -> DVec3 {
    point.to_dir() * (radius_m + elevation_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_layouts_have_stable_strides() {
        assert_eq!(
            core::mem::size_of::<DrawCall>(),
            24,
            "DrawCall 必须为 6×u32"
        );
        assert_eq!(
            core::mem::size_of::<InstanceRaw>(),
            48,
            "InstanceRaw 必须为 48 字节无填充"
        );
    }

    #[test]
    fn validate_rejects_out_of_range_instances() {
        let mut snapshot = SceneSnapshot::default();
        snapshot.instances.push(InstanceRaw {
            translation: [0.0; 3],
            mesh_slot: 0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: 1.0,
            material_id: Material::Rock.id(),
            flags: 0,
            _reserved: 0,
        });
        snapshot
            .opaque
            .push(DrawCall::new(MeshHandle(3), Material::Rock, 0, 1));
        assert!(snapshot.validate().is_ok());

        snapshot
            .opaque
            .push(DrawCall::new(MeshHandle(3), Material::Rock, 1, 4));
        assert!(matches!(
            snapshot.validate(),
            Err(SceneError::InstanceRangeOutOfBounds { .. })
        ));
    }

    #[test]
    fn invalid_mesh_is_caught() {
        let mut snapshot = SceneSnapshot::default();
        snapshot.instances.push(InstanceRaw::from_world(
            FloatingOrigin::default(),
            DVec3::new(1.0, 2.0, 3.0),
            0,
            Material::Soil,
        ));
        snapshot
            .cutout
            .push(DrawCall::new(MeshHandle::INVALID, Material::Soil, 0, 1));
        assert_eq!(
            snapshot.validate(),
            Err(SceneError::InvalidMesh {
                bucket: DrawBucket::Cutout,
                index: 0
            })
        );
    }

    #[test]
    fn view_matrix_cancels_far_origin() {
        let origin = FloatingOrigin::new(DVec3::new(6_000_000.0, 0.0, 0.0));
        let camera = CameraSnapshot {
            position_world_m: DVec3::new(6_000_100.0, 0.0, 0.0),
            ..Default::default()
        };
        let view = camera.view_matrix(origin);
        assert!(view.is_finite());
        // 相机在渲染空间的 X 偏移应为 100 米，其余为 0。
        assert!((view.w_axis.x + 100.0).abs() < 1e-3, "view = {view:?}");
    }
}
