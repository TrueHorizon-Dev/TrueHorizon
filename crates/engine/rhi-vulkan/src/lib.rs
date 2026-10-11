//! # rhi-vulkan —— Vulkan 硬件层
//!
//! 本 crate 是**唯一允许出现 `unsafe` 的边界之一**（另一个是 `world-meshing` 的 SIMD 路径）。
//! 上层只接触本 crate 提供的安全抽象：实例、设备、分配器、无绑定描述符。
//!
//! ```text
//! VulkanLoader ──► VulkanInstance ──► VulkanBackend ──► GpuAllocator
//!   (loader)        (vkInstance)      (device + queues)   (gpu-allocator)
//!        └── ash-window ──► VkSurfaceKHR（窗口路径）
//! ```
//!
//! 目标特性集（Vulkan 1.3+）：
//!
//! | 能力 | 用途 |
//! |------|------|
//! | `VK_EXT_mesh_shader` | 球面 patch 直接由 mesh shader 展开，省掉顶点拉取 |
//! | `descriptor_indexing` | 无绑定纹理/材质表（bindless） |
//! | `buffer_device_address` | GPU 侧指针，供 GPU-Driven 剔除回读 |
//! | `timeline_semaphore` | 帧内多队列同步 |
//!
//! 现状（**每一步都可运行，且失败不 panic**）：
//!
//! 1. [`VulkanLoader`] 动态加载 loader 并报告其支持的实例版本；
//! 2. [`InstanceConfig`] 校验必需扩展/层（缺失即 [`RhiError::ExtensionMissing`] /
//!    [`RhiError::LayerMissing`]），[`VulkanInstance`] 负责实例生命周期；
//! 3. [`select_best_candidate`] 按设备类型/特性/显存做**确定性**评分并选出物理设备，
//!    [`select_queue_families`] 选出图形+计算族与传输族；
//! 4. [`VulkanBackend`] 创建逻辑设备、取回队列句柄，窗口路径下经 `ash-window` 创建
//!    `VkSurfaceKHR`（平台句柄 → surface 的胶水只在本 crate 出现）；
//! 5. [`GpuAllocator`] 封装 `gpu-allocator`，支持子分配与专用分配并记账。
//!
//! 没有 Vulkan 的环境（缺 loader、只有 loader 没 ICD、驱动低于 1.3、没有合适设备）
//! 一律得到明确的 [`RhiError`]，上层据此降级——`dedicated-server` 与 `tests/integration`
//! 因此在无 GPU 的 CI 里照常构建与测试。所有 GPU 相关测试都写成「有设备就跑真实路径，
//! 没有就跳过」，不做「假装有 GPU」的桩。
//!
//! 尚未实现（路线图）：交换链与呈现、管线/描述符的真实绑定、mesh shader 管线的 SPIR-V 模块装配。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod allocator;
pub mod descriptors;
pub mod device;
pub mod instance;
pub mod probe;

pub use allocator::{AllocationDesc, GpuAllocator, MemoryClass};
pub use descriptors::{
    BindlessLayout, DescriptorArena, MAX_BINDLESS_SAMPLERS, MAX_BINDLESS_TEXTURES,
};
pub use device::{
    select_best_candidate, select_queue_families, DeviceCandidate, DeviceInfo, FeatureRequirements,
    QueueFamilies, VulkanBackend,
};
pub use instance::{InstanceConfig, VulkanInstance, VulkanLoader};
pub use probe::{probe, RhiProbe};

/// RHI 层错误。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RhiError {
    /// 加载不到 Vulkan loader（`vulkan-1.dll` / `libvulkan.so.1`）。
    NoVulkanImplementation {
        /// 底层加载失败原因。
        reason: String,
    },
    /// 实例或逻辑设备创建失败（例如只有 loader、没有 ICD 驱动）。
    InstanceCreationFailed(String),
    /// 当前平台/显示句柄无法映射到 Vulkan 表面扩展。
    SurfaceUnsupported(String),
    /// 表面创建失败。
    SurfaceCreationFailed(String),
    /// 没有满足特性要求的物理设备。
    NoSuitableDevice {
        /// 缺失的特性列表。
        missing: Vec<&'static str>,
    },
    /// 缺少必需的扩展。
    ExtensionMissing(&'static str),
    /// 缺少必需的层。
    LayerMissing(&'static str),
    /// 显存分配失败。
    AllocationFailed(String),
    /// 该路径尚未实现（骨架阶段的显式占位）。
    NotImplemented(&'static str),
}

impl core::fmt::Display for RhiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoVulkanImplementation { reason } => {
                write!(f, "找不到 Vulkan 实现（缺少 loader 或驱动）: {reason}")
            },
            Self::InstanceCreationFailed(message) => {
                write!(f, "Vulkan 实例/设备创建失败: {message}")
            },
            Self::SurfaceUnsupported(message) => write!(f, "该平台无法创建 Vulkan 表面: {message}"),
            Self::SurfaceCreationFailed(message) => write!(f, "Vulkan 表面创建失败: {message}"),
            Self::NoSuitableDevice { missing } => {
                write!(
                    f,
                    "没有满足要求的物理设备，缺失特性: {}",
                    missing.join(", ")
                )
            },
            Self::ExtensionMissing(ext) => write!(f, "缺少必需扩展: {ext}"),
            Self::LayerMissing(layer) => write!(f, "缺少必需层: {layer}"),
            Self::AllocationFailed(msg) => write!(f, "显存分配失败: {msg}"),
            Self::NotImplemented(what) => write!(f, "尚未实现: {what}"),
        }
    }
}

impl std::error::Error for RhiError {}

/// 编译期常量：本引擎要求的最低 Vulkan 版本（1.3.0）。
pub const REQUIRED_API_VERSION: u32 = ash::vk::make_api_version(0, 1, 3, 0);

/// `1.3.280` 形式的可读版本串。
pub fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        ash::vk::api_version_major(version),
        ash::vk::api_version_minor(version),
        ash::vk::api_version_patch(version)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_version_encodes_1_3() {
        assert_eq!(ash::vk::api_version_major(REQUIRED_API_VERSION), 1);
        assert_eq!(ash::vk::api_version_minor(REQUIRED_API_VERSION), 3);
        assert_eq!(ash::vk::api_version_patch(REQUIRED_API_VERSION), 0);
        assert_eq!(version_string(REQUIRED_API_VERSION), "1.3.0");
    }

    /// 有 GPU 就拿到真设备；没有 GPU 就必须是可诊断错误——**任何时候都不许 panic**。
    #[test]
    fn backend_creation_degrades_without_a_gpu() {
        match VulkanBackend::new(&FeatureRequirements::minimum()) {
            Ok(backend) => {
                assert!(
                    backend.info().api_version >= REQUIRED_API_VERSION,
                    "选中的设备必须满足 Vulkan 1.3"
                );
                assert!(backend
                    .info()
                    .missing_requirements(&FeatureRequirements::minimum())
                    .is_empty());
                assert!(backend.allocator().is_ok(), "有设备时必须能创建分配器");
                assert!(backend.surface().is_none(), "无头创建不应有表面");
            },
            Err(error) => {
                assert!(
                    matches!(
                        error,
                        RhiError::NoVulkanImplementation { .. }
                            | RhiError::InstanceCreationFailed(_)
                            | RhiError::NoSuitableDevice { .. }
                            | RhiError::ExtensionMissing(_)
                            | RhiError::LayerMissing(_)
                    ),
                    "无 GPU 环境必须返回可诊断错误，实际: {error}"
                );
                assert!(!error.to_string().is_empty());
            },
        }
    }

    #[test]
    fn not_implemented_error_is_explicit() {
        let error = RhiError::NotImplemented("交换链");
        assert!(error.to_string().contains("交换链"));
    }
}
