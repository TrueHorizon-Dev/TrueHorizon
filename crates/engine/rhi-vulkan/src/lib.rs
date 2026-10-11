//! # rhi-vulkan —— Vulkan 硬件层
//!
//! 本 crate 是**唯一允许出现 `unsafe` 的边界之一**（另一个是 `world-meshing` 的 SIMD 路径）。
//! 上层只接触本 crate 提供的安全抽象：设备、分配器、无绑定描述符。
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
//! 骨架现状：类型与能力校验已就位，**实例/设备的真实创建路径尚未实现**，
//! 相关入口返回 [`RhiError::NotImplemented`] 而不是 `panic!`——这样上层可以在
//! 无 Vulkan 环境的 CI 里照常构建与测试（`dedicated-server`、`tests/integration`）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod allocator;
pub mod descriptors;
pub mod device;

pub use allocator::{AllocationDesc, MemoryClass};
pub use descriptors::{
    BindlessLayout, DescriptorArena, MAX_BINDLESS_SAMPLERS, MAX_BINDLESS_TEXTURES,
};
pub use device::{DeviceInfo, FeatureRequirements, QueueFamilies, VulkanBackend};

/// RHI 层错误。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RhiError {
    /// 目标平台上没有可用的 Vulkan 实现。
    NoVulkanImplementation,
    /// 没有满足特性要求的物理设备。
    NoSuitableDevice {
        /// 缺失的特性列表。
        missing: Vec<&'static str>,
    },
    /// 缺少必需的扩展。
    ExtensionMissing(&'static str),
    /// 显存分配失败。
    AllocationFailed(String),
    /// 该路径尚未实现（骨架阶段的显式占位）。
    NotImplemented(&'static str),
}

impl core::fmt::Display for RhiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoVulkanImplementation => write!(f, "找不到 Vulkan 实现（缺少 loader 或驱动）"),
            Self::NoSuitableDevice { missing } => {
                write!(
                    f,
                    "没有满足要求的物理设备，缺失特性: {}",
                    missing.join(", ")
                )
            }
            Self::ExtensionMissing(ext) => write!(f, "缺少必需扩展: {ext}"),
            Self::AllocationFailed(msg) => write!(f, "显存分配失败: {msg}"),
            Self::NotImplemented(what) => write!(f, "尚未实现: {what}"),
        }
    }
}

impl std::error::Error for RhiError {}

/// 编译期常量：本引擎要求的最低 Vulkan 版本（1.3.0）。
pub const REQUIRED_API_VERSION: u32 = ash::vk::make_api_version(0, 1, 3, 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_version_encodes_1_3() {
        assert_eq!(ash::vk::api_version_major(REQUIRED_API_VERSION), 1);
        assert_eq!(ash::vk::api_version_minor(REQUIRED_API_VERSION), 3);
        assert_eq!(ash::vk::api_version_patch(REQUIRED_API_VERSION), 0);
    }

    #[test]
    fn not_implemented_is_a_recoverable_error() {
        let err = VulkanBackend::new(&FeatureRequirements::minimum());
        assert!(matches!(err, Err(RhiError::NotImplemented(_))));
    }
}
