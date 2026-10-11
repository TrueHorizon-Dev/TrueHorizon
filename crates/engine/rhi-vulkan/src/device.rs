//! 物理设备选择与逻辑设备。

use crate::{RhiError, REQUIRED_API_VERSION};
use ash::vk;

/// 队列族索引。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct QueueFamilies {
    /// 图形 + 计算队列族（本引擎强制二者同族，简化同步）。
    pub graphics_compute: u32,
    /// 传输队列族（异步上传）。
    pub transfer: u32,
    /// 是否与图形族不同，需要额外的队列所有权转移。
    pub transfer_is_separate: bool,
}

/// 运行时能力要求。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureRequirements {
    /// 要求 Vulkan 1.3 核心特性。
    pub vulkan_13: bool,
    /// 要求 `VK_EXT_mesh_shader`。
    pub mesh_shader: bool,
    /// 要求 `descriptor_indexing`（bindless）。
    pub descriptor_indexing: bool,
    /// 要求 `buffer_device_address`。
    pub buffer_device_address: bool,
    /// 要求 `timeline_semaphore`。
    pub timeline_semaphore: bool,
    /// 要求填充模式为 `SOLID` 的几何。
    pub fill_mode_non_solid: bool,
}

impl FeatureRequirements {
    /// 最低要求：只要 Vulkan 1.3 核心（供无头 CI 与降级路径使用）。
    pub const fn minimum() -> Self {
        Self {
            vulkan_13: true,
            mesh_shader: false,
            descriptor_indexing: false,
            buffer_device_address: false,
            timeline_semaphore: false,
            fill_mode_non_solid: false,
        }
    }

    /// 桌面客户端目标：全特性。
    pub const fn desktop_full() -> Self {
        Self {
            vulkan_13: true,
            mesh_shader: true,
            descriptor_indexing: true,
            buffer_device_address: true,
            timeline_semaphore: true,
            fill_mode_non_solid: false,
        }
    }
}

impl Default for FeatureRequirements {
    fn default() -> Self {
        Self::minimum()
    }
}

/// 已选定设备的信息摘要（用于日志、诊断报告与 `world-tool diagnose`）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// 设备名（驱动上报）。
    pub name: String,
    /// 设备上报的 Vulkan 版本（编码后的 `u32`）。
    pub api_version: u32,
    /// 驱动版本（编码后的 `u32`）。
    pub driver_version: u32,
    /// 设备类型（独显/集显/CPU）。
    pub device_type: vk::PhysicalDeviceType,
    /// 是否支持 mesh shader。
    pub mesh_shader: bool,
    /// 是否支持无绑定描述符。
    pub descriptor_indexing: bool,
    /// 显存堆总大小（字节），无法查询时为 0。
    pub vram_bytes: u64,
}

impl DeviceInfo {
    /// `1.3.280` 形式的可读版本串。
    pub fn api_version_string(&self) -> String {
        format!(
            "{}.{}.{}",
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version)
        )
    }

    /// 是否满足全部要求；返回缺失项（空表示全部满足）。
    pub fn missing_requirements(&self, required: &FeatureRequirements) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if required.vulkan_13 && self.api_version < REQUIRED_API_VERSION {
            missing.push("Vulkan 1.3");
        }
        if required.mesh_shader && !self.mesh_shader {
            missing.push("VK_EXT_mesh_shader");
        }
        if required.descriptor_indexing && !self.descriptor_indexing {
            missing.push("descriptor_indexing");
        }
        missing
    }
}

/// Vulkan 后端：持有实例、逻辑设备与队列。
///
/// 目前只能由 [`VulkanBackend::new`] 返回 [`RhiError::NotImplemented`]；
/// 真实创建流程（loader 加载 → 实例 → 物理设备评分 → 逻辑设备 → 队列）将在
/// `rhi-vulkan` 的实现阶段补齐，届时本类型成为上层唯一可见的 GPU 入口。
pub struct VulkanBackend {
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    physical_device: vk::PhysicalDevice,
    queues: QueueFamilies,
    info: DeviceInfo,
}

impl VulkanBackend {
    /// 按特性要求创建后端。
    pub fn new(requirements: &FeatureRequirements) -> Result<Self, RhiError> {
        // TODO(rhi): 1) ash::Entry 加载 loader；2) 校验实例扩展；
        //            3) ash::Instance 创建；4) 物理设备评分与选择；
        //            5) ash::Device 与队列创建。见 docs/ROADMAP.md。
        let _ = vk::api_version_major(REQUIRED_API_VERSION);
        let _ = requirements;
        Err(RhiError::NotImplemented(
            "Vulkan 实例/设备创建（rhi-vulkan::VulkanBackend::new）",
        ))
    }

    /// 选定的物理设备信息。
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// 队列族。
    pub fn queues(&self) -> QueueFamilies {
        self.queues
    }

    /// 原始逻辑设备句柄（仅供本 crate 内部与 `render-pipeline` 使用）。
    pub fn raw_device(&self) -> Option<&ash::Device> {
        self.device.as_ref()
    }

    /// 原始实例句柄。
    pub fn raw_instance(&self) -> Option<&ash::Instance> {
        self.instance.as_ref()
    }

    /// 物理设备句柄。
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_info(version: u32) -> DeviceInfo {
        DeviceInfo {
            name: "Fake GPU".into(),
            api_version: version,
            driver_version: 0,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            mesh_shader: false,
            descriptor_indexing: false,
            vram_bytes: 8 << 30,
        }
    }

    #[test]
    fn missing_requirements_are_reported_not_silently_ignored() {
        let info = fake_info(REQUIRED_API_VERSION);
        let missing = info.missing_requirements(&FeatureRequirements::desktop_full());
        assert_eq!(missing, vec!["VK_EXT_mesh_shader", "descriptor_indexing"]);
        assert!(info
            .missing_requirements(&FeatureRequirements::minimum())
            .is_empty());
    }

    #[test]
    fn version_string_is_readable() {
        let info = fake_info(ash::vk::make_api_version(0, 1, 3, 280));
        assert_eq!(info.api_version_string(), "1.3.280");
    }

    #[test]
    fn old_device_fails_minimum() {
        let info = fake_info(ash::vk::make_api_version(0, 1, 2, 0));
        assert_eq!(
            info.missing_requirements(&FeatureRequirements::minimum()),
            vec!["Vulkan 1.3"]
        );
    }
}
