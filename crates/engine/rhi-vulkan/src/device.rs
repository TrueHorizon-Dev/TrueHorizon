//! 物理设备评分、逻辑设备创建与队列。
//!
//! 评分与队列选择被刻意拆成**纯函数**（[`select_best_candidate`]、[`select_queue_families`]），
//! 这样「选哪块 GPU、选哪个队列族」的策略可以在无 GPU 的机器上被单元测试覆盖；
//! 真正调用 Vulkan 的部分只做「把查询结果喂给纯函数」与错误映射。

use crate::allocator::GpuAllocator;
use crate::instance::{InstanceConfig, VulkanInstance, VulkanLoader};
use crate::{version_string, RhiError, REQUIRED_API_VERSION};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::CStr;

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
    /// 是否支持 `buffer_device_address`。
    pub buffer_device_address: bool,
    /// 是否支持时间线信号量。
    pub timeline_semaphore: bool,
    /// 显存堆总大小（字节），无法查询时为 0。
    pub vram_bytes: u64,
}

impl DeviceInfo {
    /// `1.3.280` 形式的可读版本串。
    pub fn api_version_string(&self) -> String {
        version_string(self.api_version)
    }

    /// 是否满足全部要求；返回缺失项（空表示全部满足）。
    ///
    /// 覆盖 [`FeatureRequirements`] 的**每一个**开关：漏检会让上层以为拿到了 bindless /
    /// 时间线信号量，直到运行期才在驱动里炸开。
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
        if required.buffer_device_address && !self.buffer_device_address {
            missing.push("buffer_device_address");
        }
        if required.timeline_semaphore && !self.timeline_semaphore {
            missing.push("timeline_semaphore");
        }
        missing
    }
}

/// 一个物理设备候选（查询结果 + 队列族）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCandidate {
    /// 在 `vkEnumeratePhysicalDevices` 返回列表中的下标（用于稳定裁决与回查句柄）。
    pub index: usize,
    /// 能力摘要。
    pub info: DeviceInfo,
    /// 选中的队列族。
    pub queues: QueueFamilies,
}

impl DeviceCandidate {
    /// 是否满足全部要求。
    pub fn satisfies(&self, required: &FeatureRequirements) -> bool {
        self.info.missing_requirements(required).is_empty()
    }

    /// 确定性评分：设备类型 → 特性 → 显存。
    ///
    /// 同一台机器上重复运行结果恒定（不含时间、线程、地址等非确定输入），
    /// 因此「选中的设备」本身可复现，便于对照性能数据。
    pub fn score(&self) -> i64 {
        let type_score: i64 = match self.info.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3000,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1000,
            vk::PhysicalDeviceType::CPU => 500,
            _ => 0,
        };
        let feature_score = i64::from(self.info.mesh_shader) * 200
            + i64::from(self.info.descriptor_indexing) * 100
            + i64::from(self.info.buffer_device_address) * 50
            + i64::from(self.info.timeline_semaphore) * 50;
        let vram_gib = (self.info.vram_bytes / (1 << 30)) as i64;
        type_score + feature_score + vram_gib * 10
    }
}

/// 从候选中挑出最优设备，返回其在切片中的下标。
///
/// 平手时取枚举顺序靠前者（确定性）。没有候选满足要求时，报告**评分最高**的候选缺失了哪些项，
/// 这样日志里能直接看到「差在哪」而不是一句「没有可用设备」。
pub fn select_best_candidate(
    candidates: &[DeviceCandidate],
    required: &FeatureRequirements,
) -> Result<usize, RhiError> {
    let mut best: Option<(usize, i64)> = None;
    for (slice_index, candidate) in candidates.iter().enumerate() {
        if !candidate.satisfies(required) {
            continue;
        }
        let score = candidate.score();
        match best {
            Some((_, best_score)) if best_score >= score => {},
            _ => best = Some((slice_index, score)),
        }
    }
    if let Some((slice_index, _)) = best {
        return Ok(slice_index);
    }

    let closest = candidates.iter().max_by(|left, right| {
        left.score()
            .cmp(&right.score())
            .then_with(|| right.index.cmp(&left.index))
    });
    let missing = match closest {
        Some(candidate) => candidate.info.missing_requirements(required),
        None => vec!["任何 Vulkan 物理设备"],
    };
    Err(RhiError::NoSuitableDevice { missing })
}

/// 选择图形/计算与传输队列族。
///
/// * 图形族必须同时支持 `GRAPHICS | COMPUTE`——本引擎的球面 LOD 剔除与网格化都在计算队列上，
///   拆成两族会引入额外的所有权转移与信号量；
/// * 有 surface 时该族还必须支持呈现（`present_support`），否则会选中一块「能算不能画」的设备；
/// * 传输族优先取**独立的** `TRANSFER` 族（避免上传与绘制争用），没有就复用图形族。
pub fn select_queue_families(
    families: &[vk::QueueFamilyProperties],
    present_support: Option<&[bool]>,
) -> Option<QueueFamilies> {
    let supports_present = |index: usize| {
        present_support
            .map(|flags| flags.get(index).copied().unwrap_or(false))
            .unwrap_or(true)
    };

    let graphics_compute = families
        .iter()
        .enumerate()
        .find(|(index, family)| {
            family.queue_count > 0
                && family
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                && supports_present(*index)
        })
        .map(|(index, _)| index as u32)?;

    let transfer = families
        .iter()
        .enumerate()
        .find(|(index, family)| {
            *index as u32 != graphics_compute
                && family.queue_count > 0
                && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
        })
        .map(|(index, _)| index as u32)
        .unwrap_or(graphics_compute);

    Some(QueueFamilies {
        graphics_compute,
        transfer,
        transfer_is_separate: transfer != graphics_compute,
    })
}

/// Vulkan 后端：持有实例、逻辑设备、队列与（可选）窗口表面。
///
/// 创建失败一律返回 [`RhiError`]，**不 panic**：无 Vulkan 的 CI 里上层据此走 CPU 降级路径。
pub struct VulkanBackend {
    instance: VulkanInstance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queues: QueueFamilies,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    info: DeviceInfo,
    surface: Option<vk::SurfaceKHR>,
    device_extensions: Vec<String>,
}

/// 逻辑设备创建请求（把参数聚成结构体，避免长出八参数函数）。
struct LogicalDeviceRequest<'a> {
    requirements: &'a FeatureRequirements,
    queues: QueueFamilies,
    info: &'a DeviceInfo,
    with_surface: bool,
}

impl VulkanBackend {
    /// 无窗口（无头）创建：不要求任何表面扩展，供专用服务器与离线工具使用。
    pub fn new(requirements: &FeatureRequirements) -> Result<Self, RhiError> {
        Self::create(requirements, None, InstanceConfig::headless())
    }

    /// 为窗口创建：自动追加 `VK_KHR_surface` 与平台扩展，并创建 `VkSurfaceKHR`。
    ///
    /// 调用方必须保证窗口在返回的后端（及其 surface）存活期间不被销毁。
    pub fn new_for_window(
        requirements: &FeatureRequirements,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self, RhiError> {
        let config = InstanceConfig::headless().with_surface(display_handle)?;
        Self::create(requirements, Some((display_handle, window_handle)), config)
    }

    /// 创建流程：loader → 实例 → 物理设备评分 → 逻辑设备与队列 →（可选）surface。
    fn create(
        requirements: &FeatureRequirements,
        window: Option<(RawDisplayHandle, RawWindowHandle)>,
        config: InstanceConfig,
    ) -> Result<Self, RhiError> {
        let loader = VulkanLoader::load()?;
        if requirements.vulkan_13 {
            let loader_version = loader.instance_api_version()?;
            if loader_version < REQUIRED_API_VERSION {
                return Err(RhiError::InstanceCreationFailed(format!(
                    "loader 只提供 Vulkan {}，低于要求的 1.3",
                    version_string(loader_version)
                )));
            }
        }
        // 校验层只在「调试构建 + 环境里确实存在」时打开：否则开发机没装 SDK 就会直接创建失败。
        let mut config = config;
        if cfg!(debug_assertions) && loader.validation_layer_available() {
            config = config.with_validation(true);
        }
        let instance = loader.create_instance(&config)?;

        let surface = match window {
            Some((display_handle, window_handle)) => {
                Some(instance.create_surface(display_handle, window_handle)?)
            },
            None => None,
        };

        let physical_devices = instance.physical_devices()?;
        let mut candidates = Vec::with_capacity(physical_devices.len());
        for (index, physical_device) in physical_devices.iter().enumerate() {
            match query_candidate(&instance, *physical_device, index) {
                Ok(candidate) => candidates.push(candidate),
                // 队列族不完整的设备（例如纯计算加速卡）不是错误，直接跳过。
                Err(_) => continue,
            }
        }
        let selected = select_best_candidate(&candidates, requirements)?;
        let candidate = candidates[selected].clone();
        let physical_device = physical_devices[candidate.index];

        let logical = create_logical_device(
            &instance,
            physical_device,
            &LogicalDeviceRequest {
                requirements,
                queues: candidate.queues,
                info: &candidate.info,
                with_surface: surface.is_some(),
            },
        )?;

        Ok(Self {
            instance,
            device: logical.device,
            physical_device,
            queues: candidate.queues,
            graphics_queue: logical.graphics_queue,
            transfer_queue: logical.transfer_queue,
            info: candidate.info,
            surface,
            device_extensions: logical.extensions,
        })
    }

    /// 选定的物理设备信息。
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// 队列族。
    pub fn queues(&self) -> QueueFamilies {
        self.queues
    }

    /// 图形/计算队列。
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// 传输队列。
    pub fn transfer_queue(&self) -> vk::Queue {
        self.transfer_queue
    }

    /// 原始逻辑设备句柄。
    pub fn raw_device(&self) -> &ash::Device {
        &self.device
    }

    /// 原始实例句柄。
    pub fn raw_instance(&self) -> &ash::Instance {
        self.instance.handle()
    }

    /// 实例对象（创建 surface 等平台操作需要）。
    pub fn instance(&self) -> &VulkanInstance {
        &self.instance
    }

    /// 物理设备句柄。
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// 窗口表面句柄（无头创建时为 `None`）。
    pub fn surface(&self) -> Option<vk::SurfaceKHR> {
        self.surface
    }

    /// 本次逻辑设备实际启用的扩展名。
    pub fn device_extensions(&self) -> &[String] {
        &self.device_extensions
    }

    /// 创建显存分配器。
    ///
    /// 返回的 [`GpuAllocator`] 必须在**本后端之前**销毁（它持有设备句柄的克隆），
    /// 因此调用方应把它放在比后端更小的作用域里。
    pub fn allocator(&self) -> Result<GpuAllocator, RhiError> {
        GpuAllocator::new(self)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // SAFETY: 设备由本类型独占持有；surface 也由本类型持有，必须在实例销毁前释放。
        // 字段声明顺序保证 `instance` 最后析构。
        unsafe { self.device.destroy_device(None) };
        if let Some(surface) = self.surface.take() {
            self.instance.destroy_surface(surface);
        }
    }
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("info", &self.info)
            .field("queues", &self.queues)
            .field("surface", &self.surface.is_some())
            .field("device_extensions", &self.device_extensions)
            .finish_non_exhaustive()
    }
}

/// 已创建的逻辑设备及其队列。
struct LogicalDevice {
    device: ash::Device,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    extensions: Vec<String>,
}

/// 查询单个物理设备，构造候选。
pub(crate) fn query_candidate(
    instance: &VulkanInstance,
    physical_device: vk::PhysicalDevice,
    index: usize,
) -> Result<DeviceCandidate, RhiError> {
    // SAFETY: 以下全部是对有效物理设备的只读查询。
    let properties = unsafe {
        instance
            .handle()
            .get_physical_device_properties(physical_device)
    };
    let memory = unsafe {
        instance
            .handle()
            .get_physical_device_memory_properties(physical_device)
    };
    let family_properties = unsafe {
        instance
            .handle()
            .get_physical_device_queue_family_properties(physical_device)
    };
    let extensions = unsafe {
        instance
            .handle()
            .enumerate_device_extension_properties(physical_device)
    }
    .unwrap_or_default();

    let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let mut timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut descriptor_indexing)
        .push_next(&mut timeline_semaphore)
        .push_next(&mut buffer_device_address)
        .push_next(&mut mesh_shader);
    // SAFETY: `features` 的链上结构体都在本函数内保持存活，满足 `vkGetPhysicalDeviceFeatures2` 的要求。
    unsafe {
        instance
            .handle()
            .get_physical_device_features2(physical_device, &mut features)
    };

    let queues =
        select_queue_families(&family_properties, None).ok_or(RhiError::NoSuitableDevice {
            missing: vec!["支持 GRAPHICS + COMPUTE 的队列族"],
        })?;

    let vram_bytes = (0..memory.memory_heap_count)
        .filter(|heap| {
            memory.memory_heaps[*heap as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|heap| memory.memory_heaps[heap as usize].size)
        .sum();

    let mesh_shader_supported = mesh_shader.mesh_shader == vk::TRUE
        && extension_supported(&extensions, ash::ext::mesh_shader::NAME);
    // `VkPhysicalDeviceDescriptorIndexingFeatures` 没有单一的 `descriptorIndexing` 开关：
    // bindless 表依赖「运行时数组 + 非一致索引 + update-after-bind + 可变描述符计数」这一组子特性。
    let descriptor_indexing_supported = descriptor_indexing.runtime_descriptor_array == vk::TRUE
        && descriptor_indexing.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
        && descriptor_indexing.descriptor_binding_sampled_image_update_after_bind == vk::TRUE
        && descriptor_indexing.descriptor_binding_variable_descriptor_count == vk::TRUE;

    Ok(DeviceCandidate {
        index,
        info: DeviceInfo {
            name: properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "未知设备".to_string()),
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            device_type: properties.device_type,
            mesh_shader: mesh_shader_supported,
            descriptor_indexing: descriptor_indexing_supported,
            buffer_device_address: buffer_device_address.buffer_device_address == vk::TRUE,
            timeline_semaphore: timeline_semaphore.timeline_semaphore == vk::TRUE,
            vram_bytes,
        },
        queues,
    })
}

/// 设备扩展是否可用。
fn extension_supported(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .map(|found| found == name)
            .unwrap_or(false)
    })
}

/// 按候选创建逻辑设备、取回队列句柄。
fn create_logical_device(
    instance: &VulkanInstance,
    physical_device: vk::PhysicalDevice,
    request: &LogicalDeviceRequest<'_>,
) -> Result<LogicalDevice, RhiError> {
    let priorities = [1.0f32];
    let mut families = vec![request.queues.graphics_compute];
    if request.queues.transfer_is_separate {
        families.push(request.queues.transfer);
    }
    let queue_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = families
        .iter()
        .map(|family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*family)
                .queue_priorities(&priorities)
        })
        .collect();

    // 只启用「要求了 且 设备支持」的特性/扩展；不支持的组合在评分阶段已被排除。
    let needs_mesh_shader = request.requirements.mesh_shader && request.info.mesh_shader;
    let needs_descriptor_indexing =
        request.requirements.descriptor_indexing && request.info.descriptor_indexing;
    let needs_buffer_device_address =
        request.requirements.buffer_device_address && request.info.buffer_device_address;
    let needs_timeline = request.requirements.timeline_semaphore && request.info.timeline_semaphore;

    let mut extension_names: Vec<&'static CStr> = Vec::new();
    if needs_mesh_shader {
        extension_names.push(ash::ext::mesh_shader::NAME);
    }
    if request.with_surface {
        // 呈现能力是设备扩展，必须在逻辑设备上显式启用。
        let device_extensions = unsafe {
            instance
                .handle()
                .enumerate_device_extension_properties(physical_device)
        }
        .unwrap_or_default();
        if !extension_supported(&device_extensions, ash::khr::swapchain::NAME) {
            return Err(RhiError::ExtensionMissing("VK_KHR_swapchain"));
        }
        extension_names.push(ash::khr::swapchain::NAME);
    }
    let extension_pointers: Vec<*const std::os::raw::c_char> =
        extension_names.iter().map(|name| name.as_ptr()).collect();

    let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    if needs_descriptor_indexing {
        descriptor_indexing = descriptor_indexing
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_variable_descriptor_count(true);
    }
    let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    if needs_buffer_device_address {
        buffer_device_address = buffer_device_address.buffer_device_address(true);
    }
    let mut timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    if needs_timeline {
        timeline_semaphore = timeline_semaphore.timeline_semaphore(true);
    }
    let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    if needs_mesh_shader {
        mesh_shader = mesh_shader.mesh_shader(true);
    }

    let mut create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_pointers);
    if needs_descriptor_indexing {
        create_info = create_info.push_next(&mut descriptor_indexing);
    }
    if needs_buffer_device_address {
        create_info = create_info.push_next(&mut buffer_device_address);
    }
    if needs_timeline {
        create_info = create_info.push_next(&mut timeline_semaphore);
    }
    if needs_mesh_shader {
        create_info = create_info.push_next(&mut mesh_shader);
    }

    // SAFETY: `create_info` 引用的队列信息、扩展名与特性结构体都在本函数内保持存活。
    let device = unsafe {
        instance
            .handle()
            .create_device(physical_device, &create_info, None)
    }
    .map_err(|result| {
        RhiError::InstanceCreationFailed(format!("vkCreateDevice 失败: {result:?}"))
    })?;
    // SAFETY: 队列族与索引都来自本设备的查询结果，且已按同一族创建。
    let graphics_queue = unsafe { device.get_device_queue(request.queues.graphics_compute, 0) };
    let transfer_queue = unsafe { device.get_device_queue(request.queues.transfer, 0) };

    Ok(LogicalDevice {
        device,
        graphics_queue,
        transfer_queue,
        extensions: extension_names
            .iter()
            .map(|name| name.to_string_lossy().into_owned())
            .collect(),
    })
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
            buffer_device_address: false,
            timeline_semaphore: false,
            vram_bytes: 8 << 30,
        }
    }

    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            ..Default::default()
        }
    }

    #[test]
    fn missing_requirements_are_reported_not_silently_ignored() {
        let info = fake_info(REQUIRED_API_VERSION);
        let missing = info.missing_requirements(&FeatureRequirements::desktop_full());
        assert_eq!(
            missing,
            vec![
                "VK_EXT_mesh_shader",
                "descriptor_indexing",
                "buffer_device_address",
                "timeline_semaphore"
            ]
        );
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

    #[test]
    fn queue_selection_prefers_a_combined_graphics_compute_family() {
        let families = [
            family(vk::QueueFlags::TRANSFER, 1),
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];
        let queues = select_queue_families(&families, None).expect("应能选出队列族");
        assert_eq!(queues.graphics_compute, 1);
        assert_eq!(queues.transfer, 0);
        assert!(queues.transfer_is_separate);
    }

    #[test]
    fn queue_selection_reuses_graphics_family_without_a_transfer_only_family() {
        let families = [family(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            4,
        )];
        let queues = select_queue_families(&families, None).expect("应能选出队列族");
        assert_eq!(queues.graphics_compute, 0);
        assert_eq!(queues.transfer, 0);
        assert!(!queues.transfer_is_separate);
    }

    #[test]
    fn queue_selection_rejects_devices_without_graphics() {
        let families = [family(vk::QueueFlags::COMPUTE, 1)];
        assert!(select_queue_families(&families, None).is_none());
    }

    #[test]
    fn queue_selection_rejects_families_that_cannot_present() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];
        // 只有 1 号族支持呈现。
        assert!(select_queue_families(&families, Some(&[false, false])).is_none());
        let queues =
            select_queue_families(&families, Some(&[false, true])).expect("应选可呈现的族");
        assert_eq!(queues.graphics_compute, 1);
    }

    #[test]
    fn candidate_scoring_prefers_discrete_over_integrated() {
        let discrete = DeviceCandidate {
            index: 0,
            info: fake_info(REQUIRED_API_VERSION),
            queues: QueueFamilies::default(),
        };
        let mut integrated_info = fake_info(REQUIRED_API_VERSION);
        integrated_info.device_type = vk::PhysicalDeviceType::INTEGRATED_GPU;
        integrated_info.vram_bytes = 32 << 30;
        let integrated = DeviceCandidate {
            index: 1,
            info: integrated_info,
            queues: QueueFamilies::default(),
        };
        assert!(discrete.score() > integrated.score());
        assert_eq!(
            select_best_candidate(
                &[integrated.clone(), discrete],
                &FeatureRequirements::minimum()
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn candidate_selection_reports_the_closest_device() {
        let candidate = DeviceCandidate {
            index: 0,
            info: fake_info(REQUIRED_API_VERSION),
            queues: QueueFamilies::default(),
        };
        let error = select_best_candidate(&[candidate], &FeatureRequirements::desktop_full())
            .expect_err("缺特性时不应选出设备");
        assert_eq!(
            error,
            RhiError::NoSuitableDevice {
                missing: vec![
                    "VK_EXT_mesh_shader",
                    "descriptor_indexing",
                    "buffer_device_address",
                    "timeline_semaphore"
                ]
            }
        );
        let empty = select_best_candidate(&[], &FeatureRequirements::minimum())
            .expect_err("没有设备时必须报错");
        assert_eq!(
            empty,
            RhiError::NoSuitableDevice {
                missing: vec!["任何 Vulkan 物理设备"]
            }
        );
    }

    #[test]
    fn candidate_selection_is_deterministic_on_ties() {
        let candidates = [0, 1, 2]
            .iter()
            .map(|index| DeviceCandidate {
                index: *index,
                info: fake_info(REQUIRED_API_VERSION),
                queues: QueueFamilies::default(),
            })
            .collect::<Vec<_>>();
        let first = select_best_candidate(&candidates, &FeatureRequirements::minimum()).unwrap();
        assert_eq!(first, 0, "同分时必须稳定选择枚举顺序靠前者");
        assert_eq!(
            first,
            select_best_candidate(&candidates, &FeatureRequirements::minimum()).unwrap()
        );
    }
}
