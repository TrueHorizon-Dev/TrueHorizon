//! 显存分配：按用途分类，屏蔽 `gpu-allocator` 细节。
//!
//! 分类策略：**GpuOnly 走设备本地堆；上传/回读走宿主可见堆并做持久映射**。
//! 真正的 `gpu_allocator::vulkan::Allocator` 由 [`GpuAllocator`] 持有，
//! 它会在设备本地堆与宿主堆之间做子分配，避免「一个资源一次 `vkAllocateMemory`」。

use crate::device::VulkanBackend;
use crate::RhiError;
use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings, MemoryLocation};

/// 分配用途分类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryClass {
    /// 纯 GPU 使用（顶点/索引/实例缓冲、纹理、SDF）。
    GpuOnly,
    /// 每帧从 CPU 写入（统一缓冲、实例上传环形区）。
    Upload,
    /// 从 GPU 读回（遮挡查询、GPU-Driven 剔除结果）。
    Readback,
}

impl MemoryClass {
    /// 映射到 `gpu-allocator` 的内存位置。
    #[inline]
    pub fn location(self) -> MemoryLocation {
        match self {
            Self::GpuOnly => MemoryLocation::GpuOnly,
            Self::Upload => MemoryLocation::CpuToGpu,
            Self::Readback => MemoryLocation::GpuToCpu,
        }
    }

    /// 是否需要持久映射。
    #[inline]
    pub fn wants_persistent_map(self) -> bool {
        matches!(self, Self::Upload | Self::Readback)
    }

    /// 全部类别。
    pub const ALL: [MemoryClass; 3] = [Self::GpuOnly, Self::Upload, Self::Readback];
}

/// 一次分配请求。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationDesc {
    /// 用途分类。
    pub class: MemoryClass,
    /// 字节数。
    pub size: u64,
    /// 对齐要求（`vk::MemoryRequirements::alignment`）。
    pub alignment: u64,
    /// 是否要求专用分配（大纹理、大缓冲建议开启）。
    pub dedicated: bool,
}

impl AllocationDesc {
    /// 构造。
    pub const fn new(class: MemoryClass, size: u64, alignment: u64) -> Self {
        Self {
            class,
            size,
            alignment,
            dedicated: false,
        }
    }

    /// 对齐必须是 2 的幂且非零，否则 Vulkan 驱动会返回未定义行为。
    pub fn is_valid(&self) -> bool {
        self.size > 0 && self.alignment > 0 && self.alignment.is_power_of_two()
    }

    /// 向上对齐后的字节数。
    pub fn aligned_size(&self) -> u64 {
        self.size.div_ceil(self.alignment) * self.alignment
    }

    /// 按本请求合成内存需求（在没有真实资源句柄时用于测试与预分配）。
    pub fn synthetic_requirements(&self) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size: self.aligned_size(),
            alignment: self.alignment,
            memory_type_bits: u32::MAX,
        }
    }
}

/// 分配请求与分配方案是否自洽（纯校验，便于在无 GPU 环境下测试）。
///
/// `dedicated` 请求必须携带真实资源句柄（`vk::Buffer` / `vk::Image`），
/// 否则驱动无法创建专用分配——这类错误必须在这里明确拒绝，而不是悄悄退化成子分配。
fn validate_request(desc: &AllocationDesc, carries_resource: bool) -> Result<(), RhiError> {
    if !desc.is_valid() {
        return Err(RhiError::AllocationFailed(format!(
            "非法分配请求：size={} alignment={}（要求非零且 2 的幂）",
            desc.size, desc.alignment
        )));
    }
    if desc.dedicated && !carries_resource {
        return Err(RhiError::AllocationFailed(
            "dedicated 分配需要真实资源句柄，请改用 allocate_dedicated_buffer".to_string(),
        ));
    }
    Ok(())
}

/// `gpu-allocator` 的显存分配器封装。
///
/// **析构顺序**：本类型持有设备句柄的克隆，必须在 [`VulkanBackend`] 之前销毁，
/// 因此调用方应把它放在比后端更内层的作用域（见 [`VulkanBackend::allocator`] 的文档）。
pub struct GpuAllocator {
    inner: Allocator,
    live_allocations: usize,
    allocated_bytes: u64,
}

impl GpuAllocator {
    /// 为已创建的设备建立分配器。
    pub fn new(backend: &VulkanBackend) -> Result<Self, RhiError> {
        let desc = AllocatorCreateDesc {
            instance: backend.raw_instance().clone(),
            device: backend.raw_device().clone(),
            physical_device: backend.physical_device(),
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: backend.info().buffer_device_address,
            allocation_sizes: AllocationSizes::default(),
        };
        let inner =
            Allocator::new(&desc).map_err(|error| RhiError::AllocationFailed(error.to_string()))?;
        Ok(Self {
            inner,
            live_allocations: 0,
            allocated_bytes: 0,
        })
    }

    /// 走子分配（`GpuAllocatorManaged`）的常规分配。
    ///
    /// `requirements` 通常来自 `vkGetBufferMemoryRequirements` / `vkGetImageMemoryRequirements`；
    /// 没有真实资源时可用 [`AllocationDesc::synthetic_requirements`]。
    pub fn allocate(
        &mut self,
        desc: &AllocationDesc,
        requirements: vk::MemoryRequirements,
    ) -> Result<Allocation, RhiError> {
        validate_request(desc, false)?;
        self.allocate_with(desc, requirements, AllocationScheme::GpuAllocatorManaged)
    }

    /// 专用分配：一个资源独占一块内存（大缓冲/大纹理，或驱动要求专用）。
    pub fn allocate_dedicated_buffer(
        &mut self,
        desc: &AllocationDesc,
        buffer: vk::Buffer,
        requirements: vk::MemoryRequirements,
    ) -> Result<Allocation, RhiError> {
        validate_request(desc, true)?;
        self.allocate_with(
            desc,
            requirements,
            AllocationScheme::DedicatedBuffer(buffer),
        )
    }

    /// 专用分配（图像版本）。
    pub fn allocate_dedicated_image(
        &mut self,
        desc: &AllocationDesc,
        image: vk::Image,
        requirements: vk::MemoryRequirements,
    ) -> Result<Allocation, RhiError> {
        validate_request(desc, true)?;
        self.allocate_with(desc, requirements, AllocationScheme::DedicatedImage(image))
    }

    /// 实际调用 `gpu-allocator` 并记账。
    fn allocate_with(
        &mut self,
        desc: &AllocationDesc,
        requirements: vk::MemoryRequirements,
        allocation_scheme: AllocationScheme,
    ) -> Result<Allocation, RhiError> {
        let allocation = self
            .inner
            .allocate(&AllocationCreateDesc {
                name: "truehorizon",
                requirements,
                location: desc.class.location(),
                // 缓冲与线性图像都按线性布局分配；tiling 资源由调用方给出真实需求。
                linear: true,
                allocation_scheme,
            })
            .map_err(|error| RhiError::AllocationFailed(error.to_string()))?;
        self.live_allocations += 1;
        self.allocated_bytes += allocation.size();
        Ok(allocation)
    }

    /// 归还分配。
    pub fn free(&mut self, allocation: Allocation) -> Result<(), RhiError> {
        let size = allocation.size();
        self.inner
            .free(allocation)
            .map_err(|error| RhiError::AllocationFailed(error.to_string()))?;
        self.live_allocations = self.live_allocations.saturating_sub(1);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
        Ok(())
    }

    /// 当前存活的分配数量（用于泄漏自检与诊断输出）。
    pub fn live_allocations(&self) -> usize {
        self.live_allocations
    }

    /// 当前存活的分配总字节数。
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// 底层分配器（`render-pipeline` 需要做块级/线性分配时使用）。
    pub fn inner(&mut self) -> &mut Allocator {
        &mut self.inner
    }
}

impl std::fmt::Debug for GpuAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAllocator")
            .field("live_allocations", &self.live_allocations)
            .field("allocated_bytes", &self.allocated_bytes)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeatureRequirements;

    #[test]
    fn class_maps_to_expected_memory_location() {
        assert_eq!(MemoryClass::GpuOnly.location(), MemoryLocation::GpuOnly);
        assert_eq!(MemoryClass::Upload.location(), MemoryLocation::CpuToGpu);
        assert_eq!(MemoryClass::Readback.location(), MemoryLocation::GpuToCpu);
        assert!(MemoryClass::Upload.wants_persistent_map());
        assert!(!MemoryClass::GpuOnly.wants_persistent_map());
    }

    #[test]
    fn validation_rejects_non_power_of_two_alignment() {
        assert!(AllocationDesc::new(MemoryClass::GpuOnly, 1024, 256).is_valid());
        assert!(!AllocationDesc::new(MemoryClass::GpuOnly, 1024, 300).is_valid());
        assert!(!AllocationDesc::new(MemoryClass::GpuOnly, 0, 256).is_valid());
        assert!(!AllocationDesc::new(MemoryClass::GpuOnly, 1024, 0).is_valid());
    }

    #[test]
    fn aligned_size_rounds_up() {
        let desc = AllocationDesc::new(MemoryClass::GpuOnly, 1000, 256);
        assert_eq!(desc.aligned_size(), 1024);
        assert_eq!(
            AllocationDesc::new(MemoryClass::GpuOnly, 1024, 256).aligned_size(),
            1024
        );
    }

    #[test]
    fn synthetic_requirements_are_aligned_and_fully_masked() {
        let desc = AllocationDesc::new(MemoryClass::Upload, 1000, 256);
        let requirements = desc.synthetic_requirements();
        assert_eq!(requirements.size, 1024);
        assert_eq!(requirements.alignment, 256);
        assert_eq!(requirements.memory_type_bits, u32::MAX);
    }

    #[test]
    fn dedicated_requests_require_a_real_resource_handle() {
        let mut desc = AllocationDesc::new(MemoryClass::GpuOnly, 4096, 256);
        assert!(validate_request(&desc, false).is_ok());
        desc.dedicated = true;
        assert!(matches!(
            validate_request(&desc, false),
            Err(RhiError::AllocationFailed(_))
        ));
        assert!(validate_request(&desc, true).is_ok());

        let broken = AllocationDesc::new(MemoryClass::GpuOnly, 4096, 300);
        assert!(matches!(
            validate_request(&broken, false),
            Err(RhiError::AllocationFailed(_))
        ));
    }

    /// 有 GPU 的机器上真实走一遍分配/归还；无 GPU 时静默跳过（CI 无 Vulkan 也必须是绿的）。
    #[test]
    fn real_allocation_round_trip_when_a_device_exists() {
        let Ok(backend) = VulkanBackend::new(&FeatureRequirements::minimum()) else {
            return;
        };
        let mut allocator = backend.allocator().expect("有设备时必须能建立分配器");
        let desc = AllocationDesc::new(MemoryClass::Upload, 4096, 256);
        let allocation = allocator
            .allocate(&desc, desc.synthetic_requirements())
            .expect("分配 4 KiB 上传缓冲");
        assert!(allocation.size() >= desc.size);
        assert_eq!(allocator.live_allocations(), 1);
        assert!(allocator.allocated_bytes() >= desc.size);
        allocator.free(allocation).expect("归还分配");
        assert_eq!(allocator.live_allocations(), 0);
        assert_eq!(allocator.allocated_bytes(), 0);
    }
}
