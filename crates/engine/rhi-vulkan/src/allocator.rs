//! 显存分配：按用途分类，屏蔽 `gpu-allocator` 细节。

use gpu_allocator::MemoryLocation;

/// 分配用途分类。策略：**GpuOnly 走设备本地堆并配 `DEVICE_LOCAL | SAMPLED`；
/// 上传/回读走宿主可见堆并做持久映射**。
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
    /// 是否允许专用分配（大纹理、大缓冲建议开启）。
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
