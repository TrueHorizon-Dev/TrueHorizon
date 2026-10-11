//! 无绑定（bindless）描述符。

use crate::RhiError;

/// 单个 bindless 纹理数组的容量上限。
///
/// 65536 = 该引擎全部方块材质图集页 + 地形高度场 + 调试可视化的总和上限，
/// 同时是「16 位纹理索引」能表达的最大值。
pub const MAX_BINDLESS_TEXTURES: u32 = 65_536;

/// 单个 bindless 采样器数组的容量上限。
pub const MAX_BINDLESS_SAMPLERS: u32 = 64;

/// 一个 bindless 绑定的描述。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindlessLayout {
    /// 描述符集索引。
    pub set: u32,
    /// 绑定号。
    pub binding: u32,
    /// 数组长度。
    pub count: u32,
    /// 是否为可变长度（运行时按剩余空间调整）。
    pub variable_count: bool,
    /// 允许在绑定后更新（bindless 的硬性前提）。
    pub update_after_bind: bool,
    /// 允许部分绑定（只填充前 N 项）。
    pub partially_bound: bool,
}

impl BindlessLayout {
    /// 引擎标准布局：set 0 固定为「bindless 资源表」。
    pub const fn standard() -> Self {
        Self {
            set: 0,
            binding: 0,
            count: MAX_BINDLESS_TEXTURES,
            variable_count: true,
            update_after_bind: true,
            partially_bound: true,
        }
    }

    /// 校验容量与标志位组合是否自洽。
    pub fn validate(&self) -> Result<(), RhiError> {
        if self.count == 0 {
            return Err(RhiError::AllocationFailed(
                "bindless 数组长度不能为 0".into(),
            ));
        }
        if self.variable_count && self.count < MAX_BINDLESS_TEXTURES {
            // 可变长度数组必须声明为最大容量，否则驱动按声明值截断。
            return Err(RhiError::AllocationFailed(
                "可变长度 bindless 数组必须声明最大容量".into(),
            ));
        }
        Ok(())
    }
}

/// GPU 侧的资源索引。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ResourceIndex(pub u32);

impl Default for ResourceIndex {
    /// 默认值取 [`ResourceIndex::INVALID`]（而非 `0`）——`0` 是合法槽位，
    /// 用它当默认值会把「未设置」误判成「指向第一个资源」。
    fn default() -> Self {
        Self::INVALID
    }
}

impl ResourceIndex {
    /// 无效索引。
    pub const INVALID: Self = Self(u32::MAX);

    /// 是否有效。
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 < MAX_BINDLESS_TEXTURES
    }
}

/// 宿主侧的描述符槽位分配器。
///
/// 采用**只增不减 + 分代回收**：槽位一经分配即固定索引（GPU 侧索引必须稳定，
/// 否则需要重建命令缓冲）。容量耗尽时返回 `None`，由调用方决定淘汰策略。
#[derive(Debug)]
pub struct DescriptorArena {
    capacity: u32,
    next_free: u32,
    live: u32,
}

impl DescriptorArena {
    /// 以容量构造。
    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next_free: 0,
            live: 0,
        }
    }

    /// 标准容量的资源表。
    pub const fn standard() -> Self {
        Self::new(MAX_BINDLESS_TEXTURES)
    }

    /// 分配一个槽位。
    pub fn allocate(&mut self) -> Option<ResourceIndex> {
        if self.next_free >= self.capacity {
            return None;
        }
        let index = ResourceIndex(self.next_free);
        self.next_free += 1;
        self.live += 1;
        Some(index)
    }

    /// 已分配槽位数（含已淘汰但未复用的）。
    #[inline]
    pub fn live(&self) -> u32 {
        self.live
    }

    /// 水位（已分配到的最大索引 + 1）。
    #[inline]
    pub fn watermark(&self) -> u32 {
        self.next_free
    }

    /// 剩余可分配数量。
    #[inline]
    pub fn remaining(&self) -> u32 {
        self.capacity - self.next_free
    }

    /// 是否已满。
    #[inline]
    pub fn is_full(&self) -> bool {
        self.next_free >= self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_allocates_monotonic_indices_and_stops_at_capacity() {
        let mut arena = DescriptorArena::new(3);
        assert_eq!(arena.allocate(), Some(ResourceIndex(0)));
        assert_eq!(arena.allocate(), Some(ResourceIndex(1)));
        assert_eq!(arena.allocate(), Some(ResourceIndex(2)));
        assert!(arena.is_full(), "容量用尽后必须报告已满");
        assert_eq!(arena.allocate(), None);
        assert!(arena.is_full());
        assert_eq!(arena.remaining(), 0);
        assert_eq!(arena.live(), 3);
    }

    #[test]
    fn standard_layout_is_consistent_with_capacity() {
        let layout = BindlessLayout::standard();
        assert!(layout.validate().is_ok());
        assert_eq!(layout.count, MAX_BINDLESS_TEXTURES);
        assert!(ResourceIndex(MAX_BINDLESS_TEXTURES - 1).is_valid());
        assert!(!ResourceIndex(MAX_BINDLESS_TEXTURES).is_valid());
        assert!(!ResourceIndex::INVALID.is_valid());
    }

    #[test]
    fn undersized_variable_array_is_rejected() {
        let bad = BindlessLayout {
            count: 16,
            ..BindlessLayout::standard()
        };
        assert!(bad.validate().is_err());
    }
}
