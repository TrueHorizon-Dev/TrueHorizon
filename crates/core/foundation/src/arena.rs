//! 世代索引对象池。

use crate::handle::Handle;

/// 空闲链表末尾标记。
const NULL: u32 = u32::MAX;

/// 单个槽位：占用时持有值，空闲时持有「下一个空闲槽位」指针。
enum Slot<T> {
    Occupied { generation: u32, value: T },
    Vacant { generation: u32, next_free: u32 },
}

/// 由 [`Handle`] 寻址的对象池。
///
/// * 插入与删除均为 O(1)，删除留下的空位由内部空闲链表复用；
/// * 索引是 `u32`，因此单池上限为 `u32::MAX - 1` 个槽位（远超 32³ Chunk 池所需）；
/// * 删除后槽位世代号自增，旧句柄立即失效。
///
/// 并发说明：本结构是**单线程**存储体。面向流式加载的跨线程共享将在此之上叠加分片
/// （每线程一个 Arena + 句柄回收队列），见 `storage::ChunkPool` 的路线图。
pub struct Arena<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    len: usize,
}

impl<T> Arena<T> {
    /// 创建空池。不分配内存。
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: NULL,
            len: 0,
        }
    }

    /// 预分配 `capacity` 个槽位。
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: NULL,
            len: 0,
        }
    }

    /// 存活元素数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 已申请的槽位总数（含空闲槽位）。
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// 插入一个值并返回其句柄。
    pub fn insert(&mut self, value: T) -> Handle<T> {
        if self.free_head != NULL {
            let index = self.free_head as usize;
            // 空闲链表只指向 Vacant 槽位，此处用断言兜住内部不变量。
            let generation = match &self.slots[index] {
                Slot::Vacant { generation, .. } => *generation,
                Slot::Occupied { .. } => unreachable!("Arena 空闲链表指向了已占用槽位"),
            };
            let next_free = match &self.slots[index] {
                Slot::Vacant { next_free, .. } => *next_free,
                Slot::Occupied { .. } => unreachable!(),
            };
            self.slots[index] = Slot::Occupied { generation, value };
            self.free_head = next_free;
            self.len += 1;
            Handle::new(index as u32, generation)
        } else {
            let index = self.slots.len();
            assert!(index < NULL as usize, "Arena 槽位索引溢出 u32");
            self.slots.push(Slot::Occupied {
                generation: 0,
                value,
            });
            self.len += 1;
            Handle::new(index as u32, 0)
        }
    }

    /// 取得不可变引用；句柄失效时返回 `None`。
    #[inline]
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == handle.generation() => {
                Some(value)
            },
            _ => None,
        }
    }

    /// 取得可变引用；句柄失效时返回 `None`。
    #[inline]
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == handle.generation() => {
                Some(value)
            },
            _ => None,
        }
    }

    /// 句柄当前是否有效。
    #[inline]
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// 移除并返回元素；槽位进入空闲链表，旧句柄全部失效。
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let index = handle.index() as usize;
        let hit = matches!(
            self.slots.get(index),
            Some(Slot::Occupied { generation, .. }) if *generation == handle.generation()
        );
        if !hit {
            return None;
        }
        let vacant = Slot::Vacant {
            generation: handle.generation().wrapping_add(1),
            next_free: self.free_head,
        };
        let old = core::mem::replace(&mut self.slots[index], vacant);
        self.free_head = index as u32;
        self.len -= 1;
        match old {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => None,
        }
    }

    /// 遍历全部存活元素及其句柄。
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((Handle::new(i as u32, *generation), value))
                },
                Slot::Vacant { .. } => None,
            })
    }

    /// 遍历全部存活元素的可变引用。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> + '_ {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((Handle::new(i as u32, *generation), value))
                },
                Slot::Vacant { .. } => None,
            })
    }

    /// 清空全部槽位。**所有既有句柄立即失效**，但已分配的容量保留复用。
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = NULL;
        self.len = 0;
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut arena = Arena::new();
        let a = arena.insert("alpha");
        let b = arena.insert("beta");
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(a), Some(&"alpha"));
        assert_eq!(arena.remove(a), Some("alpha"));
        assert_eq!(arena.get(a), None);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(b), Some(&"beta"));
    }

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut arena = Arena::new();
        let old = arena.insert(1u32);
        arena.remove(old);
        let new = arena.insert(2u32);
        assert_eq!(old.index(), new.index(), "空闲槽位应被复用");
        assert_ne!(old.generation(), new.generation(), "世代号必须自增");
        assert_eq!(arena.get(old), None, "旧句柄必须失效");
        assert_eq!(arena.get(new), Some(&2));
    }

    #[test]
    fn clear_invalidates_everything() {
        let mut arena = Arena::with_capacity(4);
        let handle = arena.insert(9u8);
        arena.clear();
        assert!(arena.is_empty());
        assert_eq!(arena.get(handle), None);
    }
}
