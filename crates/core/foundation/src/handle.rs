//! 世代索引句柄。

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// 指向 [`crate::Arena`] 中某个槽位的强引用。
///
/// 句柄只携带 `(index, generation)` 两个 `u32`，因此：
///
/// * 永远 `Copy`，与 `T` 的实际大小无关；
/// * 槽位被回收时世代号自增，旧句柄随即失效（`Arena::get` 返回 `None`），
///   从根本上消除 32 位索引复用导致的 ABA 问题；
/// * 标记类型为 `fn() -> T`，句柄不拥有 `T`，也不会意外继承 `T` 的 `Send`/`Sync`。
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    /// 无效句柄哨兵。可用作 `Option<Handle<T>>` 之外的零开销占位。
    pub const DANGLING: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
        _marker: PhantomData,
    };

    /// 由裸索引与世代号构造（内部使用；外部请经由 `Arena::insert` 获得句柄）。
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// 槽位索引。
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// 世代号。
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// 是否为哨兵句柄。
    #[inline]
    pub const fn is_dangling(self) -> bool {
        self.index == u32::MAX && self.generation == u32::MAX
    }

    /// 拆成可直接落盘 / 上线的裸整型对。
    #[inline]
    pub const fn to_raw(self) -> (u32, u32) {
        (self.index, self.generation)
    }

    /// 从裸整型对还原（例如从快照或网络包中读出）。
    #[inline]
    pub const fn from_raw(raw: (u32, u32)) -> Self {
        Self::new(raw.0, raw.1)
    }
}

// 手写 trait 实现：派生宏会额外要求 `T: Clone + Hash + ...`，而句柄与 `T` 无关。
impl<T> Clone for Handle<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> PartialOrd for Handle<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Handle<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index, self.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangling_is_detectable() {
        assert!(Handle::<u8>::DANGLING.is_dangling());
        assert!(!Handle::<u8>::new(0, 0).is_dangling());
    }

    #[test]
    fn raw_round_trip() {
        let handle = Handle::<u32>::new(7, 3);
        assert_eq!(Handle::<u32>::from_raw(handle.to_raw()), handle);
    }
}
