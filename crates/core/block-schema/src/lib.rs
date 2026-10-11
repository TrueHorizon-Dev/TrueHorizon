//! # block-schema —— 方块规范
//!
//! 只描述「方块是什么」，不描述「世界长什么样」：
//!
//! * [`BlockId`]：`u16` 稳定 ID，可零拷贝上 GPU、进网络包、落盘；
//! * [`BlockDef`]：材质、渲染分类与物理属性；
//! * `BlockRegistry`：ID → 定义的只读查表，运行期不可变（新增方块必须追加 ID，禁止重排）。
//!
//! ID 契约：`0` 恒为空气（[`BlockId::AIR`]），其余 ID 一经发布永不改变含义——
//! 存档与网络包的长久兼容依赖这一点。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod material;

use serde::{Deserialize, Serialize};

pub use material::{BlockPhysics, Material, RenderClass};

/// 方块 ID 的位宽类型。
pub type BlockIdRepr = u16;

/// 方块 ID。
///
/// `repr(transparent)` + `Pod` 使其可以直接作为 GPU 顶点属性或缓冲内容使用。
#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Debug,
    Default,
    Serialize,
    Deserialize,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
pub struct BlockId(pub BlockIdRepr);

impl BlockId {
    /// 空气。永远是 `0`，且永远是「非实体、不渲染」。
    pub const AIR: Self = Self(0);

    /// 保留给「未定义/损坏数据」的 ID，渲染为醒目错误材质。
    pub const UNKNOWN: Self = Self(BlockIdRepr::MAX);

    /// 可用的稳定 ID 数量（不含 [`Self::UNKNOWN`]）。
    pub const CAPACITY: usize = BlockIdRepr::MAX as usize;

    /// 构造。
    #[inline]
    pub const fn new(id: BlockIdRepr) -> Self {
        Self(id)
    }

    /// 是否为空气。
    #[inline]
    pub const fn is_air(self) -> bool {
        self.0 == Self::AIR.0
    }

    /// 裸 ID，用于落盘与网络编码。
    #[inline]
    pub const fn raw(self) -> BlockIdRepr {
        self.0
    }
}

/// 方块定义。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockDef {
    /// 稳定 ID。
    pub id: BlockId,
    /// 日志/UI 名称（`'static`，注册期一次性确定）。
    pub name: &'static str,
    /// 材质与渲染分类。
    pub material: Material,
    /// 物理属性。
    pub physics: BlockPhysics,
    /// 渲染分类。
    pub render: RenderClass,
}

impl BlockDef {
    /// 是否为空气。
    #[inline]
    pub const fn is_air(&self) -> bool {
        self.id.is_air()
    }

    /// 是否参与碰撞与光照遮挡。
    #[inline]
    pub const fn is_solid(&self) -> bool {
        self.physics.solid
    }
}

/// 方块注册表：`BlockId` → [`BlockDef`] 的只读查表。
///
/// 构建期自由（可从磁盘/脚本加载），运行期冻结：所有查询都是无分支的数组下标。
#[derive(Clone, Debug)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
}

impl BlockRegistry {
    /// 创建只含空气的注册表。
    pub fn new() -> Self {
        let air = BlockDef {
            id: BlockId::AIR,
            name: "air",
            material: Material::Air,
            physics: BlockPhysics::AIR,
            render: RenderClass::Invisible,
        };
        Self { defs: vec![air] }
    }

    /// 追加一个定义。ID 由调用方指定，且必须等于当前长度以保证查表无空洞。
    ///
    /// 返回 `Err` 表示 ID 非法（空洞、重复或越界）——注册期错误必须立刻暴露。
    pub fn push(&mut self, def: BlockDef) -> Result<(), RegistryError> {
        let expected = self.defs.len();
        let actual = def.id.raw() as usize;
        if actual == BlockId::UNKNOWN.raw() as usize {
            return Err(RegistryError::ReservedId(def.id));
        }
        if actual < expected {
            return Err(RegistryError::DuplicateId(def.id));
        }
        if actual > expected {
            return Err(RegistryError::IdGap {
                expected: expected as BlockIdRepr,
                actual: def.id.raw(),
            });
        }
        self.defs.push(def);
        Ok(())
    }

    /// 按 ID 查询。未知 ID（含 [`BlockId::UNKNOWN`]）返回 `None`。
    #[inline]
    pub fn get(&self, id: BlockId) -> Option<&BlockDef> {
        self.defs.get(id.raw() as usize)
    }

    /// 按 ID 查询，缺失时回退到空气定义。
    #[inline]
    pub fn get_or_air(&self, id: BlockId) -> &BlockDef {
        self.get(id).unwrap_or(&self.defs[0])
    }

    /// 已注册方块数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// 是否只注册了空气。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.defs.len() <= 1
    }

    /// 遍历全部定义。
    pub fn iter(&self) -> impl Iterator<Item = &BlockDef> {
        self.defs.iter()
    }

    /// 判断某个 ID 是否为实体方块；未知 ID 视为空气。
    #[inline]
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).map(|d| d.physics.solid).unwrap_or(false)
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// 注册期错误。所有变体都意味着**数据布局错误**，必须让构建流程失败而不是静默降级。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// ID 与已有定义重复。
    DuplicateId(BlockId),
    /// ID 与当前长度之间有空洞，会使查表出现未定义项。
    IdGap {
        /// 期望的下一个 ID。
        expected: BlockIdRepr,
        /// 实际传入的 ID。
        actual: BlockIdRepr,
    },
    /// 使用了保留 ID。
    ReservedId(BlockId),
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "方块 ID {} 重复注册", id.raw()),
            Self::IdGap { expected, actual } => {
                write!(f, "方块 ID 出现空洞：期望 {expected}，实际 {actual}")
            }
            Self::ReservedId(id) => write!(f, "方块 ID {} 为保留值", id.raw()),
        }
    }
}

impl std::error::Error for RegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_def(id: BlockIdRepr, solid: bool) -> BlockDef {
        BlockDef {
            id: BlockId::new(id),
            name: "test",
            material: Material::Rock,
            physics: BlockPhysics {
                solid,
                ..BlockPhysics::AIR
            },
            render: RenderClass::Opaque,
        }
    }

    #[test]
    fn registry_requires_contiguous_ids() {
        let mut registry = BlockRegistry::new();
        assert!(registry.push(test_def(1, true)).is_ok());
        assert_eq!(
            registry.push(test_def(3, true)),
            Err(RegistryError::IdGap {
                expected: 2,
                actual: 3
            })
        );
        assert_eq!(
            registry.push(test_def(1, true)),
            Err(RegistryError::DuplicateId(BlockId::new(1)))
        );
        assert_eq!(
            registry.push(test_def(BlockIdRepr::MAX, true)),
            Err(RegistryError::ReservedId(BlockId::UNKNOWN))
        );
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn air_is_never_solid() {
        let mut registry = BlockRegistry::new();
        registry.push(test_def(1, true)).unwrap();
        assert!(!registry.is_solid(BlockId::AIR));
        assert!(registry.is_solid(BlockId::new(1)));
        assert!(
            !registry.is_solid(BlockId::new(42)),
            "未知 ID 必须按空气处理"
        );
        assert!(registry.get_or_air(BlockId::new(42)).is_air());
    }

    #[test]
    fn block_id_is_pod() {
        let ids = [BlockId::AIR, BlockId::new(7)];
        let bytes: &[u8] = bytemuck::cast_slice(&ids);
        assert_eq!(bytes.len(), 4, "u16 且无填充");
    }
}
