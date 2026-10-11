//! # persistence —— 存档与崩溃恢复
//!
//! 存档策略：**只存差量**。
//!
//! ```text
//! 程序化世界（种子）  +  稀疏差量表（玩家改动）  =  完整世界
//!      ↑ 无需存储              ↑ 与行星大小无关
//! ```
//!
//! * [`delta`]：`ChunkDelta`（体素下标 + 方块，4 字节/条）与 `DeltaTable` 聚合。
//! * [`wal`]：长度前缀的写时复制日志，支持撕裂写检测与部分恢复。
//!
//! 崩溃恢复流程：
//!
//! 1. 用种子重新生成 Chunk 基准；
//! 2. 顺序重放 WAL 中 `base_version` 匹配的差量（不匹配的记录直接跳过，见
//!    `DeltaTable::apply_all`）；
//! 3. 遇到 [`PersistError::Truncated`] 即停止——尾帧残缺是预期情况，不是故障。
//!
//! 尚未实现（路线图）：快照 + WAL 的压实（compaction）、每 Chunk 的增量快照、
//! 校验和（CRC32C）与多副本原子切换。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod delta;
pub mod wal;

pub use delta::{ChunkDelta, DeltaTable, VoxelDelta, MAX_DELTA_EDITS};
pub use wal::{WalReader, WalRecord, WalWriter, WAL_MAGIC};

/// 持久化层错误。
#[derive(Debug)]
pub enum PersistError {
    /// 底层 I/O 失败。
    Io(std::io::Error),
    /// 文件头魔数不匹配（不是 WAL 文件，或版本不兼容）。
    BadMagic {
        /// 实际读到的魔数。
        found: [u8; 8],
    },
    /// 记录尾部被截断（进程被杀）。**可恢复**：丢弃该帧即可。
    Truncated,
    /// 记录类型未知（文件比本程序更新）。
    UnknownRecordKind(u8),
    /// 负载结构与声明不符。
    Malformed(String),
}

impl core::fmt::Display for PersistError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O 失败: {error}"),
            Self::BadMagic { found } => write!(f, "WAL 魔数不匹配: {found:?}"),
            Self::Truncated => write!(f, "WAL 记录被截断（可安全丢弃该帧）"),
            Self::UnknownRecordKind(kind) => write!(f, "未知的 WAL 记录类型: {kind}"),
            Self::Malformed(message) => write!(f, "WAL 负载损坏: {message}"),
        }
    }
}

impl std::error::Error for PersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl PartialEq for PersistError {
    /// 逐类比较。`io::Error` 本身不是 `PartialEq`，因此按「错误类别 + 文本」比较——
    /// 对测试与日志去重而言这已经足够，且不会掩盖不同类别的错误。
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Io(left), Self::Io(right)) => {
                left.kind() == right.kind() && left.to_string() == right.to_string()
            }
            (Self::BadMagic { found: left }, Self::BadMagic { found: right }) => left == right,
            (Self::Truncated, Self::Truncated) => true,
            (Self::UnknownRecordKind(left), Self::UnknownRecordKind(right)) => left == right,
            (Self::Malformed(left), Self::Malformed(right)) => left == right,
            _ => false,
        }
    }
}

/// 崩溃恢复的可恢复性判断：只有 [`PersistError::Truncated`] 与
/// [`PersistError::UnknownRecordKind`] 允许「就此停下」（前者是预期，后者是版本更新）。
pub fn is_recoverable(error: &PersistError) -> bool {
    matches!(
        error,
        PersistError::Truncated | PersistError::UnknownRecordKind(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_tail_problems_are_recoverable() {
        assert!(is_recoverable(&PersistError::Truncated));
        assert!(is_recoverable(&PersistError::UnknownRecordKind(7)));
        assert!(!is_recoverable(&PersistError::BadMagic { found: [0; 8] }));
        assert!(!is_recoverable(&PersistError::Malformed("x".into())));
    }

    #[test]
    fn io_errors_convert_and_keep_source() {
        let error: PersistError = std::io::Error::other("boom").into();
        assert!(matches!(error, PersistError::Io(_)));
        assert!(std::error::Error::source(&error).is_some());
        assert!(error.to_string().contains("boom"));
    }
}
