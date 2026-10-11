//! 写时复制日志（Write-Ahead Log）。
//!
//! 设计取舍：**帧头手写、负载分层**。
//!
//! * 帧头（`kind | len`）手写：定长字段 + 显式小端，天然支持「跳过整条记录」与部分恢复；
//!   长度前缀使**撕裂写**（进程被杀时写了一半）可被检测为截断错误，而不是解析出垃圾；
//! * 差量负载手写：4 字节/条的紧凑布局，格式能被 10 年后的代码读懂，
//!   且 `world-tool diagnose` 不链接引擎也能解析；
//! * 快照负载用 `rkyv` 0.8 归档：整块 Chunk 是「读多写少」的数据，
//!   归档支持零拷贝读取与字节级校验（见 [`crate::snapshot`]）。
//!
//! 帧格式：
//!
//! ```text
//! 文件头: "THWAL001" (8 字节)
//! 记录:   kind: u8 | payload_len: u32 (小端) | payload[payload_len]
//! ```
//!
//! 记录负载：
//!
//! * `Delta`：`coord(3×i64) | base_version: u32 | count: u32 | count × (index: u16, block: u16)`
//! * `Checkpoint`：`tick: u64`
//! * `Snapshot`：`rkyv` 归档字节（`ChunkSnapshot`）

use std::io::{Read, Write};

use world_contracts::ChunkCoord;

use crate::delta::{ChunkDelta, VoxelDelta};
use crate::snapshot::ChunkSnapshot;
use crate::PersistError;

/// WAL 文件魔数与版本号（改格式必须改这里）。
pub const WAL_MAGIC: [u8; 8] = *b"THWAL001";

/// 单条记录的最大负载，防止损坏的长度前缀导致巨额分配。
pub const MAX_RECORD_BYTES: u32 = 16 * 1024 * 1024;

const KIND_DELTA: u8 = 1;
const KIND_CHECKPOINT: u8 = 2;
const KIND_SNAPSHOT: u8 = 3;

/// WAL 记录。
#[derive(Clone, Debug, PartialEq)]
pub enum WalRecord {
    /// 一个 Chunk 的差量。
    Delta(ChunkDelta),
    /// 事务边界：恢复时只重放到最后一个 checkpoint 之后的记录也可安全丢弃。
    Checkpoint {
        /// 该 checkpoint 覆盖到的 tick。
        tick: u64,
    },
    /// 整块 Chunk 快照（`rkyv` 归档）：压实后的稠密基准。
    Snapshot(ChunkSnapshot),
}

impl WalRecord {
    /// 编码为负载字节。
    pub fn encode(&self) -> Result<Vec<u8>, PersistError> {
        Ok(match self {
            Self::Delta(delta) => {
                let mut out = Vec::with_capacity(24 + 8 + delta.len() * 4);
                out.extend_from_slice(&delta.coord.x.to_le_bytes());
                out.extend_from_slice(&delta.coord.y.to_le_bytes());
                out.extend_from_slice(&delta.coord.z.to_le_bytes());
                out.extend_from_slice(&delta.base_version.to_le_bytes());
                out.extend_from_slice(&(delta.len() as u32).to_le_bytes());
                for edit in &delta.edits {
                    out.extend_from_slice(&edit.index.to_le_bytes());
                    out.extend_from_slice(&edit.block.to_le_bytes());
                }
                out
            },
            Self::Checkpoint { tick } => tick.to_le_bytes().to_vec(),
            Self::Snapshot(snapshot) => snapshot.archive()?.to_vec(),
        })
    }

    /// 记录类型标签。
    pub fn kind(&self) -> u8 {
        match self {
            Self::Delta(_) => KIND_DELTA,
            Self::Checkpoint { .. } => KIND_CHECKPOINT,
            Self::Snapshot(_) => KIND_SNAPSHOT,
        }
    }

    /// 由类型标签与负载解码。
    pub fn decode(kind: u8, payload: &[u8]) -> Result<Self, PersistError> {
        match kind {
            KIND_DELTA => {
                if payload.len() < 32 {
                    return Err(PersistError::Malformed("Delta 负载长度不足".into()));
                }
                let read_i64 = |offset: usize| -> i64 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&payload[offset..offset + 8]);
                    i64::from_le_bytes(bytes)
                };
                let read_u32 = |offset: usize| -> u32 {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&payload[offset..offset + 4]);
                    u32::from_le_bytes(bytes)
                };
                let coord = ChunkCoord::new(read_i64(0), read_i64(8), read_i64(16));
                let base_version = read_u32(24);
                let count = read_u32(28) as usize;
                let expected = 32 + count * 4;
                if payload.len() != expected {
                    return Err(PersistError::Malformed(format!(
                        "Delta 负载长度 {} 与声明条数 {count} 不符（应为 {expected}）",
                        payload.len()
                    )));
                }
                let mut edits = Vec::with_capacity(count);
                for index in 0..count {
                    let offset = 32 + index * 4;
                    edits.push(VoxelDelta {
                        index: u16::from_le_bytes([payload[offset], payload[offset + 1]]),
                        block: u16::from_le_bytes([payload[offset + 2], payload[offset + 3]]),
                    });
                }
                Ok(Self::Delta(ChunkDelta {
                    coord,
                    base_version,
                    edits,
                }))
            },
            KIND_CHECKPOINT => {
                if payload.len() != 8 {
                    return Err(PersistError::Malformed(
                        "Checkpoint 负载必须为 8 字节".into(),
                    ));
                }
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(payload);
                Ok(Self::Checkpoint {
                    tick: u64::from_le_bytes(bytes),
                })
            },
            KIND_SNAPSHOT => ChunkSnapshot::from_archive(payload).map(Self::Snapshot),
            other => Err(PersistError::UnknownRecordKind(other)),
        }
    }
}

/// WAL 写入端。
///
/// 每次 [`WalWriter::append`] 只做「序列化 + 一次 `write_all`」，
/// 调用方按批次 [`WalWriter::flush`]——批量落盘是 WAL 吞吐的关键。
pub struct WalWriter<W: Write> {
    inner: W,
    records: u64,
    payload_bytes: u64,
}

impl<W: Write> WalWriter<W> {
    /// 创建写入端并写入文件头。
    pub fn new(mut inner: W) -> Result<Self, PersistError> {
        inner.write_all(&WAL_MAGIC)?;
        Ok(Self {
            inner,
            records: 0,
            payload_bytes: 0,
        })
    }

    /// 追加一条记录（含长度前缀）。
    pub fn append(&mut self, record: &WalRecord) -> Result<(), PersistError> {
        let payload = record.encode()?;
        if payload.len() as u64 > u64::from(MAX_RECORD_BYTES) {
            return Err(PersistError::Malformed(format!(
                "记录负载 {} 字节超出上限 {MAX_RECORD_BYTES}",
                payload.len()
            )));
        }
        self.inner.write_all(&[record.kind()])?;
        self.inner
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.inner.write_all(&payload)?;
        self.records += 1;
        self.payload_bytes += payload.len() as u64;
        Ok(())
    }

    /// 冲刷到底层。
    pub fn flush(&mut self) -> Result<(), PersistError> {
        self.inner.flush()?;
        Ok(())
    }

    /// 已写入的记录数。
    #[inline]
    pub fn records(&self) -> u64 {
        self.records
    }

    /// 已写入的负载字节数（不含帧头）。
    #[inline]
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// 取回底层写入器。
    pub fn into_inner(self) -> W {
        self.inner
    }
}

/// WAL 读取端。
pub struct WalReader<R: Read> {
    inner: R,
    records_read: u64,
}

impl<R: Read> core::fmt::Debug for WalReader<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WalReader")
            .field("records_read", &self.records_read)
            .finish()
    }
}

impl<W: Write> core::fmt::Debug for WalWriter<W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WalWriter")
            .field("records", &self.records)
            .field("payload_bytes", &self.payload_bytes)
            .finish()
    }
}

impl<R: Read> WalReader<R> {
    /// 创建读取端并校验文件头。
    pub fn new(mut inner: R) -> Result<Self, PersistError> {
        let mut magic = [0u8; 8];
        inner.read_exact(&mut magic)?;
        if magic != WAL_MAGIC {
            return Err(PersistError::BadMagic { found: magic });
        }
        Ok(Self {
            inner,
            records_read: 0,
        })
    }

    /// 读下一条记录；到达文件末尾返回 `Ok(None)`。
    ///
    /// 撕裂写会表现为 `read_exact` 的 `UnexpectedEof`，即
    /// [`PersistError::Truncated`]——**这是可恢复的**：丢弃残缺的尾帧即可。
    pub fn next_record(&mut self) -> Result<Option<WalRecord>, PersistError> {
        let mut kind = [0u8; 1];
        match self.inner.read_exact(&mut kind) {
            Ok(()) => {},
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(PersistError::Io(error)),
        }
        let mut length = [0u8; 4];
        self.inner.read_exact(&mut length).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                PersistError::Truncated
            } else {
                PersistError::Io(error)
            }
        })?;
        let length = u32::from_le_bytes(length);
        if length > MAX_RECORD_BYTES {
            return Err(PersistError::Malformed(format!(
                "记录长度 {length} 超出上限 {MAX_RECORD_BYTES}"
            )));
        }
        let mut payload = vec![0u8; length as usize];
        self.inner.read_exact(&mut payload).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                PersistError::Truncated
            } else {
                PersistError::Io(error)
            }
        })?;
        self.records_read += 1;
        WalRecord::decode(kind[0], &payload).map(Some)
    }

    /// 已读取的记录数。
    #[inline]
    pub fn records_read(&self) -> u64 {
        self.records_read
    }

    /// 读到全部记录（遇到截断的尾帧视为正常结束）。
    pub fn read_all(mut self) -> Result<Vec<WalRecord>, PersistError> {
        let mut records = Vec::new();
        loop {
            match self.next_record() {
                Ok(Some(record)) => records.push(record),
                Ok(None) => return Ok(records),
                Err(PersistError::Truncated) => return Ok(records),
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::ChunkSnapshot;
    use block_schema::BlockId;
    use std::io::Cursor;
    use storage::Chunk;

    fn sample_delta() -> ChunkDelta {
        let mut delta = ChunkDelta::new(ChunkCoord::new(-3, 7, 0), 42);
        delta.record(1, 2, 3, BlockId::new(5)).unwrap();
        delta.record(31, 0, 31, BlockId::new(9)).unwrap();
        delta
    }

    #[test]
    fn delta_round_trip_through_wal() {
        let mut buffer = Vec::new();
        {
            let mut writer = WalWriter::new(&mut buffer).unwrap();
            writer.append(&WalRecord::Delta(sample_delta())).unwrap();
            writer
                .append(&WalRecord::Checkpoint { tick: 1234 })
                .unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.records(), 2);
            assert!(writer.payload_bytes() > 0);
        }

        let reader = WalReader::new(Cursor::new(&buffer)).unwrap();
        let records = reader.read_all().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], WalRecord::Delta(sample_delta()));
        assert_eq!(records[1], WalRecord::Checkpoint { tick: 1234 });
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = WAL_MAGIC.to_vec();
        bytes[0] = b'X';
        let error = WalReader::new(Cursor::new(bytes)).unwrap_err();
        assert!(matches!(error, PersistError::BadMagic { .. }));
    }

    #[test]
    fn truncated_tail_is_recoverable() {
        let mut buffer = Vec::new();
        {
            let mut writer = WalWriter::new(&mut buffer).unwrap();
            writer.append(&WalRecord::Checkpoint { tick: 1 }).unwrap();
            writer.append(&WalRecord::Delta(sample_delta())).unwrap();
            writer.flush().unwrap();
        }
        // 砍掉最后 6 个字节：模拟进程在写入途中被杀。
        buffer.truncate(buffer.len() - 6);

        let mut reader = WalReader::new(Cursor::new(&buffer)).unwrap();
        let first = reader.next_record().unwrap();
        assert_eq!(first, Some(WalRecord::Checkpoint { tick: 1 }));
        assert_eq!(
            reader.next_record().unwrap_err(),
            PersistError::Truncated,
            "残缺尾帧必须报截断而不是解析出垃圾"
        );

        // read_all 把截断当作正常结束，返回已恢复的记录。
        let recovered = WalReader::new(Cursor::new(&buffer))
            .unwrap()
            .read_all()
            .unwrap();
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn unknown_kind_and_bad_length_are_reported() {
        assert_eq!(
            WalRecord::decode(99, &[]).unwrap_err(),
            PersistError::UnknownRecordKind(99)
        );
        // 声明 3 条编辑但只给了 1 条的负载。
        let mut payload = Vec::new();
        payload.extend_from_slice(&0i64.to_le_bytes());
        payload.extend_from_slice(&0i64.to_le_bytes());
        payload.extend_from_slice(&0i64.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&3u32.to_le_bytes());
        payload.extend_from_slice(&[0u8; 4]);
        assert!(matches!(
            WalRecord::decode(KIND_DELTA, &payload).unwrap_err(),
            PersistError::Malformed(_)
        ));
    }

    #[test]
    fn empty_delta_round_trips() {
        let record = WalRecord::Delta(ChunkDelta::new(ChunkCoord::default(), 0));
        let payload = record.encode().unwrap();
        assert_eq!(payload.len(), 32, "空差量只有头部");
        assert_eq!(WalRecord::decode(KIND_DELTA, &payload).unwrap(), record);
    }

    #[test]
    fn snapshot_record_round_trips_through_the_wal() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(4, -1, 2));
        chunk.set(3, 4, 5, BlockId::new(7)).unwrap();
        chunk.set(31, 31, 31, BlockId::new(9)).unwrap();
        let snapshot = ChunkSnapshot::from_chunk(&chunk);

        let mut buffer = Vec::new();
        {
            let mut writer = WalWriter::new(&mut buffer).unwrap();
            writer
                .append(&WalRecord::Snapshot(snapshot.clone()))
                .unwrap();
            writer.flush().unwrap();
        }

        let records = WalReader::new(Cursor::new(&buffer))
            .unwrap()
            .read_all()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0], WalRecord::Snapshot(snapshot.clone()));

        // 解出来的快照必须能重建出逐体素一致的 Chunk。
        let WalRecord::Snapshot(restored) = &records[0] else {
            panic!("记录类型必须是 Snapshot");
        };
        let rebuilt = restored.to_chunk().unwrap();
        assert_eq!(rebuilt.get(3, 4, 5), BlockId::new(7));
        assert_eq!(rebuilt.get(31, 31, 31), BlockId::new(9));
        assert_eq!(rebuilt.non_air_count(), chunk.non_air_count());
    }

    #[test]
    fn a_corrupted_snapshot_frame_is_reported_not_panicking() {
        let mut chunk = Chunk::new_empty(ChunkCoord::default());
        chunk.set(1, 1, 1, BlockId::new(1)).unwrap();
        let snapshot = ChunkSnapshot::from_chunk(&chunk);
        let mut payload = snapshot.archive().unwrap().to_vec();
        // 破坏负载中段：校验必须发现，而不是读出越界数据。
        let length = payload.len();
        for byte in payload.iter_mut().skip(length / 2).take(32) {
            *byte = 0xAB;
        }
        match WalRecord::decode(KIND_SNAPSHOT, &payload) {
            Err(PersistError::Malformed(_)) => {},
            // 极少数改动恰好仍是合法归档：允许解码成功，但绝不允许 panic。
            Ok(WalRecord::Snapshot(_)) => {},
            other => panic!("损坏的快照帧只能报 Malformed 或解出合法快照，实际: {other:?}"),
        }
    }
}
