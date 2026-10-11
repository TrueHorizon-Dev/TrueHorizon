//! Morton（Z-order）空间索引。
//!
//! Chunk 局部坐标（每轴 5 位）压进 15 位键，宏观四叉树坐标（每轴 21 位）压进 63 位键。
//!
//! 为什么用 Morton 而不是哈希：
//!
//! * **局部性好**：空间相邻的键在数值上也相邻，顺序遍历即近似空间遍历，
//!   对 Chunk 上传顺序、GPU 剔除的早期拒绝率都有直接收益；
//! * **可逆**：`encode ∘ decode = id`，可把一维数组当三维结构用（省一层间接）。

/// 把 21 位以下的整数按位散布到 64 位整数的每第 3 位上。
fn spread3(value: u32) -> u64 {
    let mut x = u64::from(value) & 0x1F_FFFF;
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// 逆操作：从每第 3 位恢复原始整数。
fn compact3(mut x: u64) -> u32 {
    x &= 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x >> 4)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x >> 8)) & 0x001F_0000_FF00_00FF;
    x = (x | (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x | (x >> 32)) & 0x1F_FFFF;
    x as u32
}

/// 编码三维坐标（每轴 ≤ 21 位，即 `< 2_097_152`）为 Morton 键。
///
/// 超出范围的高位会被**静默截断**，因此调用方必须先校验范围；
/// 提供 [`encode_checked`] 做校验版本。
pub fn encode(x: u32, y: u32, z: u32) -> u64 {
    spread3(x) | (spread3(y) << 1) | (spread3(z) << 2)
}

/// 带范围校验的编码。
pub fn encode_checked(x: u32, y: u32, z: u32) -> Option<u64> {
    const LIMIT: u32 = 1 << 21;
    if x >= LIMIT || y >= LIMIT || z >= LIMIT {
        return None;
    }
    Some(encode(x, y, z))
}

/// 解码 Morton 键。
pub fn decode(code: u64) -> (u32, u32, u32) {
    (compact3(code), compact3(code >> 1), compact3(code >> 2))
}

/// Chunk 局部坐标（每轴 `0..32`，5 位）的 15 位 Morton 键。
pub fn encode_chunk_local(x: u32, y: u32, z: u32) -> u32 {
    debug_assert!(x < 32 && y < 32 && z < 32, "Chunk 局部坐标必须在 0..32");
    let key = encode(x & 31, y & 31, z & 31);
    (key & 0x7FFF) as u32
}

/// 逆操作。
pub fn decode_chunk_local(code: u32) -> (u32, u32, u32) {
    let (x, y, z) = decode(u64::from(code & 0x7FFF));
    (x & 31, y & 31, z & 31)
}

/// 同一 Morton 键的两个坐标在三维中的最大切比雪夫距离上界 = `1`，
/// 用于断言「相邻键 ⇒ 相邻格」这一局部性。
pub fn is_neighbor(a: u64, b: u64) -> bool {
    let (ax, ay, az) = decode(a);
    let (bx, by, bz) = decode(b);
    let dx = ax.abs_diff(bx);
    let dy = ay.abs_diff(by);
    let dz = az.abs_diff(bz);
    dx <= 1 && dy <= 1 && dz <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_over_a_grid() {
        for x in (0..64).step_by(7) {
            for y in (0..64).step_by(5) {
                for z in (0..64).step_by(3) {
                    let code = encode(x, y, z);
                    assert_eq!(decode(code), (x, y, z), "往返失败: {x},{y},{z}");
                }
            }
        }
    }

    #[test]
    fn round_trip_at_max_range() {
        let max = (1 << 21) - 1;
        for point in [
            (0, 0, 0),
            (max, 0, 0),
            (0, max, 0),
            (0, 0, max),
            (max, max, max),
        ] {
            let code = encode(point.0, point.1, point.2);
            assert_eq!(decode(code), point);
        }
        assert_eq!(encode_checked(1 << 21, 0, 0), None);
        assert!(encode_checked(max, max, max).is_some());
    }

    #[test]
    fn chunk_local_keys_fit_in_15_bits() {
        for x in [0, 1, 15, 31] {
            for y in [0, 31] {
                for z in [0, 7, 31] {
                    let key = encode_chunk_local(x, y, z);
                    assert!(key < (1 << 15));
                    assert_eq!(decode_chunk_local(key), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn consecutive_keys_are_spatially_adjacent() {
        // Morton 顺序的核心性质：键相邻 ⇒ 空间相邻（反之不成立）。
        let mut previous = 0u64;
        for x in 0..32u32 {
            let code = encode(x, 0, 0);
            if x > 0 {
                assert!(
                    is_neighbor(previous, code),
                    "键 {previous} 与 {code} 应相邻"
                );
            }
            previous = code;
        }
    }
}
