//! 确定性噪声。
//!
//! 全部基于 `splitmix64` 整数哈希，**不依赖 `rand`**：
//!
//! * 同一 `seed` 在任何平台、任何线程、任何编译优化级别下产生逐位相同的结果；
//! * 无全局状态，可安全并行（地质烘焙天然是数据并行的）。
//!
//! 稳定性承诺：本模块的哈希常量与插值方式一经发布不得修改——地貌是玩家存档的一部分，
//! 换噪声等于换星球。

/// 64 位整数混合函数（splitmix64）。
///
/// 输出具有良好的雪崩性；作为整个引擎所有程序化生成的熵源。
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 整数格点上的 `[0, 1)` 哈希值。
pub fn hash_unit(seed: u64, x: i64, y: i64, z: i64) -> f64 {
    let mixed = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    (splitmix64(mixed) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// 三次平滑插值权重（`smoothstep`），保证一阶连续。
#[inline]
fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// 三维值噪声，输出 `[0, 1]`。
pub fn value_noise_3d(seed: u64, p: [f64; 3]) -> f64 {
    let base = [p[0].floor(), p[1].floor(), p[2].floor()];
    let frac = [p[0] - base[0], p[1] - base[1], p[2] - base[2]];
    let (x0, y0, z0) = (base[0] as i64, base[1] as i64, base[2] as i64);

    let corner = |dx: i64, dy: i64, dz: i64| hash_unit(seed, x0 + dx, y0 + dy, z0 + dz);
    let tx = smooth(frac[0]);
    let ty = smooth(frac[1]);
    let tz = smooth(frac[2]);

    let c00 = lerp(corner(0, 0, 0), corner(1, 0, 0), tx);
    let c10 = lerp(corner(0, 1, 0), corner(1, 1, 0), tx);
    let c01 = lerp(corner(0, 0, 1), corner(1, 0, 1), tx);
    let c11 = lerp(corner(0, 1, 1), corner(1, 1, 1), tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    lerp(c0, c1, tz)
}

/// 分形布朗运动（多倍频叠加），输出 `[0, 1]`。
///
/// * `lacunarity`：每层频率倍率（典型 2.0）；
/// * `gain`：每层振幅衰减（典型 0.5，即 1/f 谱）。
pub fn fbm(seed: u64, p: [f64; 3], octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for octave in 0..octaves {
        let octave_seed = seed.wrapping_add((octave as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        sum += amplitude
            * value_noise_3d(
                octave_seed,
                [p[0] * frequency, p[1] * frequency, p[2] * frequency],
            );
        norm += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_unit_is_in_range_and_deterministic() {
        for i in 0..64 {
            let value = hash_unit(42, i, -i, i * 7);
            assert!((0.0..1.0).contains(&value), "越界: {value}");
            assert_eq!(value, hash_unit(42, i, -i, i * 7), "同输入必须同输出");
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let a = hash_unit(1, 3, 4, 5);
        let b = hash_unit(2, 3, 4, 5);
        assert_ne!(a, b);
    }

    #[test]
    fn value_noise_is_bounded_and_continuous_at_integer_lattice() {
        for i in 0..32 {
            let p = [i as f64 * 0.37, i as f64 * -0.11, i as f64 * 0.91];
            let value = value_noise_3d(7, p);
            assert!((0.0..=1.0).contains(&value), "越界: {value}");
        }
        // 在格点上噪声退化为该点的哈希值。
        let lattice = value_noise_3d(7, [2.0, 3.0, 4.0]);
        assert_eq!(lattice, hash_unit(7, 2, 3, 4));
    }

    #[test]
    fn fbm_stays_normalized_and_is_reproducible() {
        let p = [0.31, -1.77, 4.02];
        let first = fbm(9, p, 5, 2.0, 0.5);
        let second = fbm(9, p, 5, 2.0, 0.5);
        assert_eq!(first, second);
        assert!((0.0..=1.0).contains(&first), "越界: {first}");
        assert_eq!(fbm(9, p, 0, 2.0, 0.5), 0.0, "零倍频必须返回 0 而不是 NaN");
    }
}
