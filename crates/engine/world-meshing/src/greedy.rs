//! 贪心网格合并。
//!
//! 对 32³ 的 Chunk，逐轴、逐切片地把同类朝向的相邻面片合并成尽可能大的矩形。
//! 相比逐面输出，地面的面片数量通常下降一个数量级以上（平坦区域呈平方级下降）。
//!
//! 算法（每个轴、每个朝向各做一次）：
//!
//! 1. 在切片平面 `slice` 上构建掩码：属于该面的方块写入 `mask[v * edge + u]`；
//! 2. 扫描掩码，先沿 `u` 方向尽量拉宽，再在保持整行一致的前提下沿 `v` 方向拉高；
//! 3. 输出一个矩形 `Quad` 并把占用格标记为已消费。

use block_schema::BlockId;

/// 一个合并后的面片，坐标位于 Chunk 局部体素空间。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quad {
    /// 面法线所在的轴：`0 = X`，`1 = Y`，`2 = Z`。
    pub axis: u8,
    /// 法线是否朝向正方向。
    pub positive: bool,
    /// 面所在的切片平面（沿 `axis` 的体素坐标，`0..=edge`）。
    pub slice: u32,
    /// 面内起始坐标（沿 `u` 轴，`u = (axis + 1) % 3`）。
    pub u0: u32,
    /// 面内起始坐标（沿 `v` 轴，`v = (axis + 2) % 3`）。
    pub v0: u32,
    /// 沿 `u` 轴的合并宽度（体素数）。
    pub width: u32,
    /// 沿 `v` 轴的合并高度（体素数）。
    pub height: u32,
    /// 该面片所属方块。
    pub block: BlockId,
}

impl Quad {
    /// 面片包含的体素面数。
    #[inline]
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// 面法线（Chunk 局部坐标，未归一化以外均为单位长度）。
    pub fn normal(&self) -> [f32; 3] {
        let mut normal = [0.0f32; 3];
        normal[self.axis as usize] = if self.positive { 1.0 } else { -1.0 };
        normal
    }

    /// 面内两个轴的索引 `(u_axis, v_axis)`。
    #[inline]
    pub fn uv_axes(&self) -> (usize, usize) {
        let axis = self.axis as usize;
        ((axis + 1) % 3, (axis + 2) % 3)
    }
}

/// 网格生成错误。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// 体素缓冲长度与 `edge³` 不符。
    VolumeSizeMismatch {
        /// 期望长度。
        expected: usize,
        /// 实际长度。
        actual: usize,
    },
    /// Chunk 边长必须为正。
    EdgeMustBePositive,
    /// 输入体素缓冲的边长与请求中的 Chunk 边长不一致。
    ChunkEdgeMismatch {
        /// 请求声明的边长。
        declared: u32,
        /// 由缓冲长度反推出的边长。
        derived: u32,
    },
}

impl core::fmt::Display for MeshError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::VolumeSizeMismatch { expected, actual } => {
                write!(f, "体素缓冲长度不匹配：期望 {expected}，实际 {actual}")
            },
            Self::EdgeMustBePositive => write!(f, "Chunk 边长必须为正数"),
            Self::ChunkEdgeMismatch { declared, derived } => {
                write!(f, "Chunk 边长不一致：声明 {declared}，缓冲推出 {derived}")
            },
        }
    }
}

impl std::error::Error for MeshError {}

/// 对 `edge³` 体素做贪心合并，输出面片列表。
///
/// `volume` 以 `x + y * edge + z * edge²` 线性索引，越界一律视为空气——
/// 因此 Chunk 边界处的面会被正确保留（不能剔除，否则接缝会漏光）。
pub fn greedy_quads(volume: &[BlockId], edge: u32) -> Result<Vec<Quad>, MeshError> {
    if edge == 0 {
        return Err(MeshError::EdgeMustBePositive);
    }
    let edge_usize = edge as usize;
    let expected = edge_usize * edge_usize * edge_usize;
    if volume.len() != expected {
        return Err(MeshError::VolumeSizeMismatch {
            expected,
            actual: volume.len(),
        });
    }

    let sample = |x: i64, y: i64, z: i64| -> BlockId {
        let e = i64::from(edge);
        if x < 0 || y < 0 || z < 0 || x >= e || y >= e || z >= e {
            return BlockId::AIR;
        }
        volume[(x + y * e + z * e * e) as usize]
    };

    let cells = edge_usize * edge_usize;
    let mut mask: Vec<Option<BlockId>> = vec![None; cells];
    let mut used: Vec<bool> = vec![false; cells];
    let mut quads: Vec<Quad> = Vec::new();

    for axis in 0..3u8 {
        let (u_axis, v_axis) = {
            let a = axis as usize;
            ((a + 1) % 3, (a + 2) % 3)
        };
        for positive in [true, false] {
            for slice in 0..=edge {
                mask.fill(None);
                used.fill(false);

                // 1) 构建掩码。
                for v in 0..edge {
                    for u in 0..edge {
                        let mut owner = [0i64; 3];
                        let mut outside = [0i64; 3];
                        owner[u_axis] = i64::from(u);
                        owner[v_axis] = i64::from(v);
                        outside[u_axis] = i64::from(u);
                        outside[v_axis] = i64::from(v);
                        if positive {
                            owner[axis as usize] = i64::from(slice) - 1;
                            outside[axis as usize] = i64::from(slice);
                        } else {
                            owner[axis as usize] = i64::from(slice);
                            outside[axis as usize] = i64::from(slice) - 1;
                        }
                        let owner_block = sample(owner[0], owner[1], owner[2]);
                        let outside_block = sample(outside[0], outside[1], outside[2]);
                        if !owner_block.is_air() && outside_block.is_air() {
                            mask[(v * edge + u) as usize] = Some(owner_block);
                        }
                    }
                }

                // 2) 贪心扫描。
                for v in 0..edge {
                    for u in 0..edge {
                        let index = (v * edge + u) as usize;
                        if used[index] {
                            continue;
                        }
                        let Some(block) = mask[index] else {
                            continue;
                        };

                        let mut width = 1;
                        while u + width < edge {
                            let next = (v * edge + u + width) as usize;
                            if used[next] || mask[next] != Some(block) {
                                break;
                            }
                            width += 1;
                        }

                        let mut height = 1;
                        'grow: while v + height < edge {
                            for k in 0..width {
                                let next = ((v + height) * edge + u + k) as usize;
                                if used[next] || mask[next] != Some(block) {
                                    break 'grow;
                                }
                            }
                            height += 1;
                        }

                        for dv in 0..height {
                            for du in 0..width {
                                used[((v + dv) * edge + u + du) as usize] = true;
                            }
                        }

                        quads.push(Quad {
                            axis,
                            positive,
                            slice,
                            u0: u,
                            v0: v,
                            width,
                            height,
                            block,
                        });
                    }
                }
            }
        }
    }

    Ok(quads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(edge: u32, solid: &[(u32, u32, u32)]) -> Vec<BlockId> {
        let mut voxels = vec![BlockId::AIR; (edge * edge * edge) as usize];
        for &(x, y, z) in solid {
            voxels[(x + y * edge + z * edge * edge) as usize] = BlockId::new(1);
        }
        voxels
    }

    fn total_area(quads: &[Quad]) -> u32 {
        quads.iter().map(Quad::area).sum()
    }

    #[test]
    fn rejects_bad_inputs() {
        assert_eq!(greedy_quads(&[], 0), Err(MeshError::EdgeMustBePositive));
        assert_eq!(
            greedy_quads(&[BlockId::AIR; 7], 2),
            Err(MeshError::VolumeSizeMismatch {
                expected: 8,
                actual: 7
            })
        );
    }

    #[test]
    fn empty_chunk_emits_nothing() {
        let quads = greedy_quads(&volume(4, &[]), 4).unwrap();
        assert!(quads.is_empty());
    }

    #[test]
    fn full_chunk_merges_each_side_into_one_quad() {
        let edge = 2;
        let mut voxels = volume(edge, &[]);
        voxels.fill(BlockId::new(1));
        let quads = greedy_quads(&voxels, edge).unwrap();
        assert_eq!(quads.len(), 6, "实心立方体的六个面应各合并为 1 个面片");
        assert!(quads.iter().all(|q| q.width == edge && q.height == edge));
        assert_eq!(total_area(&quads), 6 * edge * edge);
    }

    #[test]
    fn single_voxel_emits_six_unit_quads() {
        let quads = greedy_quads(&volume(3, &[(1, 1, 1)]), 3).unwrap();
        assert_eq!(quads.len(), 6);
        assert!(quads.iter().all(|q| q.width == 1 && q.height == 1));
        assert_eq!(quads.iter().filter(|q| q.positive).count(), 3);
        assert_eq!(quads.iter().filter(|q| !q.positive).count(), 3);
    }

    #[test]
    fn adjacent_voxels_merge_along_the_shared_plane() {
        let edge = 4;
        let quads = greedy_quads(&volume(edge, &[(1, 1, 1), (2, 1, 1)]), edge).unwrap();
        assert_eq!(quads.len(), 6, "共面部分必须合并，仍只有 6 个面片");
        assert_eq!(total_area(&quads), 10, "2×1×1 立方体的表面积 = 10");
        // 沿 X 的两个端面仍是 1×1。
        let x_faces: Vec<&Quad> = quads.iter().filter(|q| q.axis == 0).collect();
        assert_eq!(x_faces.len(), 2);
        assert!(x_faces.iter().all(|q| q.area() == 1));
    }

    #[test]
    fn interior_faces_are_culled() {
        // 3³ 的实心块在 5³ 空间正中：内部面全部被剔除。
        let edge = 5;
        let mut solid = Vec::new();
        for x in 1..4 {
            for y in 1..4 {
                for z in 1..4 {
                    solid.push((x, y, z));
                }
            }
        }
        let quads = greedy_quads(&volume(edge, &solid), edge).unwrap();
        assert_eq!(quads.len(), 6, "3³ 实心块的六个面各 1 片");
        assert_eq!(total_area(&quads), 54, "3³ 立方体表面积 = 54");
    }

    #[test]
    fn quads_never_overlap_and_stay_in_bounds() {
        let edge = 8;
        let mut solid = Vec::new();
        for x in 2..6 {
            for z in 0..8 {
                solid.push((x, 3, z));
            }
        }
        let quads = greedy_quads(&volume(edge, &solid), edge).unwrap();

        let mut covered = vec![0u32; (edge * edge * edge) as usize * 3];
        for quad in &quads {
            let cells = quad.area();
            let mut seen = std::collections::HashSet::new();
            for du in 0..quad.width {
                for dv in 0..quad.height {
                    let (u, v) = (quad.u0 + du, quad.v0 + dv);
                    assert!(u < edge && v < edge, "面片越界");
                    assert!(seen.insert((u, v)), "同一面片内出现重复格");
                    let key = ((quad.axis as usize) * (edge * edge * edge) as usize)
                        + (u * edge + v) as usize;
                    covered[key] += 1;
                }
            }
            assert_eq!(seen.len() as u32, cells);
        }
        // 每个 (轴, u, v) 位置上，正负朝向合计最多出现 2 次（各 1 次）。
        assert!(covered.iter().all(|count| *count <= 2));
        assert!(covered.iter().any(|count| *count > 0));
    }
}
