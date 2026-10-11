//! # world-meshing —— 近景体素网格生成
//!
//! 覆盖 `0 ~ 256 m` 的可行走区域：把 32³ 的体素 Chunk 转成可上传显存的顶点/索引缓冲。
//!
//! * [`greedy`]：贪心四边形合并（纯 CPU、可单测、结果确定）。
//! * [`MeshBuffer`]：`Pod` 顶点布局，零拷贝上传。
//! * [`MeshTaskPool`]：rayon 工作池，把 Chunk 网格化并行铺满所有核心。
//!
//! 性能约定：单个 32³ Chunk 的网格化目标为 **< 200 µs**（8 核并行下每帧可处理
//! 数百个 Chunk）。所有热路径**不分配**除输出缓冲以外的内存（掩码与占用位图按
//! Chunk 复用，见 [`ChunkMesher`] 的路线图）。
//!
//! 安全边界：本 crate 是少数允许 `unsafe` 的地方之一（SIMD 面片扫描），
//! 但骨架阶段全部为安全代码，未使用 `unsafe`。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod greedy;

use std::sync::Arc;

use block_schema::BlockId;
use rayon::prelude::*;
use world_contracts::ChunkCoord;

pub use greedy::{greedy_quads, MeshError, Quad};

/// 顶点布局（`Pod`，36 字节且无填充）。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    /// 位置（Chunk 局部坐标，米）。
    pub position: [f32; 3],
    /// 面法线。
    pub normal: [f32; 3],
    /// 图集 UV（体素为单位，供着色器做纹理铺贴）。
    pub uv: [f32; 2],
    /// 方块 ID。
    ///
    /// 这里刻意放**方块 ID 而非材质槽位**：材质表查表由 GPU 完成（bindless 表常驻显存），
    /// 从而 CPU 侧无需持有 `BlockRegistry`，网格化任务也就不必跨线程共享注册表。
    pub block_id: u32,
}

/// 一个 Chunk 的网格产物。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshBuffer {
    /// 顶点。
    pub vertices: Vec<VoxelVertex>,
    /// 索引（三角形列表）。
    pub indices: Vec<u32>,
    /// 合并后的面片（保留下来供 GPU-Driven 剔除与调试可视化使用）。
    pub quads: Vec<Quad>,
}

impl MeshBuffer {
    /// 由面片列表展开为顶点 + 索引。
    ///
    /// `edge` 为 Chunk 边长（体素），`voxel_edge_m` 为体素边长（米）。
    /// 负面（`positive == false`）使用反向绕序，使三角形正面始终朝外。
    pub fn from_quads(quads: Vec<Quad>, edge: u32, voxel_edge_m: f32) -> Self {
        let mut vertices = Vec::with_capacity(quads.len() * 4);
        let mut indices = Vec::with_capacity(quads.len() * 6);

        for quad in &quads {
            let base = vertices.len() as u32;
            let normal = quad.normal();
            let corners = quad_corners(quad, edge, voxel_edge_m);
            for (position, uv) in corners {
                vertices.push(VoxelVertex {
                    position,
                    normal,
                    uv,
                    block_id: u32::from(quad.block.raw()),
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self {
            vertices,
            indices,
            quads,
        }
    }

    /// 顶点字节数。
    pub fn vertex_bytes(&self) -> usize {
        self.vertices.len() * core::mem::size_of::<VoxelVertex>()
    }

    /// 索引字节数。
    pub fn index_bytes(&self) -> usize {
        self.indices.len() * core::mem::size_of::<u32>()
    }

    /// 是否为空网格。
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// 三角形数量。
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// 计算面片的四个角点（位置与 UV）。
///
/// UV 用**面内体素坐标**而非 `0..1`：合并后的面片必须让纹理按体素密度平铺，
/// 否则一块 8×8 的地面会把整张贴图拉伸上去。
fn quad_corners(quad: &Quad, edge: u32, voxel_edge_m: f32) -> [([f32; 3], [f32; 2]); 4] {
    let (u_axis, v_axis) = quad.uv_axes();
    let axis = quad.axis as usize;
    let plane = quad.slice.min(edge) as f32;
    let corner = |du: u32, dv: u32| -> ([f32; 3], [f32; 2]) {
        let mut position = [0.0f32; 3];
        position[axis] = plane;
        position[u_axis] = (quad.u0 + du) as f32;
        position[v_axis] = (quad.v0 + dv) as f32;
        (
            [
                position[0] * voxel_edge_m,
                position[1] * voxel_edge_m,
                position[2] * voxel_edge_m,
            ],
            [du as f32, dv as f32],
        )
    };
    // 绕序跟随法线：正面逆时针，负面反向。
    if quad.positive {
        [
            corner(0, 0),
            corner(quad.width, 0),
            corner(quad.width, quad.height),
            corner(0, quad.height),
        ]
    } else {
        [
            corner(0, 0),
            corner(0, quad.height),
            corner(quad.width, quad.height),
            corner(quad.width, 0),
        ]
    }
}

/// 一个待网格化的 Chunk。
#[derive(Clone, Debug)]
pub struct ChunkMeshRequest {
    /// Chunk 坐标。
    pub coord: ChunkCoord,
    /// 体素数据（`Arc` 共享，避免跨线程复制 32³ 的负载）。
    pub voxels: Arc<[BlockId]>,
    /// LOD 层级（0 = 全分辨率）。
    pub lod: u32,
}

impl ChunkMeshRequest {
    /// 构造。
    pub fn new(coord: ChunkCoord, voxels: Arc<[BlockId]>, lod: u32) -> Self {
        Self { coord, voxels, lod }
    }

    /// 由缓冲长度反推 Chunk 边长，校验声明的边长是否自洽。
    pub fn validate_edge(&self, declared: u32) -> Result<(), MeshError> {
        if declared == 0 {
            return Err(MeshError::EdgeMustBePositive);
        }
        let expected = (declared as usize).pow(3);
        if self.voxels.len() != expected {
            return Err(MeshError::VolumeSizeMismatch {
                expected,
                actual: self.voxels.len(),
            });
        }
        Ok(())
    }
}

/// 一次网格化的结果。
#[derive(Clone, Debug)]
pub struct ChunkMeshOutput {
    /// Chunk 坐标。
    pub coord: ChunkCoord,
    /// LOD 层级。
    pub lod: u32,
    /// 网格产物。
    pub mesh: MeshBuffer,
}

/// rayon 工作池。
///
/// 固定线程数（默认取 `min(物理核心数, 8)`）：网格化的并行收益在 8 线程后饱和，
/// 再多的线程只会与渲染线程抢缓存。
pub struct MeshTaskPool {
    pool: rayon::ThreadPool,
}

impl MeshTaskPool {
    /// 以指定线程数创建。
    pub fn new(threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("th-meshing-{index}"))
            .build()?;
        Ok(Self { pool })
    }

    /// 默认线程数。
    pub fn default_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8)
    }

    /// 线程数。
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// 并行网格化一批 Chunk。
    ///
    /// 单个 Chunk 失败不会中断整批：错误与成功结果一一对应地返回，顺序与输入一致。
    pub fn mesh_batch(
        &self,
        requests: &[ChunkMeshRequest],
        edge: u32,
        voxel_edge_m: f32,
    ) -> Vec<Result<ChunkMeshOutput, MeshError>> {
        self.pool.install(|| {
            requests
                .par_iter()
                .map(|request| {
                    request.validate_edge(edge)?;
                    let quads = greedy_quads(&request.voxels, edge)?;
                    Ok(ChunkMeshOutput {
                        coord: request.coord,
                        lod: request.lod,
                        mesh: MeshBuffer::from_quads(quads, edge, voxel_edge_m),
                    })
                })
                .collect()
        })
    }
}

impl core::fmt::Debug for MeshTaskPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MeshTaskPool")
            .field("threads", &self.threads())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use planet_coords::voxel_edge_m;

    fn solid_chunk(edge: u32) -> Arc<[BlockId]> {
        vec![BlockId::new(1); (edge * edge * edge) as usize].into()
    }

    #[test]
    fn empty_chunk_produces_empty_mesh() {
        let edge = 32;
        let voxels: Arc<[BlockId]> = vec![BlockId::AIR; (edge * edge * edge) as usize].into();
        let quads = greedy_quads(&voxels, edge).unwrap();
        let mesh = MeshBuffer::from_quads(quads, edge, voxel_edge_m() as f32);
        assert!(mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn vertex_and_index_counts_follow_quad_count() {
        let edge = 4;
        let voxels: Arc<[BlockId]> = {
            let mut data = vec![BlockId::AIR; (edge * edge * edge) as usize];
            data[(1 + edge + edge * edge) as usize] = BlockId::new(1);
            data.into()
        };
        let quads = greedy_quads(&voxels, edge).unwrap();
        let mesh = MeshBuffer::from_quads(quads.clone(), edge, voxel_edge_m() as f32);
        assert_eq!(mesh.vertices.len(), quads.len() * 4);
        assert_eq!(mesh.indices.len(), quads.len() * 6);
        assert_eq!(mesh.triangle_count(), quads.len() * 2);
        assert_eq!(
            core::mem::size_of::<VoxelVertex>(),
            36,
            "顶点布局不得有填充"
        );
        assert_eq!(mesh.vertex_bytes(), mesh.vertices.len() * 36);
        assert!(
            mesh.vertices
                .iter()
                .all(|v| v.position.iter().all(|c| c.is_finite())),
            "顶点必须是有限值"
        );
        assert!(mesh.vertices.iter().all(|v| v.block_id == 1));
        assert!((voxel_edge_m() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn vertices_stay_inside_chunk_bounds() {
        let edge = 8;
        let voxels: Arc<[BlockId]> = {
            let mut data = vec![BlockId::AIR; (edge * edge * edge) as usize];
            data.fill(BlockId::new(1));
            data.into()
        };
        let quads = greedy_quads(&voxels, edge).unwrap();
        let mesh = MeshBuffer::from_quads(quads, edge, 2.0);
        let limit = edge as f32 * 2.0;
        for vertex in &mesh.vertices {
            for axis in vertex.position {
                assert!((0.0..=limit).contains(&axis), "顶点越界: {axis}");
            }
        }
    }

    #[test]
    fn batch_returns_one_result_per_request_in_order() {
        let pool = MeshTaskPool::new(2).expect("线程池创建失败");
        assert_eq!(pool.threads(), 2);
        let edge = 32;
        let requests: Vec<ChunkMeshRequest> = (0..4)
            .map(|i| {
                ChunkMeshRequest::new(
                    ChunkCoord::new(i, 0, 0),
                    if i == 2 {
                        // 故意给错长度，验证错误不会被并行执行吞掉。
                        vec![BlockId::AIR; 5].into()
                    } else {
                        solid_chunk(edge)
                    },
                    i as u32,
                )
            })
            .collect();

        let results = pool.mesh_batch(&requests, edge, voxel_edge_m() as f32);
        assert_eq!(results.len(), 4);
        assert!(results[0].is_ok() && results[1].is_ok());
        assert!(matches!(
            results[2],
            Err(MeshError::VolumeSizeMismatch { .. })
        ));
        assert!(results[3].is_ok());
        assert_eq!(results[0].as_ref().unwrap().coord, ChunkCoord::new(0, 0, 0));
        assert_eq!(results[3].as_ref().unwrap().lod, 3);
        assert_eq!(results[0].as_ref().unwrap().mesh.quads.len(), 6);
    }

    #[test]
    fn declared_edge_must_match_buffer() {
        let request =
            ChunkMeshRequest::new(ChunkCoord::default(), vec![BlockId::AIR; 27].into(), 0);
        assert!(request.validate_edge(3).is_ok());
        assert_eq!(
            request.validate_edge(4),
            Err(MeshError::VolumeSizeMismatch {
                expected: 64,
                actual: 27
            })
        );
        assert_eq!(request.validate_edge(0), Err(MeshError::EdgeMustBePositive));
    }
}
