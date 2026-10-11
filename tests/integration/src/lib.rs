//! 端到端测试的共享装配代码（**无图形界面**）。
//!
//! 这里的每个函数都刻意只用「公开 API」拼装世界，不碰任何内部捷径——
//! 如果某个 crate 的公共接口无法支撑一次完整的
//! 「生成 → 存储 → 网格化 → 落盘 → 联机校验 → 插值」流程，
//! 这个测试箱就会编译失败，这正是它的价值所在。

use block_schema::{BlockDef, BlockId, BlockPhysics, BlockRegistry, Material, RenderClass};
use geology::{GeologyBlocks, GeologyConfig, GeologyGenerator};
use glam::DVec3;
use planet_coords::VoxelCoord;
use storage::{Chunk, MemoryWorld};
use streaming::generate_chunk;
use world_contracts::{ChunkCoord, StratumSampler, CHUNK_EDGE, CHUNK_VOLUME};

/// 测试统一使用的行星半径（米）。
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// 测试统一使用的种子。
pub const TEST_SEED: u64 = 0x7A0E_2026;

/// 引擎默认方块注册表（与 [`GeologyBlocks::canonical`] 对齐）。
pub fn registry() -> BlockRegistry {
    let mut registry = BlockRegistry::new();
    let definitions: [(BlockId, &'static str, Material, BlockPhysics, RenderClass); 5] = [
        (
            BlockId::new(1),
            "stone",
            Material::Rock,
            BlockPhysics::ROCK,
            RenderClass::Opaque,
        ),
        (
            BlockId::new(2),
            "soil",
            Material::Soil,
            BlockPhysics::ROCK,
            RenderClass::Opaque,
        ),
        (
            BlockId::new(3),
            "sand",
            Material::Sand,
            BlockPhysics::ROCK,
            RenderClass::Opaque,
        ),
        (
            BlockId::new(4),
            "water",
            Material::Water,
            BlockPhysics::WATER,
            RenderClass::Translucent,
        ),
        (
            BlockId::new(5),
            "mantle",
            Material::Metal,
            BlockPhysics::ROCK,
            RenderClass::Opaque,
        ),
    ];
    for (id, name, material, physics, render) in definitions {
        registry
            .push(BlockDef {
                id,
                name,
                material,
                physics,
                render,
            })
            .expect("测试注册表必须是连续的 ID 序列");
    }
    registry
}

/// 构造地质生成器。
pub fn generator(seed: u64) -> GeologyGenerator {
    GeologyGenerator::new(GeologyConfig {
        seed,
        ..GeologyConfig::default()
    })
}

/// 某个方向上的地平线弧长（米），供流式/兴趣管理测试复用。
pub fn horizon_arc_m(height_m: f64) -> f64 {
    planet_coords::SphereGeometry::new(EARTH_RADIUS_M).horizon_arc_m(height_m)
}

/// 取包含「该方向地表点」的 Chunk 坐标。
pub fn surface_chunk_containing(geo: &GeologyGenerator, dir: DVec3) -> ChunkCoord {
    let dir = dir.normalize();
    let radius = EARTH_RADIUS_M + geo.surface_height_m(dir);
    let position = dir * radius;
    ChunkCoord::from_voxel(VoxelCoord::new(
        position.x.round() as i64,
        position.y.round() as i64,
        position.z.round() as i64,
    ))
}

/// Chunk 局部线性下标。
pub fn local_index(x: u32, y: u32, z: u32) -> usize {
    (x + y * CHUNK_EDGE + z * CHUNK_EDGE * CHUNK_EDGE) as usize
}

/// 生成一个地表 Chunk 的体素。
pub fn generate_surface_chunk(geo: &GeologyGenerator, coord: ChunkCoord) -> Vec<BlockId> {
    generate_chunk(geo, coord, EARTH_RADIUS_M)
}

/// 组装一个只含单个 Chunk 的内存世界。
pub fn world_with_chunk(coord: ChunkCoord, voxels: &[BlockId]) -> (MemoryWorld, Chunk) {
    assert_eq!(voxels.len(), CHUNK_VOLUME, "体素缓冲长度必须是 32³");
    let chunk = Chunk::from_voxels(coord, voxels).expect("Chunk 构造失败");
    let mut world = MemoryWorld::new(8, EARTH_RADIUS_M);
    world
        .pool_mut()
        .insert(chunk.clone())
        .expect("Chunk 入池失败");
    (world, chunk)
}

/// 校验地质层使用的方块 ID 都能在注册表中查到。
pub fn assert_geology_blocks_registered(registry: &BlockRegistry) {
    let blocks = GeologyBlocks::canonical();
    for (label, id) in [
        ("topsoil", blocks.topsoil),
        ("sediment", blocks.sediment),
        ("rock", blocks.rock),
        ("mantle", blocks.mantle),
    ] {
        assert!(
            registry.get(id).is_some(),
            "地质层的 {label} 使用了未注册的方块 ID {}",
            id.raw()
        );
    }
}
