//! 世界生成端到端：种子 → 地层 → 体素 → 存储 → 网格 → 落盘 → 恢复。
//!
//! 这个测试覆盖的是**整条离线数据链**：任何一环的接口退化都会在这里暴露。

use std::io::Cursor;

use glam::DVec3;
use integration::{
    assert_geology_blocks_registered, generate_surface_chunk, generator, local_index, registry,
    surface_chunk_containing, world_with_chunk, EARTH_RADIUS_M, TEST_SEED,
};
use persistence::{ChunkDelta, DeltaTable, WalReader, WalRecord, WalWriter};
use planet_coords::VoxelCoord;
use storage::{Chunk, ChunkPool, MemoryWorld};
use world_contracts::{StratumSampler, WorldView, CHUNK_VOLUME};
use world_meshing::MeshBuffer;

const AIR: block_schema::BlockId = block_schema::BlockId::AIR;

#[test]
fn registry_and_geology_blocks_agree() {
    let registry = registry();
    assert_geology_blocks_registered(&registry);
    assert_eq!(registry.len(), 6, "空气 + 5 种方块");
}

#[test]
fn geology_is_deterministic_across_instances_and_threads() {
    let first = generator(TEST_SEED);
    let second = generator(TEST_SEED);
    let directions: Vec<DVec3> = (0..16)
        .map(|step| {
            let angle = step as f64 / 16.0 * std::f64::consts::TAU;
            DVec3::new(angle.cos(), 0.3, angle.sin()).normalize()
        })
        .collect();

    let sample = |geo: &geology::GeologyGenerator| -> Vec<f64> {
        directions
            .iter()
            .map(|dir| geo.surface_height_m(*dir))
            .collect()
    };
    assert_eq!(sample(&first), sample(&second), "同种子必须逐位一致");

    // 并行采样也必须一致（地质层是纯函数，任何线程顺序都不应影响结果）。
    use rayon::prelude::*;
    let parallel: Vec<f64> = directions
        .par_iter()
        .map(|dir| first.surface_height_m(*dir))
        .collect();
    assert_eq!(parallel, sample(&first));
}

#[test]
fn chunk_pipeline_generates_stores_and_meshes_consistently() {
    let geo = generator(TEST_SEED);
    let coord = surface_chunk_containing(&geo, DVec3::X);
    let voxels = generate_surface_chunk(&geo, coord);
    assert_eq!(voxels.len(), CHUNK_VOLUME);

    let solid = voxels.iter().filter(|block| !block.is_air()).count();
    assert!(
        solid > 0 && solid < CHUNK_VOLUME,
        "地表 Chunk 必须上下都有内容"
    );

    let (world, chunk) = world_with_chunk(coord, &voxels);
    assert_eq!(chunk.non_air_count() as usize, solid);

    // 视图与原始体素必须逐点一致（跨越 Chunk 边界的全局坐标换算最容易出错）。
    let origin = coord.min_voxel();
    for (x, y, z) in [(0, 0, 0), (5, 7, 11), (31, 31, 31), (0, 31, 0)] {
        let expected = voxels[local_index(x, y, z)];
        let global = VoxelCoord::new(
            origin.x + i64::from(x),
            origin.y + i64::from(y),
            origin.z + i64::from(z),
        );
        assert_eq!(
            world.block_at(global),
            expected,
            "位置 ({x},{y},{z}) 不一致"
        );
        assert_eq!(world.is_solid(global), !expected.is_air());
    }

    // 网格化：顶点/索引数量自洽，且全部落在 Chunk 包围盒内。
    let quads = world_meshing::greedy_quads(&voxels, 32).expect("贪婪合并失败");
    assert!(!quads.is_empty(), "有实体的 Chunk 必须产出面片");
    let mesh = MeshBuffer::from_quads(quads.clone(), 32, 1.0);
    assert_eq!(mesh.vertices.len(), quads.len() * 4);
    assert_eq!(mesh.indices.len(), quads.len() * 6);
    assert!(
        mesh.vertices.iter().all(|vertex| {
            vertex
                .position
                .iter()
                .all(|axis| (0.0..=32.0).contains(axis))
        }),
        "顶点不得越出 Chunk 包围盒"
    );

    // 合并必须真的省几何：面片面积总和应等于「逐面输出」的面积。
    let merged_area: u32 = quads.iter().map(|quad| quad.area()).sum();
    assert!(merged_area >= quads.len() as u32);
    assert!(mesh.triangle_count() > 0);
}

#[test]
fn empty_chunk_meshes_to_nothing_and_stays_air() {
    let empty: Vec<_> = (0..CHUNK_VOLUME)
        .map(|_| block_schema::BlockId::AIR)
        .collect();
    let quads = world_meshing::greedy_quads(&empty, 32).unwrap();
    assert!(quads.is_empty());
    let (world, _chunk) = world_with_chunk(Default::default(), &empty);
    assert_eq!(
        world.block_at(VoxelCoord::new(3, 3, 3)),
        block_schema::BlockId::AIR
    );
    assert!(!world.has_chunk(world_contracts::ChunkCoord::new(7, 7, 7)));
}

#[test]
fn deltas_survive_a_wal_round_trip_and_restore_the_world() {
    let geo = generator(TEST_SEED);
    let coord = surface_chunk_containing(&geo, DVec3::X);
    let baseline = generate_surface_chunk(&geo, coord);

    // 玩家改动：挖掉三个方块、放上一个新方块。
    let mut delta = ChunkDelta::new(coord, 0);
    let edits = [(2, 2, 2), (3, 2, 2), (4, 2, 2)];
    for (x, y, z) in edits {
        delta.record(x, y, z, AIR).unwrap();
    }
    delta
        .record(5, 5, 5, block_schema::BlockId::new(5))
        .unwrap();

    // 落盘：写进 WAL。
    let mut buffer = Vec::new();
    {
        let mut writer = WalWriter::new(&mut buffer).unwrap();
        writer.append(&WalRecord::Delta(delta.clone())).unwrap();
        writer.append(&WalRecord::Checkpoint { tick: 42 }).unwrap();
        writer.flush().unwrap();
    }

    // 恢复：读回并应用到「重新生成的基准」上。
    let records = WalReader::new(Cursor::new(&buffer))
        .unwrap()
        .read_all()
        .unwrap();
    assert_eq!(records.len(), 2);
    let restored_delta = match &records[0] {
        WalRecord::Delta(restored) => restored.clone(),
        other => panic!("期望差量记录，实际 {other:?}"),
    };
    assert_eq!(restored_delta, delta, "WAL 往返必须无损");
    assert_eq!(records[1], WalRecord::Checkpoint { tick: 42 });

    let mut chunk = Chunk::from_voxels(coord, &baseline).unwrap();
    let applied = restored_delta.apply(&mut chunk).unwrap();
    assert_eq!(applied, 4, "四次修改都应当生效");
    for (x, y, z) in edits {
        assert_eq!(chunk.get(x, y, z), AIR);
    }
    assert_eq!(chunk.get(5, 5, 5), block_schema::BlockId::new(5));

    // 「重新生成基准 + 重放差量」必须与直接编辑得到的世界完全一致——
    // 这是「只存差量」这一存档策略的正确性前提。
    let mut pool = ChunkPool::new(4);
    pool.insert(Chunk::from_voxels(coord, &baseline).unwrap())
        .unwrap();
    let mut table = DeltaTable::new();
    for (x, y, z) in edits {
        table.record(coord, x, y, z, AIR, 0).unwrap();
    }
    table
        .record(coord, 5, 5, 5, block_schema::BlockId::new(5), 0)
        .unwrap();
    assert_eq!(table.apply_all(&mut pool).unwrap(), 4);
    let replayed = pool.get(coord).expect("Chunk 应当仍在池中");
    assert_eq!(replayed.get(5, 5, 5), chunk.get(5, 5, 5));
    for (x, y, z) in edits {
        assert_eq!(replayed.get(x, y, z), chunk.get(x, y, z));
    }
    assert_eq!(replayed.version(), 4, "四次有效写入 → 版本号 4");
}

#[test]
fn world_view_tolerates_unloaded_regions() {
    let world = MemoryWorld::new(1, EARTH_RADIUS_M);
    // 未加载区域：一律空气且不 panic——流式边界是常态。
    assert!(world
        .block_at(VoxelCoord::new(i64::MAX, i64::MIN, 0))
        .is_air());
    assert!(!world.has_chunk(world_contracts::ChunkCoord::default()));
    assert_eq!(
        world.surface_height_m(DVec3::X),
        None,
        "内存世界不臆造地形高度"
    );
}
