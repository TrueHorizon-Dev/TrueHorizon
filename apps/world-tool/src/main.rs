//! # world-tool —— 地质烘焙、世界迁移与诊断 CLI
//!
//! 面向**构建流程与运维**的命令行工具：不启动窗口、不连接网络，纯离线处理世界数据。
//!
//! ```text
//! world-tool bake     --seed 42 --origin-radius 6371000 --chunks 4 --out bake-out
//! world-tool migrate  --in save.wal --out save.migrated.wal
//! world-tool diagnose --seed 42
//! ```
//!
//! * `bake`：把种子烘焙成 Chunk 快照，并以 WAL 记录差量——供专用服务器预热或分发给客户端；
//! * `migrate`：把旧 WAL 逐条读出、按当前格式重写（不认识的记录类型会明确报告并停下）；
//! * `diagnose`：打印行星摘要、方块注册表校验、地层剖面抽样与 RHI 环境探测，用于排查
//!   「地形看起来不对」「这台机器为什么起不来 GPU 路径」这类问题的第一现场取证。

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

use block_schema::{BlockDef, BlockId, BlockPhysics, BlockRegistry, Material, RenderClass};
use geology::{GeologyBlocks, GeologyConfig, GeologyGenerator};
use persistence::{ChunkDelta, ChunkSnapshot, WalReader, WalRecord, WalWriter};
use planet_coords::VoxelCoord;
use rhi_vulkan::FeatureRequirements;
use storage::Chunk;
use streaming::generate_chunk;
use world_contracts::{ChunkCoord, StratumSampler, CHUNK_VOLUME};

const EARTH_RADIUS_M: f64 = 6_371_000.0;
const DEFAULT_SEED: u64 = 0x7A0E_2026;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().map(String::as_str) else {
        print_usage();
        return Err("缺少子命令".into());
    };
    let options = Options::parse(&args[1..])?;

    match command {
        "bake" => bake(&options),
        "migrate" => migrate(&options),
        "diagnose" => diagnose(&options),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        },
        other => {
            print_usage();
            Err(format!("未知子命令: {other}").into())
        },
    }
}

fn print_usage() {
    println!(
        "TrueHorizon world-tool\n\
         \n\
         USAGE:\n\
         \x20 world-tool bake     [--seed N] [--chunks N] [--out DIR]\n\
         \x20 world-tool migrate  --in FILE --out FILE\n\
         \x20 world-tool diagnose [--seed N]\n"
    );
}

/// 极简 `--key value` 解析器（不引入 clap：工具链要保持零依赖）。
#[derive(Debug, Default)]
struct Options {
    seed: Option<u64>,
    chunks: Option<usize>,
    out: Option<PathBuf>,
    input: Option<PathBuf>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("参数 {flag} 缺少取值"))?;
            match flag {
                "--seed" => options.seed = Some(value.parse().map_err(|_| "种子必须是 u64")?),
                "--chunks" => {
                    options.chunks = Some(value.parse().map_err(|_| "chunks 必须是整数")?)
                },
                "--out" => options.out = Some(PathBuf::from(value)),
                "--in" => options.input = Some(PathBuf::from(value)),
                other => return Err(format!("未知参数: {other}")),
            }
            index += 2;
        }
        Ok(options)
    }

    fn seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }
}

/// 引擎默认方块注册表（与 [`GeologyBlocks::canonical`] 必须一致）。
fn canonical_registry() -> Result<BlockRegistry, Box<dyn Error>> {
    let mut registry = BlockRegistry::new();
    let definitions: [(BlockId, &str, Material, BlockPhysics, RenderClass); 5] = [
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
        registry.push(BlockDef {
            id,
            name,
            material,
            physics,
            render,
        })?;
    }
    Ok(registry)
}

/// 烘焙：生成一批 Chunk 并把差量写进 WAL。
fn bake(options: &Options) -> Result<(), Box<dyn Error>> {
    let count = options.chunks.unwrap_or(4).max(1);
    let out_dir = options
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("bake-out"));
    std::fs::create_dir_all(&out_dir)?;
    let wal_path = out_dir.join(format!("bake-{:016x}.wal", options.seed()));

    let geology = GeologyGenerator::new(GeologyConfig {
        seed: options.seed(),
        ..GeologyConfig::default()
    });
    println!(
        "[bake] 种子 {} | 板块 {} | 半径 {:.0} m | Chunk {} 个",
        options.seed(),
        geology.plates().len(),
        EARTH_RADIUS_M,
        count
    );

    // 以赤道方向的表面为起点，沿 +X 方向排布 Chunk。
    let surface_voxel = (EARTH_RADIUS_M + geology.surface_height_m(glam::DVec3::X)).round() as i64;
    let start = Instant::now();
    let mut written = 0u64;
    let mut solid_total = 0u64;
    let mut last_chunk: Option<Chunk> = None;

    let file = BufWriter::new(File::create(&wal_path)?);
    let mut wal = WalWriter::new(file)?;

    for index in 0..count {
        let coord = ChunkCoord::from_voxel(VoxelCoord::new(surface_voxel, index as i64 * 32, 0));
        let voxels = generate_chunk(&geology, coord, EARTH_RADIUS_M);
        let chunk = Chunk::from_voxels(coord, &voxels)?;
        solid_total += u64::from(chunk.non_air_count());

        // 差量：记录全部非空气体素相对于「空基准」的内容，
        // 使接收方无需重放地质算法即可得到一致结果。
        let mut delta = ChunkDelta::new(coord, 0);
        for (x, y, z, block) in chunk.iter_solid() {
            delta.record(x, y, z, block)?;
        }
        let edits = delta.len();
        wal.append(&WalRecord::Delta(delta))?;
        written += 1;
        println!(
            "[bake]   {} -> {} 体素（调色板 {} 项，{:.1} KiB，{} 条编辑）",
            coord
                .key()
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            CHUNK_VOLUME,
            chunk.palette_len(),
            chunk.packed_bytes() as f64 / 1024.0,
            edits
        );
        last_chunk = Some(chunk);
    }

    // 压实锚点：整块 Chunk 快照以 rkyv 归档写入。
    // 恢复端拿到这条记录即可零拷贝读出稠密基准，不必重放前面的全部差量。
    if let Some(chunk) = &last_chunk {
        let snapshot = ChunkSnapshot::from_chunk(chunk);
        let archived_bytes = snapshot.archive()?.len();
        let coord = snapshot.coord();
        wal.append(&WalRecord::Snapshot(snapshot))?;
        println!(
            "[bake]   快照锚点 {} -> {} 体素，归档 {:.1} KiB（零拷贝可读）",
            coord
                .key()
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            ChunkSnapshot::BLOCK_COUNT,
            archived_bytes as f64 / 1024.0
        );
    }
    wal.append(&WalRecord::Checkpoint {
        tick: options.seed(),
    })?;
    wal.flush()?;

    println!(
        "[bake] 完成：{} 个 Chunk，{} 个实体体素，用时 {:.2?}，输出 {}",
        written,
        solid_total,
        start.elapsed(),
        wal_path.display()
    );
    Ok(())
}

/// 迁移：读出旧 WAL 并按当前格式重写。
fn migrate(options: &Options) -> Result<(), Box<dyn Error>> {
    let input = options.input.clone().ok_or("migrate 需要 --in FILE")?;
    let output = options.out.clone().ok_or("migrate 需要 --out FILE")?;

    let reader = WalReader::new(BufReader::new(File::open(&input)?))?;
    let records = reader.read_all()?;
    println!("[migrate] 读入 {} 条记录，开始重写", records.len());

    let mut writer = WalWriter::new(BufWriter::new(File::create(&output)?))?;
    let mut deltas = 0u64;
    let mut checkpoints = 0u64;
    let mut snapshots = 0u64;
    for record in &records {
        match record {
            WalRecord::Delta(_) => deltas += 1,
            WalRecord::Checkpoint { .. } => checkpoints += 1,
            WalRecord::Snapshot(_) => snapshots += 1,
        }
        writer.append(record)?;
    }
    writer.flush()?;
    println!(
        "[migrate] 完成：差量 {} 条，检查点 {} 条，快照 {} 条 -> {}",
        deltas,
        checkpoints,
        snapshots,
        output.display()
    );
    Ok(())
}

/// 诊断：注册表校验 + 行星摘要 + 剖面抽样。
fn diagnose(options: &Options) -> Result<(), Box<dyn Error>> {
    let registry = canonical_registry()?;
    println!(
        "[diagnose] 方块注册表：{} 项（空气 + {} 种实体）",
        registry.len(),
        registry.len() - 1
    );

    // 注册表与地质层的方块映射必须一致，否则地形会整片变成「未知方块」。
    let blocks = GeologyBlocks::canonical();
    for (label, id) in [
        ("topsoil", blocks.topsoil),
        ("sediment", blocks.sediment),
        ("rock", blocks.rock),
        ("mantle", blocks.mantle),
    ] {
        let def = registry.get(id).ok_or_else(|| {
            format!(
                "地质层使用的方块 ID {} （{label}）未在注册表中定义",
                id.raw()
            )
        })?;
        println!("[diagnose]   {label:<9} -> #{} {}", id.raw(), def.name);
    }

    let geology = GeologyGenerator::new(GeologyConfig {
        seed: options.seed(),
        ..GeologyConfig::default()
    });
    println!(
        "[diagnose] 行星：半径 {EARTH_RADIUS_M:.0} m | 板块 {} | 海平面 {:.0} m | 地壳厚 {:.0} m",
        geology.plates().len(),
        geology.config().sea_level_m,
        geology.config().crust_depth_m
    );

    let mut land = 0;
    let mut ocean = 0;
    let mut highest = f64::MIN;
    let mut lowest = f64::MAX;
    println!("[diagnose] 地层抽样（8 个方向）：");
    for step in 0..8 {
        let angle = step as f64 / 8.0 * std::f64::consts::TAU;
        let dir = glam::DVec3::new(angle.cos(), 0.15, angle.sin()).normalize();
        let profile = geology.strata_at(dir);
        let height = profile.surface_height_m;
        highest = highest.max(height);
        lowest = lowest.min(height);
        if geology.is_submerged(dir) {
            ocean += 1;
        } else {
            land += 1;
        }
        println!(
            "[diagnose]   dir({:+.3},{:+.3},{:+.3}) 高程 {:+.0} m | {} | 层数 {} | 总厚 {:.0} m",
            dir.x,
            dir.y,
            dir.z,
            height,
            if geology.is_submerged(dir) {
                "海洋"
            } else {
                "陆地"
            },
            profile.len(),
            profile.total_thickness_m()
        );
    }
    println!(
        "[diagnose] 汇总：陆地 {land} / 海洋 {ocean} | 高程区间 [{lowest:+.0}, {highest:+.0}] m"
    );

    diagnose_rhi();
    Ok(())
}

/// RHI 环境探测：这台机器到底能不能走 GPU 路径，以及差在哪个特性上。
///
/// 无 Vulkan 的 CI 会打印明确原因（而不是静默跳过），因此这条输出同时是
/// 「图形层可用性」的回归证据。
fn diagnose_rhi() {
    let probe = rhi_vulkan::probe(&FeatureRequirements::minimum());
    println!(
        "[diagnose] Vulkan loader: {} | 实例: {} | 物理设备: {}",
        match (&probe.loader_api_version, probe.loader_available) {
            (Some(version), _) => format!("可用（{version}）"),
            (None, true) => "可用（版本未知）".to_string(),
            (None, false) => "不可用".to_string(),
        },
        if probe.instance_created {
            "已创建"
        } else {
            "未创建"
        },
        probe.physical_devices
    );
    match &probe.device {
        Some(info) => {
            println!(
                "[diagnose]   GPU: {} | Vulkan {} | 驱动 {:#x} | 显存 {:.1} GiB",
                info.name,
                info.api_version_string(),
                info.driver_version,
                info.vram_bytes as f64 / (1u64 << 30) as f64
            );
            let missing = info.missing_requirements(&FeatureRequirements::desktop_full());
            println!(
                "[diagnose]   桌面全特性缺失: {}",
                if missing.is_empty() {
                    "无（可走完整 GPU-Driven 路径）".to_string()
                } else {
                    missing.join(", ")
                }
            );
        },
        None => println!("[diagnose]   GPU: 无可用设备（{}）", probe.message),
    }
}
