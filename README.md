# TrueHorizon

行星尺度的体素世界：Cube-Sphere 拓扑、双精度浮动原点、GPU-Driven 渲染与权威多人联机。

> 本仓库是**工作区骨架**：分层、依赖方向、契约与工作流都已就位，并带有可运行的
> 核心实现与单元测试；标注为 `TODO(...)` 的部分是明确的后续工作，而不是未知的空白。

---

## 快速开始

```bash
# 构建全部成员（含专用服务器与离线工具）
cargo build --workspace

# 跑全部单元测试与端到端测试
cargo test --workspace

# 离线诊断：打印行星摘要与方块注册表校验
cargo run -p world-tool -- diagnose --seed 42

# 烘焙一批 Chunk（输出 WAL）
cargo run -p world-tool -- bake --seed 42 --chunks 4 --out bake-out

# 无头服务器冒烟：跑满 400 个 Tick 后退出
cargo run -p dedicated-server -- --ticks 400 --port 27100

# 桌面客户端（需要 Vulkan 驱动；当前渲染后端为骨架，尚未出图）
cargo run -p client
```

工具链：Rust **1.80+**（`rust-version`），edition 2021，resolver 2。

---

## 目录结构

```text
apps/
  client/             桌面客户端（窗口、输入、帧编排）
  dedicated-server/   专用无头服务器（权威世界循环）
  world-tool/         地质烘焙 / 世界迁移 / 诊断 CLI
crates/
  core/               底层基底与规范（无机游戏语义、无第三方依赖优先级最高）
    foundation/       世代索引 Handle、Arena、微秒 Profiler
    planet-coords/    Cube-Sphere 拓扑、双精度浮动原点、曲率遮挡
    block-schema/     方块 ID、材质、物理属性规范
    world-contracts/  核心 Trait：地层采样器、世界视图只读接口
    render-contracts/ 渲染契约：SceneSnapshot、DrawCall 命令规范
  engine/             引擎与硬件层
    rhi-vulkan/       ash (Vulkan 1.3+) 硬件层、无绑定描述符、显存分配
    render-pipeline/  GPU-Driven 管线、mesh shader、球面 Clipmap
    world-meshing/    近景体素 CPU 贪心网格合并（0 ~ 256 m）
  world/              世界领域层
    geology/          板块 Voronoi、连续地层剖面（水文侵蚀待补）
    storage/          32³ Chunk 内存池、调色板压缩、Morton 空间索引
    physics/          局部切线标架、球面重力、光标射线拾取
    persistence/      稀疏修改差量表（Delta）、写时复制日志（WAL）
    simulation/       规则状态机、固定 Tick（20/60 Hz）、Wasm Mod 宿主
    streaming/        宏观四叉树 LOD 调度、惰性体素实例化任务队列
  net/                网络与多人层
    protocol/         纯 DTO：带版本的客户端预测与服务端事件
    client-sync/      客户端本地预测、服务端快照插值
    server-authority/ 服务端权威校验、基于曲率遮挡的兴趣管理（AoI）
tests/
  integration/        无图形界面的世界生成与多人联机端到端测试
```

## 依赖方向（单向，禁止回指）

```text
foundation ─┬─► planet-coords ─┬─► world-contracts ─┬─► storage ─┬─► persistence
            └─► block-schema ──┘                    │            ├─► simulation
                                                    │            └─► streaming
                                                    └─► render-contracts ─► render-pipeline
                                                                              ▲
                                              engine/rhi-vulkan ──────────────┘
apps/* 只依赖 crates/*，crates/* 之间只允许沿箭头方向依赖。
```

约束（CI 会检查）：

* 根 `Cargo.toml` 的 `[workspace.dependencies]` 是**唯一**的版本来源，成员清单里
  禁止出现字面版本号；
* `foundation` 不依赖任何内部 crate，也不引入第三方 crate；
* 上层不得直接使用 `ash` / `gpu-allocator` 类型，必须经由 `rhi-vulkan`。

---

## 关键设计决策

| 主题 | 决策 | 原因 |
|------|------|------|
| 空间表示 | Cube-Sphere 六面 + 面坐标 | 无极点奇点，面坐标无冗余，Morton 编码天然规整 |
| 大世界坐标 | 世界态恒为 `f64`，渲染前减浮动原点降到 `f32` | 行星尺度下 `f32` 的米级精度不可用；相对相机量级后误差 < 1 mm |
| 可见性 | 曲率遮挡 `bulge = R(1 - cos(θ/2))` | 用解析几何替代体素遍历，AoI 与地平线裁剪共用一个公式 |
| 存档 | 只存差量（种子 + 稀疏编辑） | 存档体积与被改动方块数成正比，与行星大小无关 |
| 崩溃恢复 | 长度前缀的自描述 WAL，撕裂尾帧可丢弃 | 格式必须能被十年后的独立小工具解析 |
| 仿真节奏 | 固定 Tick（20/60 Hz）+ 渲染插值 | 物理/网络可复现；渲染不因帧率波动而抖动 |
| 网络 | 服务端权威 + 客户端预测 + 仅有阈值时才和解 | 无条件重放会让角色在网络抖动下持续抽搐 |
| Mod | `wasmtime` 沙箱，默认零权限，ABI 精确匹配 | 第三方代码不可信；半兼容比不兼容更危险 |

---

## 开发约定

* **文档语言**：代码注释与文档使用中文，标识符与提交信息使用英文；
* **公共项必须有文档**：所有 crate 开启 `#![warn(missing_docs)]`；
* **`unsafe` 仅限两个边界**：`rhi-vulkan`（FFI）与 `world-meshing`（未来 SIMD 路径），
  且全部成员开启 `#![deny(unsafe_op_in_unsafe_fn)]`；
* **可复现性**：任何程序化生成只允许依赖显式传入的种子，禁止时间/地址/线程调度进入结果；
* 提交前跑 `cargo fmt --all` 与 `cargo clippy --workspace --all-targets`。

---

## 当前状态与后续工作

已具备（有测试保障）：

* 世代句柄 / Arena / 微秒 Profiler；
* Cube-Sphere 映射与逆映射、切线标架、局部米制尺度、曲率遮挡与地平线；
* 方块注册表（连续 ID 约束）、材质与物理属性；
* 确定性噪声、球面 Voronoi 板块、连续地层剖面；
* 32³ Chunk 调色板压缩 + 占用位图、Morton 编码、Chunk 池与内存世界视图；
* 贪心网格合并 + 顶点打包 + rayon 并行任务池；
* 差量表、WAL 往返、撕裂写恢复；
* 固定 Tick 时钟、规则调度与冲突检测、Wasm 模块编译与权限模型；
* 四叉树 LOD 选择、近景 Chunk 规划、惰性任务队列；
* 速度/瞬移权威校验、曲率遮挡 AoI、预测缓冲、快照插值与和解。

待实现（已定位的 `TODO`）：

| 位置 | 内容 |
|------|------|
| `rhi-vulkan` | 实例/设备创建、交换链、bindless 表实装、mesh shader 管线 |
| `render-pipeline` | 真实命令录制、实例缓冲环形上传、Hi-Z 遮挡剔除 |
| `geology` | 水文 DAG 侵蚀、沉积物输运、火山口与陨石坑 |
| `streaming` | 按列缓存地表高程、GPU 侧 LOD 反馈 |
| `simulation` | Mod 实例化、燃料计量、宿主函数白名单 |
| `net` | 报文编解码（bincode/postcard）、丢包与乱序测试、房间管理 |
| `apps/client` | 输入 → `ClientInput` 映射、真实呈现 |

## 待决事项

* **许可证冲突（需要拍板）**：仓库根目录的 `LICENSE` 是 **AGPL-3.0**，
  而 `Cargo.toml` 里声明的是 `MIT OR Apache-2.0`。二者不可并存：
  要么把 `[workspace.package] license` 改为 `AGPL-3.0-or-later`，
  要么替换 LICENSE 并补上 `LICENSE-MIT` / `LICENSE-APACHE`。
  在决定之前，**没有**添加 MIT/Apache 许可证文件，以免产生相互矛盾的授权声明。
* 行星半径与体素边长目前是常量（1 m / 6 371 km）；若要做多行星，需把它们
  收敛为启动期配置并贯穿 `planet-coords`。
