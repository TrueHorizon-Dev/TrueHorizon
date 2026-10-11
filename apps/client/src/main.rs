//! # TrueHorizon 桌面客户端
//!
//! 职责边界：**窗口、输入、帧编排**。渲染细节在 `render-pipeline`，世界在 `world/*`。
//!
//! 帧循环（`about_to_wait` → `RedrawRequested`）：
//!
//! ```text
//! 输入采集 ──► 仿真推进（固定 Tick）──► 组装 SceneSnapshot ──► 提交渲染 ──► 请求下一帧
//! ```
//!
//! 当前骨架已完成：窗口生命周期、Escape 退出、逐帧快照组装与渲染提交、
//! 把帧统计周期性打到 stdout（作为「渲染管线真的被驱动了」的证据），
//! 以及 **Vulkan 表面创建 + renet 客户端接入**（`--server IP:PORT` 时连服）。
//! TODO(client)：窗口输入 → `protocol::ClientInput` 的按键映射，
//! 以及把快照接进 `client-sync` 的预测/和解闭环（当前只做解码与统计）。

mod net;

use std::error::Error;
use std::time::Instant;

use net::NetSession;
use planet_coords::FloatingOrigin;
use render_contracts::{CameraSnapshot, RenderContract, SceneSnapshot};
use render_pipeline::{PipelineConfig, SpherePipeline};
use rhi_vulkan::{FeatureRequirements, VulkanBackend};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

/// 帧统计的输出间隔（帧）。
const STATS_INTERVAL_FRAMES: u64 = 120;

/// 客户端应用状态。
struct ClientApp {
    window: Option<Window>,
    pipeline: SpherePipeline,
    /// 真实 Vulkan 后端（无 GPU / 无 ICD 时为 `None`，进程照常运行）。
    rhi: Option<VulkanBackend>,
    /// 网络会话（未给 `--server` 时为 `None`，即单机模式）。
    net: Option<NetSession>,
    net_messages: u64,
    last_snapshot_tick: Option<u64>,
    origin: FloatingOrigin,
    frame_index: u64,
    camera_distance_m: f64,
    render_submitted: u64,
    last_frame: Instant,
}

impl ClientApp {
    fn new() -> Self {
        let origin = FloatingOrigin::new(glam::DVec3::new(6_371_000.0, 0.0, 0.0));
        let net = match server_argument() {
            Some(server) => match NetSession::connect(&server, 1) {
                Ok(session) => {
                    println!(
                        "[client] 连接服务端 {}（netcode 握手进行中）",
                        session.server()
                    );
                    Some(session)
                },
                Err(error) => {
                    eprintln!("[client] 连接 {server} 失败: {error}");
                    None
                },
            },
            None => None,
        };
        Self {
            window: None,
            pipeline: SpherePipeline::new(PipelineConfig::default()),
            rhi: None,
            net,
            net_messages: 0,
            last_snapshot_tick: None,
            origin,
            frame_index: 0,
            camera_distance_m: 0.0,
            render_submitted: 0,
            last_frame: Instant::now(),
        }
    }

    /// 组装并提交一帧。
    fn render_frame(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame);
        self.last_frame = now;

        // 网络：输入上行（当前只上行 IDLE 与视角）、快照下行。
        if let Some(session) = &mut self.net {
            match session.update(
                dt,
                // TODO(input): 由窗口按键状态填充移动轴。
                [0.0, 0.0, 0.0],
                [0.0, 0.0],
            ) {
                Ok(messages) => {
                    self.net_messages += messages.len() as u64;
                    if let Some(snapshot) = session.latest_snapshot() {
                        self.last_snapshot_tick = Some(snapshot.tick);
                    }
                },
                Err(error) => eprintln!("[client] 网络更新失败: {error}"),
            }
        }

        // 骨架阶段：相机沿径向缓慢抬升，用于验证浮动原点与截锥参数不会数值爆炸。
        self.camera_distance_m += 0.5;
        let camera = CameraSnapshot {
            position_world_m: self.origin.offset_m() * (1.0 + self.camera_distance_m / 6_371_000.0),
            height_m: self.camera_distance_m,
            ..CameraSnapshot::default()
        };

        let mut snapshot = SceneSnapshot::new(self.frame_index, self.origin, camera);
        // 近景由 `world-meshing` 填充，宏观由 clipmap 生成——此处先留空，
        // 以便把「空场景也能跑通整条提交链路」纳入回归范围。
        snapshot.clear();

        match self.pipeline.submit(&snapshot) {
            Ok(()) => {
                self.render_submitted += 1;
                let stats = self.pipeline.stats();
                if self.frame_index % STATS_INTERVAL_FRAMES == 0 {
                    let net = match &self.net {
                        Some(session) => format!(
                            " | net={} msgs={} server_tick={:?}",
                            if session.is_connected() {
                                "connected"
                            } else {
                                "connecting"
                            },
                            self.net_messages,
                            self.last_snapshot_tick
                        ),
                        None => String::new(),
                    };
                    println!(
                        "[client] frame={} draw_calls={} instances={} clipmap_patches={} uploaded={} B{}",
                        stats.frame_index,
                        stats.draw_calls,
                        stats.instance_count,
                        stats.clipmap_patches,
                        stats.uploaded_bytes,
                        net
                    );
                }
            },
            Err(error) => eprintln!("[client] 帧提交失败: {error}"),
        }
        self.frame_index += 1;
    }
}

/// 解析 `--server IP:PORT`。
fn server_argument() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--server" {
            return args.next();
        }
    }
    None
}

impl ApplicationHandler for ClientApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = Window::default_attributes()
            .with_title("TrueHorizon")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        match event_loop.create_window(attributes) {
            Ok(window) => {
                println!("[client] 窗口已创建：{}x720", window.inner_size().width);
                self.window = Some(window);
                self.create_vulkan_surface();
            },
            Err(error) => {
                eprintln!("[client] 创建窗口失败: {error}");
                event_loop.exit();
            },
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => self.render_frame(),
            WindowEvent::Resized(_) => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            },
            _ => {},
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(session) = &mut self.net {
            session.disconnect();
        }
        println!(
            "[client] 退出：共提交 {} 帧，浮动原点搬移 {} 次，Vulkan {}，网络报文 {}",
            self.render_submitted,
            self.pipeline.stats().origin_shifts,
            self.rhi
                .as_ref()
                .map(|backend| backend.info().name.as_str())
                .unwrap_or("未启用"),
            self.net_messages
        );
    }
}

impl ClientApp {
    /// 为窗口创建真实 Vulkan 后端（含 `VkSurfaceKHR`）。
    ///
    /// 失败**只降级**：无 GPU / 无 ICD / 只有 loader 的机器仍然能跑客户端帧编排，
    /// 这条路径同时是 `world-tool diagnose` 之外的第二处 RHI 现场取证。
    fn create_vulkan_surface(&mut self) {
        let handles = {
            let Some(window) = &self.window else {
                return;
            };
            match (window.display_handle(), window.window_handle()) {
                (Ok(display), Ok(handle)) => Some((display.as_raw(), handle.as_raw())),
                _ => None,
            }
        };
        let Some((display_handle, window_handle)) = handles else {
            eprintln!("[client] 拿不到平台窗口句柄，跳过 Vulkan 表面创建");
            return;
        };
        match VulkanBackend::new_for_window(
            &FeatureRequirements::minimum(),
            display_handle,
            window_handle,
        ) {
            Ok(backend) => {
                println!(
                    "[client] Vulkan 就绪：{} | Vulkan {} | surface 已创建 | 队列族 {:?}",
                    backend.info().name,
                    backend.info().api_version_string(),
                    backend.queues()
                );
                self.rhi = Some(backend);
            },
            Err(error) => {
                eprintln!("[client] GPU 路径不可用（降级为纯 CPU 帧编排）: {error}")
            },
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    // 游戏客户端需要连续重绘，不能被系统的事件合并策略挂起。
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = ClientApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
