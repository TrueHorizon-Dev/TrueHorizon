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
//! 以及把帧统计周期性打到 stdout（作为「渲染管线真的被驱动了」的证据）。
//! TODO(client)：接入 `winit` 输入 → `protocol::ClientInput` → `client-sync` 预测，
//! 以及 `rhi-vulkan` 的真实交换链呈现（当前 `SpherePipeline` 只做状态推进与统计）。

use std::error::Error;

use planet_coords::FloatingOrigin;
use render_contracts::{CameraSnapshot, RenderContract, SceneSnapshot};
use render_pipeline::{PipelineConfig, SpherePipeline};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// 帧统计的输出间隔（帧）。
const STATS_INTERVAL_FRAMES: u64 = 120;

/// 客户端应用状态。
struct ClientApp {
    window: Option<Window>,
    pipeline: SpherePipeline,
    origin: FloatingOrigin,
    frame_index: u64,
    camera_distance_m: f64,
    render_submitted: u64,
}

impl ClientApp {
    fn new() -> Self {
        let origin = FloatingOrigin::new(glam::DVec3::new(6_371_000.0, 0.0, 0.0));
        Self {
            window: None,
            pipeline: SpherePipeline::new(PipelineConfig::default()),
            origin,
            frame_index: 0,
            camera_distance_m: 0.0,
            render_submitted: 0,
        }
    }

    /// 组装并提交一帧。
    fn render_frame(&mut self) {
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
                    println!(
                        "[client] frame={} draw_calls={} instances={} clipmap_patches={} uploaded={} B",
                        stats.frame_index,
                        stats.draw_calls,
                        stats.instance_count,
                        stats.clipmap_patches,
                        stats.uploaded_bytes
                    );
                }
            }
            Err(error) => eprintln!("[client] 帧提交失败: {error}"),
        }
        self.frame_index += 1;
    }
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
            }
            Err(error) => {
                eprintln!("[client] 创建窗口失败: {error}");
                event_loop.exit();
            }
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
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        println!(
            "[client] 退出：共提交 {} 帧，浮动原点搬移 {} 次",
            self.render_submitted,
            self.pipeline.stats().origin_shifts
        );
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
