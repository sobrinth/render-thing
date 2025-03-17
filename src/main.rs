mod engine;

use crate::engine::renderer::*;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct App {
    vulkan: Option<VulkanApplication>,
    // window needs to be dropped last as Vulkan has a window reference
    window: Option<Window>,
}

struct VulkanApplication {
    renderer: VulkanRenderer,
}

impl VulkanApplication {
    fn start(window: &Window) -> Self {
        let renderer = VulkanRenderer::initialize(window);
        Self { renderer }
    }

    fn draw(&mut self) {
        self.renderer.draw();
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        if let Some(_app) = self.vulkan.as_mut() {
            // app.wheel_delta = None;
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan App with Ash")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.vulkan = Some(VulkanApplication::start(&window));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized { .. } => {
                // self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { .. } => {
                // self.vulkan.as_mut().unwrap().is_left_clicked =
                //     state == ElementState::Pressed && button == MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let _app = self.vulkan.as_mut().unwrap();

                let _position: (i32, i32) = position.into();
                // app.cursor_delta = Some([
                //     app.cursor_position[0] - position.0,
                //     app.cursor_position[1] - position.1,
                // ]);
                // app.cursor_position = [position.0, position.1];
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, _v_lines),
                ..
            } => {
                // self.vulkan.as_mut().unwrap().wheel_delta = Some(v_lines);
            }
            _ => (),
        }
    }

    /// This is not the ideal place to drive rendering from.
    /// Should really be done with the RedrawRequested Event, but here we are for now.
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let app = self.vulkan.as_mut().unwrap();
        let _window = self.window.as_ref().unwrap();

        // TODO: db: This will run as fast as possible! Not good.
        app.draw();

        // if app.dirty_swapchain {
        //     let size = window.inner_size();
        //     if size.width > 0 && size.height > 0 {
        //         // app.recreate_swapchain();
        //     } else {
        //         return;
        //     }
        // }
        // app.dirty_swapchain = app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.vulkan.as_ref().unwrap().renderer.wait_gpu_idle();
    }
}
