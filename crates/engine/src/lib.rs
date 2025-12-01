use crate::renderer::VulkanRenderer;
use winit::event::WindowEvent;
use winit::window::Window;

mod renderer;

mod context;
mod debug;
mod descriptor;
mod pipeline;
mod swapchain;
mod ui;
mod primitives;

pub struct Engine {
    renderer: VulkanRenderer,
}

impl Engine {
    pub fn initialize(window: &Window) -> Self {
        let renderer = VulkanRenderer::initialize(window);
        Self { renderer }
    }

    pub fn draw(&mut self, window: &Window) {
        self.renderer.draw(window);
    }

    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
        self.renderer.on_window_event(window, event);
    }

    pub fn stop(&mut self) {
        self.renderer.wait_gpu_idle();
    }
}
