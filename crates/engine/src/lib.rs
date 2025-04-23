use crate::renderer::VulkanRenderer;
use winit::window::Window;

mod renderer;

mod context;
mod debug;
mod descriptor;
mod swapchain;

pub struct Engine {
    renderer: VulkanRenderer,
}

impl Engine {
    pub fn initialize(window: &Window) -> Self {
        let renderer = VulkanRenderer::initialize(window);
        Self { renderer }
    }

    pub fn draw(&mut self) {
        self.renderer.draw();
    }

    pub fn stop(&mut self) {
        self.renderer.wait_gpu_idle();
    }
}
