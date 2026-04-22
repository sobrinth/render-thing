#![allow(dead_code)]
use crate::renderer::VulkanRenderer;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::Key;
use winit::window::Window;
extern crate nalgebra_glm as glm;

mod renderer;

mod camera;
mod command_buffer;
mod context;
mod debug;
mod descriptor;
mod meshes;
mod pipeline;
mod primitives;
mod swapchain;
mod sync;
mod ui;

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

    pub fn resize(&mut self, size: (u32, u32)) {
        self.renderer.resize(size);
    }

    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
        self.renderer.on_window_event(window, event);
    }

    pub fn on_key_press(&mut self, key_event: (ElementState, Key)) {
        self.renderer.on_key_event(key_event);
    }

    pub fn on_mouse_event(&mut self, new_pos: (i32, i32)) {
        self.renderer.on_mouse_event(new_pos);
    }

    pub fn on_mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        self.renderer.on_mouse_button_event(button, state);
    }

    pub fn stop(&mut self) {
        self.renderer.wait_gpu_idle();
    }
}
