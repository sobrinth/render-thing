#![allow(dead_code)]
use crate::input::{ElementState, Key, MouseButton};
use crate::renderer::VulkanRenderer;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
extern crate nalgebra_glm as glm;

mod frame;
mod renderer;
mod resources;

mod camera;
mod command_buffer;
mod context;
mod debug;
mod descriptor;
pub mod input;
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
    pub fn initialize(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        window_size: (u32, u32),
    ) -> Self {
        let renderer = VulkanRenderer::initialize(window, window_size);
        Self { renderer }
    }

    pub fn egui_context(&self) -> egui::Context {
        self.renderer.egui_context()
    }

    pub fn draw(&mut self, raw_input: egui::RawInput) -> egui::PlatformOutput {
        self.renderer.draw(raw_input)
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.renderer.resize(size);
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
