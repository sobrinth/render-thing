#![allow(dead_code)]
use crate::input::{ElementState, Key};
use crate::renderer::VulkanRenderer;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
extern crate nalgebra_glm as glm;

pub struct CameraView {
    pub view_matrix: glm::Mat4,
    pub proj_matrix: glm::Mat4,
    pub position: glm::Vec3,
}

pub struct Scene(pub(crate) Vec<Box<dyn crate::scene::Renderable>>);

impl Scene {
    pub fn empty() -> Self {
        Scene(Vec::new())
    }
}

mod frame;
mod renderer;
mod resources;

mod command_buffer;
mod context;
mod debug;
mod descriptor;
mod gltf_scene;
pub mod input;
mod material;
mod meshes;
mod pipeline;
mod primitives;
mod scene;
mod stats;
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

    pub fn load_gltf(&mut self, path: &str) -> Option<Scene> {
        crate::gltf_scene::Gltf::load(&self.renderer, path).map(|gltf| Scene(vec![Box::new(gltf)]))
    }

    pub fn draw(
        &mut self,
        camera: CameraView,
        scene: &Scene,
        raw_input: egui::RawInput,
    ) -> egui::PlatformOutput {
        self.renderer.draw(camera, scene, raw_input)
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.renderer.resize(size);
    }

    pub fn on_key_press(&mut self, key_event: (ElementState, Key)) {
        self.renderer.on_key_event(key_event);
    }

    pub fn stop(&mut self) {
        self.renderer.wait_gpu_idle();
    }
}
