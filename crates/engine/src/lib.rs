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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshHandle(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureHandle(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialHandle(pub(crate) u32);

#[derive(Debug, Clone, Copy)]
pub struct DrawCall {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub transform: glm::Mat4,
}

pub struct GltfScene {
    pub draws: Vec<DrawCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    Linear,
    Nearest,
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
mod stats;
mod swapchain;
mod sync;
mod ui;

pub use material::{MaterialConstants, MaterialPass};
pub use primitives::Vertex;

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

    pub fn load_gltf(&mut self, path: &str) -> Option<GltfScene> {
        crate::gltf_scene::load_gltf_scene(&mut self.renderer, path)
    }

    pub fn upload_mesh(&mut self, indices: &[u32], vertices: &[Vertex]) -> MeshHandle {
        self.renderer.upload_mesh(indices, vertices)
    }

    pub fn upload_texture(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        sampler: SamplerType,
    ) -> TextureHandle {
        self.renderer.upload_texture(data, width, height, sampler)
    }

    pub fn create_material(
        &mut self,
        color: TextureHandle,
        metal_rough: TextureHandle,
        constants: MaterialConstants,
        pass: MaterialPass,
    ) -> MaterialHandle {
        self.renderer
            .create_material(color, metal_rough, constants, pass)
    }

    pub fn white_texture(&self) -> TextureHandle {
        self.renderer.resources.default_white_texture
    }

    pub fn grey_texture(&self) -> TextureHandle {
        self.renderer.resources.default_grey_texture
    }

    pub fn black_texture(&self) -> TextureHandle {
        self.renderer.resources.default_black_texture
    }

    pub fn checkerboard_texture(&self) -> TextureHandle {
        self.renderer.resources.default_checkerboard_texture
    }

    pub fn draw(
        &mut self,
        camera: CameraView,
        draws: &[DrawCall],
        raw_input: egui::RawInput,
    ) -> egui::PlatformOutput {
        self.renderer.draw(camera, draws, raw_input)
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
