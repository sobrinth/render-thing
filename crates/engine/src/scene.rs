use crate::material::{MaterialInstance, MaterialPass};
use crate::meshes::{GeoSurface, MeshAsset};
use ash::vk;
use nalgebra_glm as glm;
use std::sync::Arc;

pub(crate) trait Renderable {
    fn draw(&self, top_matrix: &glm::Mat4, ctx: &mut DrawContext);
}

#[derive(Default)]
pub(crate) struct DrawContext {
    pub opaque_surfaces: Vec<RenderObject>,
    pub transparent_surfaces: Vec<RenderObject>,
}

pub(crate) struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: Arc<MaterialInstance>,
    pub transform: glm::Mat4,
    pub vertex_buffer_address: vk::DeviceAddress,
}

pub(crate) struct Node {
    pub children: Vec<Box<dyn Renderable>>,
    pub local_transform: glm::Mat4,
}

impl Node {
    pub(crate) fn new(local_transform: glm::Mat4) -> Self {
        Self {
            children: Vec::new(),
            local_transform,
        }
    }
}

pub(crate) struct MeshSurface {
    pub geo: GeoSurface,
    pub material: Arc<MaterialInstance>,
}

pub(crate) struct MeshNode {
    pub node: Node,
    pub mesh: Arc<MeshAsset>,
    pub surfaces: Vec<MeshSurface>,
}

impl Renderable for MeshNode {
    fn draw(&self, top_matrix: &glm::Mat4, ctx: &mut DrawContext) {
        let world_transform = top_matrix * self.node.local_transform;

        for surface in &self.surfaces {
            let obj = RenderObject {
                index_count: surface.geo.count,
                first_index: surface.geo.start_index,
                index_buffer: self.mesh.mesh_buffers.index_buffer.buffer,
                material: Arc::clone(&surface.material),
                transform: world_transform,
                vertex_buffer_address: self.mesh.mesh_buffers.vertex_buffer_address,
            };
            match surface.material.pass_type {
                MaterialPass::MainColor => ctx.opaque_surfaces.push(obj),
                MaterialPass::Transparent => ctx.transparent_surfaces.push(obj),
            }
        }

        for child in &self.node.children {
            child.draw(&world_transform, ctx);
        }
    }
}
