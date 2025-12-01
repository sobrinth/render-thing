use crate::renderer::AllocatedBuffer;
use ash::vk;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_x: f32,
    pub normal: [f32; 3],
    pub uv_y: f32,
    pub color: [f32; 4],
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct GPUMeshBuffers {
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

#[repr(C)]
pub struct GPUDrawPushConstants {
    world_matrix: [[f32; 4]; 4],
    vertex_buffer: vk::DeviceAddress,
}
