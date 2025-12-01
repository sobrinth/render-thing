use crate::renderer::AllocatedBuffer;
use ash::vk;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    uv_x: f32,
    normal: [f32; 3],
    uv_y: f32,
    color: [f32; 4],
}

#[derive(Debug)]
pub struct GPUMeshBuffers {
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_buffer_address: vk::DeviceAddress,
}

#[repr(C)]
pub struct GPUDrawPushConstants {
    world_matrix: [[f32; 4]; 4],
    vertex_buffer: vk::DeviceAddress,
}
