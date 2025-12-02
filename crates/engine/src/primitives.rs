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
impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            uv_x: 0.0,
            normal: [0.0, 0.0, 0.0],
            uv_y: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
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
    pub world_matrix: [[f32; 4]; 4],
    pub vertex_buffer: vk::DeviceAddress,
}
