use crate::resources::AllocatedBuffer;
use ash::vk;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_x: f32,
    pub normal: [f32; 3],
    pub uv_y: f32,
    pub color: [f32; 4],
}

#[derive(Debug)]
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

#[repr(C)]
#[derive(Default)]
pub struct GPUSceneData {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub ambient_color: [f32; 4],
    pub sunlight_direction: [f32; 4],
    pub sunlight_color: [f32; 4],
}
