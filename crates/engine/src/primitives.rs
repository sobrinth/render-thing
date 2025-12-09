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

#[repr(C)]
pub struct GPUSceneData {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub ambient_color: [f32; 4],
    pub sunlight_direction: [f32; 4],
    pub sunlight_color: [f32; 4],
}

impl GPUSceneData {
    pub fn default() -> Self {
        Self {
            view: [[0.0; 4]; 4],
            proj: [[0.0; 4]; 4],
            view_proj: [[0.0; 4]; 4],
            ambient_color: [0.0, 0.0, 0.0, 0.0],
            sunlight_direction: [0.0, 0.0, 0.0, 0.0],
            sunlight_color: [0.0, 0.0, 0.0, 0.0],
        }
    }
}
