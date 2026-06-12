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
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
    /// Offset of this mesh's indices in the renderer's global index pool.
    pub first_index: u32,
}

#[repr(C)]
pub struct GPUDrawPushConstants {
    pub object_buffer: vk::DeviceAddress,
}

/// Must match the push_constant block in assets/shaders/cull.comp.
#[repr(C)]
pub struct GPUCullPushConstants {
    pub view_proj: [[f32; 4]; 4],
    pub object_buffer: vk::DeviceAddress,
    pub command_buffer: vk::DeviceAddress,
    pub count_buffer: vk::DeviceAddress,
    pub draw_count: u32,
    pub _pad: u32,
}

/// Per-draw record read by mesh.vert and cull.comp via buffer reference.
/// Must match ObjectData in assets/shaders/object_structures.glsl.
#[repr(C)]
pub struct GPUObjectData {
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of model's upper 3×3; only that part is meaningful.
    pub normal_matrix: [[f32; 4]; 4],
    /// Local-space AABB center; w unused.
    pub bounds_origin: [f32; 4],
    /// Local-space AABB half-extents; w unused.
    pub bounds_extents: [f32; 4],
    pub vertex_buffer: vk::DeviceAddress,
    pub material_index: u32,
    pub index_count: u32,
    /// Global index-pool offset.
    pub first_index: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(size_of::<GPUObjectData>() == 192);

#[repr(C)]
#[derive(Default)]
pub struct GPUSceneData {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub ambient_color: [f32; 4],
    pub sunlight_direction: [f32; 4],
    pub sunlight_color: [f32; 4],
    pub camera_pos: [f32; 4],
}
