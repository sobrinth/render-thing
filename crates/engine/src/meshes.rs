use crate::primitives::GPUMeshBuffers;
use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Bounds {
    pub origin: glm::Vec3,
    pub extents: glm::Vec3,
    pub sphere_radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
    pub bounds: Bounds,
}

#[derive(Debug)]
pub(crate) struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: GPUMeshBuffers,
}

pub(crate) fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
    let m = node.transform().matrix(); // [[f32; 4]; 4], column-major (m[col][row])
    glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}

