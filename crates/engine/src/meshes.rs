use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::renderer::VulkanRenderer;
use gltf::accessor::Iter;
use std::path::Path;
use std::sync::Arc;
use vk_mem::Allocator;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: GPUMeshBuffers,
}

impl MeshAsset {
    pub fn destroy(&mut self, gpu_alloc: &Arc<Allocator>) {
        self.mesh_buffers.index_buffer.destroy(gpu_alloc);
        self.mesh_buffers.vertex_buffer.destroy(gpu_alloc);
    }
}

pub fn load_gltf_meshes<P: AsRef<Path>>(
    renderer: &VulkanRenderer,
    path: P,
) -> Option<Vec<MeshAsset>> {
    log::debug!("Loading glTF mesh: {}", path.as_ref().display());

    let (document, buffers, _images) = gltf::import(path).ok()?;

    let mut meshes = Vec::new();

    for mesh in document.meshes() {
        let name = String::from(mesh.name()?);
        let mut surfaces = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for primitive in mesh.primitives() {
            vertices.clear();
            indices.clear();

            let surface = GeoSurface {
                start_index: 0u32,
                count: primitive.indices()?.count() as u32, // Is this correct? I hope so lol
            };
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            if let Some(gltf::mesh::util::ReadIndices::U16(Iter::Standard(iter))) =
                reader.read_indices()
            {
                for v in iter {
                    indices.push(v as u32);
                }
            }

            let mut positions = Vec::new();
            if let Some(iter) = reader.read_positions() {
                vertices.reserve(iter.len());
                for v in iter {
                    positions.push(v);
                }
            }

            let mut normals = Vec::new();
            if let Some(iter) = reader.read_normals() {
                for v in iter {
                    normals.push(v);
                }
            }

            let mut uvs = Vec::new();
            if let Some(gltf::mesh::util::ReadTexCoords::F32(Iter::Standard(iter))) =
                reader.read_tex_coords(1)
            {
                for v in iter {
                    uvs.push(v);
                }
            }

            let mut colors = Vec::new();
            if let Some(gltf::mesh::util::ReadColors::RgbaF32(Iter::Standard(iter))) =
                reader.read_colors(0)
            {
                for v in iter {
                    colors.push(v);
                }
            }

            positions.iter().enumerate().for_each(|(i, v)| {
                let normal = if normals.is_empty() {
                    [1.0, 0.0, 0.0]
                } else {
                    normals[i]
                };
                let color = if colors.is_empty() {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    colors[i]
                };
                let vtx = Vertex {
                    position: *v,
                    normal,
                    color,
                    uv_x: 0.0,
                    uv_y: 0.0,
                };
                vertices.push(vtx);
            });
            surfaces.push(surface);
        }

        // display the vertex normals
        const OVERRIDE_COLORS: bool = true;
        if OVERRIDE_COLORS {
            for v in &mut vertices {
                v.color = [v.normal[0], v.normal[1], v.normal[2], 1.0];
            }
        }
        let mesh = renderer.upload_mesh(indices.as_slice(), vertices.as_slice());
        let mesh_asset = MeshAsset {
            surfaces,
            name,
            mesh_buffers: mesh,
        };
        meshes.push(mesh_asset);
    }
    if meshes.is_empty() {
        None
    } else {
        Some(meshes)
    }
}
