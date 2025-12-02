use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::renderer::VulkanRenderer;
use gltf::accessor::Iter;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeoSurface {
    start_index: u32,
    count: u32,
}

#[derive(Debug)]
pub(crate) struct MeshAsset {
    name: String,
    surfaces: Vec<GeoSurface>,
    mesh_buffers: GPUMeshBuffers, // TODO: Destruction of the buffers :)
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
            // TODO: Figure out why no indices are found haha
            indices.push(1u32);

            let surface = GeoSurface {
                start_index: 0u32,
                count: primitive.indices()?.count() as u32, // Is this correct? I hope so lol
            };
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            if let Some(gltf::mesh::util::ReadIndices::U32(Iter::Standard(iter))) =
                reader.read_indices()
            {
                for v in iter {
                    indices.push(v);
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
                let normal = if normals.is_empty() { [1.0, 0.0, 0.0] } else { normals[i] };
                let color = if colors.is_empty() { [1.0, 1.0, 1.0, 1.0] } else { colors[i] };
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
