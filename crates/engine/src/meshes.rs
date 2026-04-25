use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::renderer::VulkanRenderer;
use nalgebra_glm as glm;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
}

#[derive(Debug)]
pub(crate) struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: GPUMeshBuffers,
}

fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
    let m = node.transform().matrix(); // [[f32; 4]; 4], column-major (m[col][row])
    glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}

fn scene_roots(document: &gltf::Document) -> Vec<gltf::Node<'_>> {
    if let Some(scene) = document.default_scene() {
        return scene.nodes().collect();
    }
    if let Some(scene) = document.scenes().next() {
        return scene.nodes().collect();
    }
    // No scenes at all — find nodes that are not referenced as any child
    let child_indices: std::collections::HashSet<usize> = document
        .nodes()
        .flat_map(|n| n.children().map(|c| c.index()))
        .collect();
    document
        .nodes()
        .filter(|n| !child_indices.contains(&n.index()))
        .collect()
}

fn collect_mesh_nodes<'a>(
    nodes: impl Iterator<Item = gltf::Node<'a>>,
    parent_world: &glm::Mat4,
    out: &mut Vec<(gltf::Node<'a>, glm::Mat4)>,
) {
    for node in nodes {
        let world = parent_world * node_to_mat4(&node);
        if node.mesh().is_some() {
            out.push((node.clone(), world));
        }
        collect_mesh_nodes(node.children(), &world, out);
    }
}

pub fn load_gltf_meshes<P: AsRef<Path>>(
    renderer: &VulkanRenderer,
    path: P,
) -> Option<Vec<Arc<MeshAsset>>> {
    log::debug!("Loading glTF mesh: {}", path.as_ref().display());

    let (document, buffers, _images) = gltf::import(&path)
        .map_err(|e| {
            log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
        })
        .ok()?;

    let mut meshes = Vec::new();

    let mut mesh_nodes: Vec<(gltf::Node, glm::Mat4)> = Vec::new();
    collect_mesh_nodes(
        scene_roots(&document).into_iter(),
        &glm::Mat4::identity(),
        &mut mesh_nodes,
    );

    for (node, transform) in mesh_nodes {
        let mesh = node.mesh().unwrap(); // safe: collect_mesh_nodes only pushes mesh-bearing nodes

        let name = node
            .name()
            .or_else(|| mesh.name())
            .map(String::from)
            .unwrap_or_else(|| format!("node_{}", node.index()));
        let mut surfaces = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let normal_matrix = glm::transpose(&glm::inverse(&transform));

        for primitive in mesh.primitives() {
            let surface = GeoSurface {
                start_index: indices.len() as u32,
                count: primitive.indices()?.count() as u32,
            };
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            match reader.read_indices() {
                Some(read_indices) => {
                    indices.append(&mut read_indices.into_u32().collect());
                }
                None => {
                    log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
                    continue;
                }
            }

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|p| p.collect())
                .unwrap_or_default();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_default();

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_default();

            let colors: Vec<[f32; 4]> = reader
                .read_colors(0)
                .map(|tc| tc.into_rgba_f32().collect())
                .unwrap_or_default();

            positions.iter().enumerate().for_each(|(i, v)| {
                let normal = if normals.is_empty() {
                    [1.0_f32, 0.0, 0.0]
                } else {
                    normals[i]
                };
                let color = if colors.is_empty() {
                    [1.0_f32, 1.0, 1.0, 1.0]
                } else {
                    colors[i]
                };
                let (uv_x, uv_y) = uvs
                    .get(i)
                    .copied()
                    .map(|v| (v[0], v[1]))
                    .unwrap_or((0.0, 0.0));

                let world_pos = transform * glm::vec4(v[0], v[1], v[2], 1.0);
                let world_normal_raw =
                    normal_matrix * glm::vec4(normal[0], normal[1], normal[2], 0.0);
                let world_normal = glm::normalize(&world_normal_raw.xyz());

                let vtx = Vertex {
                    position: [world_pos.x, world_pos.y, world_pos.z],
                    normal: [world_normal.x, world_normal.y, world_normal.z],
                    color,
                    uv_x,
                    uv_y,
                };
                vertices.push(vtx);
            });
            surfaces.push(surface);
        }

        // display the vertex normals
        const OVERRIDE_COLORS: bool = false;
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
        Some(meshes.into_iter().map(Arc::new).collect())
    }
}
