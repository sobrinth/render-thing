use crate::graph::{NodeId, SceneGraph};
use engine::Engine;
use nalgebra_glm as glm;

pub fn load_gltf(engine: &mut Engine, path: &str) -> Option<SceneGraph> {
    let (doc, buffers, images) = gltf::import(path).ok()?;

    let mut texture_handles: Vec<engine::TextureHandle> = Vec::new();
    for image in &images {
        let data: Vec<u8> = match image.format {
            gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
            gltf::image::Format::R8G8B8 => image
                .pixels
                .chunks(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                .collect(),
            _ => {
                texture_handles.push(engine.white_texture());
                continue;
            }
        };
        let handle = engine.upload_texture(
            &data,
            image.width,
            image.height,
            engine::SamplerType::Linear,
        );
        texture_handles.push(handle);
    }

    let white = engine.white_texture();

    let mut material_handles: Vec<engine::MaterialHandle> = Vec::new();
    for material in doc.materials() {
        let pbr = material.pbr_metallic_roughness();
        let color_tex = pbr
            .base_color_texture()
            .and_then(|t| texture_handles.get(t.texture().source().index()).copied())
            .unwrap_or(white);
        let mr_tex = pbr
            .metallic_roughness_texture()
            .and_then(|t| texture_handles.get(t.texture().source().index()).copied())
            .unwrap_or(white);
        let [r, g, b, a] = pbr.base_color_factor();
        let mut constants = engine::MaterialConstants::default();
        constants.color_factors = [r, g, b, a];
        constants.metal_rough_factors = [pbr.metallic_factor(), pbr.roughness_factor(), 0.0, 0.0];
        let pass = if material.alpha_mode() == gltf::material::AlphaMode::Blend {
            engine::MaterialPass::Transparent
        } else {
            engine::MaterialPass::MainColor
        };
        material_handles.push(engine.create_material(color_tex, mr_tex, constants, pass));
    }

    let default_material = engine.create_material(
        white,
        white,
        engine::MaterialConstants::default(),
        engine::MaterialPass::MainColor,
    );

    let mut scene = SceneGraph::new();

    for gltf_scene in doc.scenes() {
        for node in gltf_scene.nodes() {
            traverse(
                &mut scene,
                None,
                &node,
                &buffers,
                engine,
                &material_handles,
                default_material,
            );
        }
    }

    Some(scene)
}

// gltf's transform().matrix() returns [[f32; 4]; 4] in column-major order (m[col][row]).
// glm::Mat4::from_column_slice expects a flat column-major slice, so we read columns directly.
fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
    let m = node.transform().matrix();
    glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}

fn traverse(
    scene: &mut SceneGraph,
    parent: Option<NodeId>,
    gltf_node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    engine: &mut Engine,
    material_handles: &[engine::MaterialHandle],
    default_material: engine::MaterialHandle,
) {
    let local = node_to_mat4(gltf_node);
    let name = gltf_node.name().unwrap_or("node").to_string();

    let node_id = match parent {
        Some(p) => scene.add_child(p, &name, local),
        None => scene.add_root(&name, local),
    };

    if let Some(mesh) = gltf_node.mesh() {
        let prim_count = mesh.primitives().count();
        let mesh_name = mesh.name().unwrap_or("mesh").to_string();

        for (prim_idx, prim) in mesh.primitives().enumerate() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|p| p.collect())
                .unwrap_or_default();
            if positions.is_empty() {
                continue;
            }

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|u| u.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let colors: Vec<[f32; 4]> = reader
                .read_colors(0)
                .map(|c| c.into_rgba_f32().collect())
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

            let vertices: Vec<engine::Vertex> = positions
                .iter()
                .enumerate()
                .map(|(i, pos)| engine::Vertex {
                    position: *pos,
                    uv_x: uvs[i][0],
                    normal: normals[i],
                    uv_y: uvs[i][1],
                    color: colors[i],
                })
                .collect();

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|idx| idx.into_u32().collect())
                .unwrap_or_default();
            if indices.is_empty() {
                continue;
            }

            let mesh_handle = engine.upload_mesh(&indices, &vertices);
            let mat_handle = prim
                .material()
                .index()
                .and_then(|i| material_handles.get(i).copied())
                .unwrap_or(default_material);

            let target_node = if prim_count == 1 {
                node_id
            } else {
                scene.add_child(
                    node_id,
                    format!("{mesh_name}_prim{prim_idx}"),
                    glm::Mat4::identity(),
                )
            };

            scene.node_mut(target_node).mesh = Some(mesh_handle);
            scene.node_mut(target_node).material = Some(mat_handle);
        }
    }

    for child in gltf_node.children() {
        traverse(
            scene,
            Some(node_id),
            &child,
            buffers,
            engine,
            material_handles,
            default_material,
        );
    }
}
