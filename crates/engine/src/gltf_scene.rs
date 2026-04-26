use nalgebra_glm as glm;

pub(crate) fn load_gltf_scene(
    renderer: &mut crate::renderer::VulkanRenderer,
    path: impl AsRef<std::path::Path>,
) -> Option<crate::GltfScene> {
    let (doc, buffers, images) = gltf::import(path).ok()?;

    // 1. Upload textures
    let mut texture_handles: Vec<crate::TextureHandle> = Vec::new();
    for image in &images {
        let data: Vec<u8> = match image.format {
            gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
            gltf::image::Format::R8G8B8 => image
                .pixels
                .chunks(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                .collect(),
            _ => {
                texture_handles.push(renderer.resources.default_white_texture);
                continue;
            }
        };
        let handle =
            renderer.upload_texture(&data, image.width, image.height, crate::SamplerType::Linear);
        texture_handles.push(handle);
    }

    let white = renderer.resources.default_white_texture;

    // 2. Upload materials
    let mut material_handles: Vec<crate::MaterialHandle> = Vec::new();
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
        let mut constants = crate::material::MaterialConstants::default();
        constants.color_factors = [r, g, b, a];
        constants.metal_rough_factors = [pbr.metallic_factor(), pbr.roughness_factor(), 0.0, 0.0];
        let pass = if material.alpha_mode() == gltf::material::AlphaMode::Blend {
            crate::material::MaterialPass::Transparent
        } else {
            crate::material::MaterialPass::MainColor
        };
        material_handles.push(renderer.create_material(color_tex, mr_tex, constants, pass));
    }

    let default_material = renderer.create_material(
        white,
        white,
        crate::material::MaterialConstants::default(),
        crate::material::MaterialPass::MainColor,
    );

    // 3. Traverse scene graph, upload meshes, collect draw calls
    let mut draw_calls: Vec<crate::DrawCall> = Vec::new();

    fn traverse(
        node: &gltf::Node,
        parent_transform: &glm::Mat4,
        buffers: &[gltf::buffer::Data],
        renderer: &mut crate::renderer::VulkanRenderer,
        material_handles: &[crate::MaterialHandle],
        default_material: crate::MaterialHandle,
        draw_calls: &mut Vec<crate::DrawCall>,
    ) {
        let local = crate::meshes::node_to_mat4(node);
        let world = parent_transform * local;

        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
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

                let vertices: Vec<crate::primitives::Vertex> = positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| crate::primitives::Vertex {
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

                let mesh_handle = renderer.upload_mesh(&indices, &vertices);
                let mat_handle = prim
                    .material()
                    .index()
                    .and_then(|i| material_handles.get(i).copied())
                    .unwrap_or(default_material);

                draw_calls.push(crate::DrawCall {
                    mesh: mesh_handle,
                    material: mat_handle,
                    transform: world,
                });
            }
        }

        for child in node.children() {
            traverse(
                &child,
                &world,
                buffers,
                renderer,
                material_handles,
                default_material,
                draw_calls,
            );
        }
    }

    for scene in doc.scenes() {
        for node in scene.nodes() {
            traverse(
                &node,
                &glm::Mat4::identity(),
                &buffers,
                renderer,
                &material_handles,
                default_material,
                &mut draw_calls,
            );
        }
    }

    Some(crate::GltfScene { draws: draw_calls })
}
