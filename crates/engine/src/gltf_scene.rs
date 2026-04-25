use crate::descriptor::GrowableAllocator;
use crate::material::MaterialInstance;
use crate::meshes::MeshAsset;
use crate::resources::{AllocatedBuffer, AllocatedImage, Sampler};
use crate::scene::{DrawContext, Renderable};
use ash::Device;
use ash::vk;
use nalgebra_glm as glm;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use vk_mem;

pub(crate) struct Gltf {
    pub meshes: HashMap<String, Arc<MeshAsset>>,
    pub images: Vec<Arc<AllocatedImage>>,
    pub samplers: Vec<Sampler>,
    pub materials: HashMap<String, Arc<MaterialInstance>>,
    pub top_nodes: Vec<Box<dyn Renderable>>,
    pub material_data_buffer: AllocatedBuffer,
    pub descriptor_pool: GrowableAllocator,
    device: Device,
}

impl Renderable for Gltf {
    fn draw(&self, top_matrix: &glm::Mat4, ctx: &mut DrawContext) {
        for node in &self.top_nodes {
            node.draw(top_matrix, ctx);
        }
    }
}

impl Drop for Gltf {
    fn drop(&mut self) {
        self.descriptor_pool.destroy_pools(&self.device);
    }
}

fn gltf_filter_to_vk(filter: Option<gltf::texture::MinFilter>) -> vk::Filter {
    match filter {
        Some(gltf::texture::MinFilter::Nearest)
        | Some(gltf::texture::MinFilter::NearestMipmapNearest)
        | Some(gltf::texture::MinFilter::NearestMipmapLinear) => vk::Filter::NEAREST,
        _ => vk::Filter::LINEAR,
    }
}

fn gltf_mag_filter_to_vk(filter: Option<gltf::texture::MagFilter>) -> vk::Filter {
    match filter {
        Some(gltf::texture::MagFilter::Nearest) => vk::Filter::NEAREST,
        _ => vk::Filter::LINEAR,
    }
}

fn gltf_mipmap_mode(filter: Option<gltf::texture::MinFilter>) -> vk::SamplerMipmapMode {
    match filter {
        Some(gltf::texture::MinFilter::NearestMipmapNearest)
        | Some(gltf::texture::MinFilter::LinearMipmapNearest) => vk::SamplerMipmapMode::NEAREST,
        _ => vk::SamplerMipmapMode::LINEAR,
    }
}

fn gltf_wrap_to_vk(wrap: gltf::texture::WrappingMode) -> vk::SamplerAddressMode {
    match wrap {
        gltf::texture::WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

impl Gltf {
    pub(crate) fn load(
        renderer: &crate::renderer::VulkanRenderer,
        path: impl AsRef<Path>,
    ) -> Option<Self> {
        use crate::descriptor::{GrowableAllocator, PoolSizeRatio};
        use crate::material::{MaterialConstants, MaterialPass, MaterialResources};
        use crate::resources::ImageCreateInfo;

        let res = &*renderer.resources;
        let ctx = &*renderer.context;

        let (document, buffers, raw_images) = gltf::import(&path)
            .map_err(|e| log::error!("Failed to load GLTF '{}': {e}", path.as_ref().display()))
            .ok()?;
        log::info!(
            "GLTF '{}': {} meshes, {} materials, {} images, {} nodes, {} scenes",
            path.as_ref().display(),
            document.meshes().count(),
            document.materials().count(),
            raw_images.len(),
            document.nodes().count(),
            document.scenes().count(),
        );

        // ── 1. Samplers ──────────────────────────────────────────────────────────
        let samplers: Vec<Sampler> = document
            .samplers()
            .map(|s| {
                let info = vk::SamplerCreateInfo::default()
                    .mag_filter(gltf_mag_filter_to_vk(s.mag_filter()))
                    .min_filter(gltf_filter_to_vk(s.min_filter()))
                    .mipmap_mode(gltf_mipmap_mode(s.min_filter()))
                    .address_mode_u(gltf_wrap_to_vk(s.wrap_s()))
                    .address_mode_v(gltf_wrap_to_vk(s.wrap_t()))
                    .address_mode_w(vk::SamplerAddressMode::REPEAT);
                let handle = unsafe { ctx.device.create_sampler(&info, None) }.unwrap();
                Sampler::new(handle, ctx.device.clone())
            })
            .collect();

        // ── 2. Images ────────────────────────────────────────────────────────────
        let images: Vec<Arc<AllocatedImage>> = raw_images
            .iter()
            .map(|img_data| {
                // Expand all 8-bit sub-RGBA formats to R8G8B8A8 before upload.
                let rgba: Vec<u8> = match img_data.format {
                    gltf::image::Format::R8 => img_data
                        .pixels
                        .iter()
                        .flat_map(|&r| [r, r, r, 255u8])
                        .collect(),
                    gltf::image::Format::R8G8 => img_data
                        .pixels
                        .chunks_exact(2)
                        .flat_map(|c| [c[0], c[1], 0u8, 255u8])
                        .collect(),
                    gltf::image::Format::R8G8B8 => img_data
                        .pixels
                        .chunks_exact(3)
                        .flat_map(|c| [c[0], c[1], c[2], 255u8])
                        .collect(),
                    _ => img_data.pixels.clone(),
                };
                Arc::new(AllocatedImage::create_from_bytes(
                    &res.gpu_alloc,
                    ctx,
                    &res.immediate_submit,
                    &res.graphics_queue,
                    &rgba,
                    ImageCreateInfo {
                        resolution: (img_data.width, img_data.height),
                        format: vk::Format::R8G8B8A8_UNORM,
                        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        mip_mapped: false,
                    },
                ))
            })
            .collect();

        // ── 3. Materials ─────────────────────────────────────────────────────────
        let mat_count = document.materials().count().max(1);
        let mat_stride = std::mem::size_of::<MaterialConstants>();

        let material_data_buffer = AllocatedBuffer::create(
            &res.gpu_alloc,
            (mat_count * mat_stride) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
            Some(
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
            ),
        );

        let mut descriptor_pool = GrowableAllocator::init(
            &ctx.device,
            mat_count as u32 * 2,
            &[
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ratio: 1.0,
                },
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ratio: 2.0,
                },
            ],
        );

        let fallback_lin = res.default_sampler_linear.sampler;

        // helper: resolve a gltf texture to (Arc<AllocatedImage>, vk::Sampler)
        let resolve_texture = |tex: Option<gltf::texture::Texture>,
                               fallback_img: &Arc<AllocatedImage>,
                               fallback_sampler: vk::Sampler|
         -> (Arc<AllocatedImage>, vk::Sampler) {
            let Some(t) = tex else {
                return (Arc::clone(fallback_img), fallback_sampler);
            };
            let img = images
                .get(t.source().index())
                .map(Arc::clone)
                .unwrap_or_else(|| Arc::clone(fallback_img));
            let samp = samplers
                .get(t.sampler().index().unwrap_or(usize::MAX))
                .map(|s| s.sampler)
                .unwrap_or(fallback_sampler);
            (img, samp)
        };

        let mut materials: HashMap<String, Arc<MaterialInstance>> = HashMap::new();

        for (i, mat) in document.materials().enumerate() {
            let pbr = mat.pbr_metallic_roughness();
            let c = pbr.base_color_factor();
            let mr = [pbr.metallic_factor(), pbr.roughness_factor(), 0.0, 0.0];
            let mut constants = MaterialConstants::default();
            constants.color_factors = c;
            constants.metal_rough_factors = mr;

            let dst = material_data_buffer.info.mapped_data;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&constants as *const MaterialConstants).cast::<u8>(),
                    dst.cast::<u8>().add(i * mat_stride),
                    mat_stride,
                )
            };

            let (color_img, color_samp) = resolve_texture(
                pbr.base_color_texture().map(|t| t.texture()),
                &res.white_image,
                fallback_lin,
            );
            let (mr_img, mr_samp) = resolve_texture(
                pbr.metallic_roughness_texture().map(|t| t.texture()),
                &res.grey_image,
                fallback_lin,
            );

            let pass = match mat.alpha_mode() {
                gltf::material::AlphaMode::Blend => MaterialPass::Transparent,
                _ => MaterialPass::MainColor,
            };

            let mat_resources = MaterialResources {
                color_image: color_img,
                color_sampler: color_samp,
                metal_rough_image: mr_img,
                metal_rough_sampler: mr_samp,
                data_buffer: &material_data_buffer,
                data_buffer_offset: (i * mat_stride) as u32,
            };

            let instance = res.metal_rough_material.write_material(
                &ctx.device,
                pass,
                &mat_resources,
                &mut descriptor_pool,
            );
            let key = mat
                .name()
                .map(String::from)
                .unwrap_or_else(|| format!("mat_{i}"));
            materials.insert(key, Arc::new(instance));
        }

        // ── 4. Meshes ────────────────────────────────────────────────────────────
        use crate::meshes::{Bounds, GeoSurface, MeshAsset};
        use crate::primitives::Vertex;
        use crate::scene::{MeshNode, MeshSurface, Node};

        // Build fallback material if document has none
        let fallback_material: Arc<crate::material::MaterialInstance> = {
            let mat_resources = crate::material::MaterialResources {
                color_image: Arc::clone(&res.white_image),
                color_sampler: fallback_lin,
                metal_rough_image: Arc::clone(&res.grey_image),
                metal_rough_sampler: fallback_lin,
                data_buffer: &material_data_buffer,
                data_buffer_offset: 0,
            };
            Arc::new(res.metal_rough_material.write_material(
                &ctx.device,
                crate::material::MaterialPass::MainColor,
                &mat_resources,
                &mut descriptor_pool,
            ))
        };

        // material index → Arc<MaterialInstance>
        let mat_by_index: Vec<Arc<crate::material::MaterialInstance>> = document
            .materials()
            .map(|m| {
                let key = m
                    .name()
                    .map(String::from)
                    .unwrap_or_else(|| format!("mat_{}", m.index().unwrap_or(0)));
                materials
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&fallback_material))
            })
            .collect();

        // Temporary per-mesh storage: asset + per-primitive material
        struct MeshEntry {
            asset: Arc<MeshAsset>,
            prim_materials: Vec<Arc<crate::material::MaterialInstance>>,
        }
        let mut mesh_entries: Vec<Option<MeshEntry>> =
            (0..document.meshes().count()).map(|_| None).collect();
        let mut meshes: HashMap<String, Arc<MeshAsset>> = HashMap::new();

        for gltf_mesh in document.meshes() {
            let name = gltf_mesh
                .name()
                .map(String::from)
                .unwrap_or_else(|| format!("mesh_{}", gltf_mesh.index()));
            let mut vertices: Vec<Vertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut surfaces: Vec<GeoSurface> = Vec::new();
            let mut prim_materials: Vec<Arc<crate::material::MaterialInstance>> = Vec::new();

            for primitive in gltf_mesh.primitives() {
                let start_index = indices.len() as u32;
                let initial_vtx = vertices.len();
                let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

                match reader.read_indices() {
                    Some(ri) => indices.extend(ri.into_u32()),
                    None => continue,
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
                    .map(|c| c.into_rgba_f32().collect())
                    .unwrap_or_default();

                for (i, pos) in positions.iter().enumerate() {
                    vertices.push(Vertex {
                        position: *pos,
                        normal: normals.get(i).copied().unwrap_or([1.0, 0.0, 0.0]),
                        uv_x: uvs.get(i).map(|u| u[0]).unwrap_or(0.0),
                        uv_y: uvs.get(i).map(|u| u[1]).unwrap_or(0.0),
                        color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                    });
                }

                // Compute AABB bounds over this primitive's vertices (local space)
                let bounds = if !positions.is_empty() {
                    let mut min = glm::vec3(f32::MAX, f32::MAX, f32::MAX);
                    let mut max = glm::vec3(f32::MIN, f32::MIN, f32::MIN);
                    for v in &vertices[initial_vtx..] {
                        let p = glm::vec3(v.position[0], v.position[1], v.position[2]);
                        min = glm::min2(&min, &p);
                        max = glm::max2(&max, &p);
                    }
                    let origin = (max + min) / 2.0;
                    let extents = (max - min) / 2.0;
                    Bounds {
                        origin,
                        extents,
                        sphere_radius: glm::length(&extents),
                    }
                } else {
                    Bounds::default()
                };

                let mat = primitive
                    .material()
                    .index()
                    .and_then(|i| mat_by_index.get(i).cloned())
                    .unwrap_or_else(|| Arc::clone(&fallback_material));

                surfaces.push(GeoSurface {
                    start_index,
                    count: indices.len() as u32 - start_index,
                    bounds,
                });
                prim_materials.push(mat);
            }

            let gpu_mesh = renderer.upload_mesh(&indices, &vertices);
            let asset = Arc::new(MeshAsset {
                name: name.clone(),
                surfaces,
                mesh_buffers: gpu_mesh,
            });
            mesh_entries[gltf_mesh.index()] = Some(MeshEntry {
                asset: Arc::clone(&asset),
                prim_materials,
            });
            meshes.insert(name, asset);
        }

        // ── 5. Nodes ─────────────────────────────────────────────────────────────
        // NodeBuilder lets us wire children before boxing into Box<dyn Renderable>
        enum NodeBuilder {
            Mesh(MeshNode),
            Empty(Node),
        }
        impl NodeBuilder {
            fn children_mut(&mut self) -> &mut Vec<Box<dyn Renderable>> {
                match self {
                    NodeBuilder::Mesh(n) => &mut n.node.children,
                    NodeBuilder::Empty(n) => &mut n.children,
                }
            }
            fn into_renderable(self) -> Box<dyn Renderable> {
                match self {
                    NodeBuilder::Mesh(n) => Box::new(n),
                    NodeBuilder::Empty(n) => Box::new(n),
                }
            }
        }

        let node_count = document.nodes().count();
        let mut node_builders: Vec<Option<NodeBuilder>> = (0..node_count).map(|_| None).collect();

        for gltf_node in document.nodes() {
            let local_transform = crate::meshes::node_to_mat4(&gltf_node);
            let builder = if let Some(entry) = gltf_node
                .mesh()
                .and_then(|m| mesh_entries.get(m.index()).and_then(|e| e.as_ref()))
            {
                let surfaces: Vec<MeshSurface> = entry
                    .asset
                    .surfaces
                    .iter()
                    .zip(entry.prim_materials.iter())
                    .map(|(geo, mat)| MeshSurface {
                        geo: *geo,
                        material: Arc::clone(mat),
                    })
                    .collect();
                NodeBuilder::Mesh(MeshNode {
                    node: Node::new(local_transform),
                    mesh: Arc::clone(&entry.asset),
                    surfaces,
                })
            } else {
                NodeBuilder::Empty(Node::new(local_transform))
            };
            node_builders[gltf_node.index()] = Some(builder);
        }

        // Pass 2: wire children into parents in post-order.
        // Naively iterating by index is wrong: a parent may be take()n by its own
        // parent before we get to wire the parent's children, silently dropping them.
        // Post-order guarantees every node's subtree is fully wired before its parent
        // claims it.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        let mut is_child = vec![false; node_count];
        for gltf_node in document.nodes() {
            for child in gltf_node.children() {
                adj[gltf_node.index()].push(child.index());
                is_child[child.index()] = true;
            }
        }
        let mut post_order: Vec<usize> = Vec::with_capacity(node_count);
        for root in 0..node_count {
            if !is_child[root] {
                let mut stack: Vec<(usize, bool)> = vec![(root, false)];
                while let Some((idx, visited)) = stack.pop() {
                    if visited {
                        post_order.push(idx);
                    } else {
                        stack.push((idx, true));
                        for &child in adj[idx].iter().rev() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        }
        for &parent_idx in &post_order {
            let child_idxs = adj[parent_idx].clone();
            for child_idx in child_idxs {
                let child = node_builders[child_idx].take();
                if let (Some(child_b), Some(parent_b)) = (child, node_builders[parent_idx].as_mut())
                {
                    parent_b.children_mut().push(child_b.into_renderable());
                }
            }
        }

        // Remaining Some entries are roots (never taken as a child)
        let top_nodes: Vec<Box<dyn Renderable>> = node_builders
            .into_iter()
            .filter_map(|n| n.map(NodeBuilder::into_renderable))
            .collect();
        log::info!(
            "GLTF '{}': {} top-level nodes, {} meshes loaded",
            path.as_ref().display(),
            top_nodes.len(),
            meshes.len(),
        );

        Some(Gltf {
            meshes,
            images,
            samplers,
            materials,
            top_nodes,
            material_data_buffer,
            descriptor_pool,
            device: ctx.device.clone(),
        })
    }
}
