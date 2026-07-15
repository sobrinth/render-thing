//! Vulkan renderer for egui, adapted from
//! https://github.com/adrien-ben/egui-ash-renderer (MIT, see LICENSE-egui-ash-renderer)
//! and trimmed down to the single configuration this engine uses: vk-mem allocation
//! and dynamic rendering.

mod allocator;
mod error;
mod vulkan;

use std::collections::HashMap;
use std::sync::Arc;

use allocator::Allocator;
use ash::{Device, vk};
use egui::{
    ClippedPrimitive, ImageData, TextureId,
    epaint::{ImageDelta, Primitive},
};
pub(crate) use error::RendererError;
use mesh::*;
use vk_mem::Allocator as VkMemAllocator;
use vulkan::*;

pub(crate) type RendererResult<T> = Result<T, RendererError>;

const MAX_TEXTURE_COUNT: u32 = 1024;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Options {
    pub(crate) in_flight_frames: usize,
    pub(crate) enable_depth_test: bool,
    /// Depth writes are always disabled when `enable_depth_test` is false.
    pub(crate) enable_depth_write: bool,
    pub(crate) srgb_framebuffer: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            in_flight_frames: 1,
            enable_depth_test: false,
            enable_depth_write: false,
            srgb_framebuffer: false,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DynamicRendering {
    pub(crate) color_attachment_format: vk::Format,
    pub(crate) depth_attachment_format: Option<vk::Format>,
}

/// Records egui draw commands into a command buffer supplied by the caller; does not submit
/// anything itself. Holds one vertex/index buffer pair per in-flight frame, resized on demand.
pub(crate) struct Renderer {
    device: Device,
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    managed_textures: HashMap<TextureId, Texture>,
    textures: HashMap<TextureId, vk::DescriptorSet>,
    next_user_texture_id: u64,
    options: Options,
    frames: Option<Frames>,
}

impl Renderer {
    pub(crate) fn new(
        vk_mem_allocator: Arc<VkMemAllocator>,
        device: Device,
        dynamic_rendering: DynamicRendering,
        options: Options,
    ) -> RendererResult<Self> {
        log::debug!("Creating egui renderer with options {options:?}");

        if options.in_flight_frames == 0 {
            return Err(RendererError::Init(String::from(
                "'in_flight_frames' parameter should be at least one",
            )));
        }

        let allocator = Allocator::new(vk_mem_allocator);

        let descriptor_set_layout = create_vulkan_descriptor_set_layout(&device)?;
        let pipeline_layout = create_vulkan_pipeline_layout(&device, descriptor_set_layout)?;
        let pipeline =
            create_vulkan_pipeline(&device, pipeline_layout, dynamic_rendering, options)?;
        let descriptor_pool = create_vulkan_descriptor_pool(&device, MAX_TEXTURE_COUNT)?;

        Ok(Self {
            device,
            allocator,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            managed_textures: HashMap::new(),
            textures: HashMap::new(),
            next_user_texture_id: 0,
            options,
            frames: None,
        })
    }

    /// Applies `textures_delta.set`; must be called before submitting the frame's command buffer.
    pub(crate) fn set_textures(
        &mut self,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        textures_delta: &[(TextureId, ImageDelta)],
    ) -> RendererResult<()> {
        for (id, delta) in textures_delta {
            let (width, height, data) = match &delta.image {
                ImageData::Color(image) => {
                    let w = image.width() as u32;
                    let h = image.height() as u32;
                    let data = image
                        .pixels
                        .iter()
                        .flat_map(|c| c.to_array())
                        .collect::<Vec<_>>();

                    (w, h, data)
                }
            };

            if let Some([offset_x, offset_y]) = delta.pos {
                let texture = self
                    .managed_textures
                    .get_mut(id)
                    .ok_or(RendererError::BadTexture(*id))?;

                texture.update(
                    &self.device,
                    queue,
                    command_pool,
                    &mut self.allocator,
                    vk::Rect2D {
                        offset: vk::Offset2D {
                            x: offset_x as _,
                            y: offset_y as _,
                        },
                        extent: vk::Extent2D { width, height },
                    },
                    data.as_slice(),
                )?;
            } else {
                let texture = Texture::from_rgba8(
                    &self.device,
                    queue,
                    command_pool,
                    &mut self.allocator,
                    width,
                    height,
                    data.as_slice(),
                )?;

                let set = create_vulkan_descriptor_set(
                    &self.device,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    texture.image_view,
                    texture.sampler,
                )?;

                if let Some(previous) = self.managed_textures.insert(*id, texture) {
                    previous.destroy(&self.device, &mut self.allocator);
                }
                if let Some(previous) = self.textures.insert(*id, set) {
                    unsafe {
                        self.device
                            .free_descriptor_sets(self.descriptor_pool, &[previous])?
                    };
                }
            }
        }

        Ok(())
    }

    /// Frees textures named in `textures_delta.free`; must be called after the frame is done rendering.
    pub(crate) fn free_textures(&mut self, ids: &[TextureId]) -> RendererResult<()> {
        for id in ids {
            if let Some(texture) = self.managed_textures.remove(id) {
                texture.destroy(&self.device, &mut self.allocator);
            }
            if let Some(set) = self.textures.remove(id) {
                unsafe {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[set])?
                };
            }
        }

        Ok(())
    }

    pub(crate) fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        pixels_per_point: f32,
        primitives: &[ClippedPrimitive],
    ) -> RendererResult<()> {
        if primitives.is_empty() {
            return Ok(());
        }

        if self.frames.is_none() {
            self.frames.replace(Frames::new(
                &self.device,
                &mut self.allocator,
                primitives,
                self.options.in_flight_frames,
            )?);
        }

        let mesh = self.frames.as_mut().unwrap().next();
        mesh.update(&self.device, &mut self.allocator, primitives)?;

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            )
        };

        let screen_width = extent.width as f32;
        let screen_height = extent.height as f32;

        unsafe {
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    width: screen_width,
                    height: screen_height,
                    max_depth: 1.0,
                    ..Default::default()
                }],
            )
        };

        // Ortho projection
        let projection = orthographic_vk(
            0.0,
            screen_width / pixels_per_point,
            0.0,
            -(screen_height / pixels_per_point),
            -1.0,
            1.0,
        );
        unsafe {
            let push = any_as_u8_slice(&projection);
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push,
            )
        };

        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                mesh.indices,
                0,
                vk::IndexType::UINT32,
            )
        };

        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertices], &[0])
        };

        let mut index_offset = 0u32;
        let mut vertex_offset = 0i32;
        let mut current_texture_id: Option<TextureId> = None;

        for p in primitives {
            let clip_rect = p.clip_rect;
            match &p.primitive {
                Primitive::Mesh(m) => {
                    let clip_x = clip_rect.min.x * pixels_per_point;
                    let clip_y = clip_rect.min.y * pixels_per_point;
                    let clip_w = clip_rect.max.x * pixels_per_point - clip_x;
                    let clip_h = clip_rect.max.y * pixels_per_point - clip_y;

                    let scissors = [vk::Rect2D {
                        offset: vk::Offset2D {
                            x: (clip_x as i32).max(0),
                            y: (clip_y as i32).max(0),
                        },
                        extent: vk::Extent2D {
                            width: clip_w.min(screen_width) as _,
                            height: clip_h.min(screen_height) as _,
                        },
                    }];

                    unsafe {
                        self.device.cmd_set_scissor(command_buffer, 0, &scissors);
                    }

                    if Some(m.texture_id) != current_texture_id {
                        let descriptor_set = *self
                            .textures
                            .get(&m.texture_id)
                            .ok_or(RendererError::BadTexture(m.texture_id))?;

                        unsafe {
                            self.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[descriptor_set],
                                &[],
                            )
                        };
                        current_texture_id = Some(m.texture_id);
                    }

                    let index_count = m.indices.len() as u32;
                    unsafe {
                        self.device.cmd_draw_indexed(
                            command_buffer,
                            index_count,
                            1,
                            index_offset,
                            vertex_offset,
                            0,
                        )
                    };

                    index_offset += index_count;
                    vertex_offset += m.vertices.len() as i32;
                }
                Primitive::Callback(_) => {
                    log::warn!("Callback primitives not yet supported")
                }
            }
        }

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        log::debug!("Destroying egui renderer");
        let device = &self.device;

        unsafe {
            if let Some(frames) = self.frames.take() {
                frames.destroy(&mut self.allocator);
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            for (_, t) in self.managed_textures.drain() {
                t.destroy(device, &mut self.allocator);
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Holds one mesh per in-flight frame, cycled round-robin by `cmd_draw`.
struct Frames {
    index: usize,
    count: usize,
    meshes: Vec<Mesh>,
}

impl Frames {
    fn new(
        device: &Device,
        allocator: &mut Allocator,
        primitives: &[ClippedPrimitive],
        count: usize,
    ) -> RendererResult<Self> {
        let meshes = (0..count)
            .map(|_| Mesh::new(device, allocator, primitives))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            index: 0,
            count,
            meshes,
        })
    }

    fn next(&mut self) -> &mut Mesh {
        let result = &mut self.meshes[self.index];
        self.index = (self.index + 1) % self.count;
        result
    }

    fn destroy(self, allocator: &mut Allocator) {
        for mesh in self.meshes.into_iter() {
            mesh.destroy(allocator);
        }
    }
}

mod mesh {
    use super::RendererResult;
    use super::allocator::{Allocator, Memory};
    use super::vulkan::*;
    use ash::{Device, vk};
    use egui::ClippedPrimitive;
    use egui::epaint::{Primitive, Vertex};
    use std::mem::size_of;

    /// Vertex and index buffer resources for one frame in flight.
    pub(crate) struct Mesh {
        pub(crate) vertices: vk::Buffer,
        vertices_mem: Memory,
        vertex_count: usize,
        pub(crate) indices: vk::Buffer,
        indices_mem: Memory,
        index_count: usize,
    }

    impl Mesh {
        pub(crate) fn new(
            device: &Device,
            allocator: &mut Allocator,
            primitives: &[ClippedPrimitive],
        ) -> RendererResult<Self> {
            let vertices = create_vertices(primitives);
            let vertex_count = vertices.len();
            let indices = create_indices(primitives);
            let index_count = indices.len();

            let (vertices, vertices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &vertices,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            let (indices, indices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            Ok(Mesh {
                vertices,
                vertices_mem,
                vertex_count,
                indices,
                indices_mem,
                index_count,
            })
        }

        pub(crate) fn update(
            &mut self,
            device: &Device,
            allocator: &mut Allocator,
            primitives: &[ClippedPrimitive],
        ) -> RendererResult<()> {
            let vertices = create_vertices(primitives);
            if vertices.len() > self.vertex_count {
                let vertex_count = vertices.len();
                let size = vertex_count * size_of::<Vertex>();
                let (vertices, vertices_mem) =
                    allocator.create_buffer(size, vk::BufferUsageFlags::VERTEX_BUFFER)?;

                self.vertex_count = vertex_count;

                let old_vertices = self.vertices;
                self.vertices = vertices;

                let old_vertices_mem = std::mem::replace(&mut self.vertices_mem, vertices_mem);

                allocator.destroy_buffer(old_vertices, old_vertices_mem);
            }
            allocator.update_buffer(device, &mut self.vertices_mem, &vertices)?;

            let indices = create_indices(primitives);
            if indices.len() > self.index_count {
                let index_count = indices.len();
                let size = index_count * size_of::<u32>();
                let (indices, indices_mem) =
                    allocator.create_buffer(size, vk::BufferUsageFlags::INDEX_BUFFER)?;

                self.index_count = index_count;

                let old_indices = self.indices;
                self.indices = indices;

                let old_indices_mem = std::mem::replace(&mut self.indices_mem, indices_mem);

                allocator.destroy_buffer(old_indices, old_indices_mem);
            }
            allocator.update_buffer(device, &mut self.indices_mem, &indices)?;

            Ok(())
        }

        pub(crate) fn destroy(self, allocator: &mut Allocator) {
            allocator.destroy_buffer(self.vertices, self.vertices_mem);
            allocator.destroy_buffer(self.indices, self.indices_mem);
        }
    }

    fn create_vertices(primitives: &[ClippedPrimitive]) -> Vec<Vertex> {
        let vertex_count = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(m) => m.vertices.len(),
                _ => 0,
            })
            .sum();

        let mut vertices = Vec::with_capacity(vertex_count);
        for p in primitives {
            if let Primitive::Mesh(m) = &p.primitive {
                vertices.extend_from_slice(&m.vertices);
            }
        }
        vertices
    }

    fn create_indices(primitives: &[ClippedPrimitive]) -> Vec<u32> {
        let index_count = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(m) => m.indices.len(),
                _ => 0,
            })
            .sum();

        let mut indices = Vec::with_capacity(index_count);
        for p in primitives {
            if let Primitive::Mesh(m) = &p.primitive {
                indices.extend_from_slice(&m.indices);
            }
        }

        indices
    }
}

/// Orthographic projection matrix for use with Vulkan: right-handed y-up source space,
/// right-handed y-down destination space, depth clip range [0, 1].
/// from: https://github.com/fu5ha/ultraviolet (to limit dependencies)
#[inline]
fn orthographic_vk(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    let rml = right - left;
    let rpl = right + left;
    let tmb = top - bottom;
    let tpb = top + bottom;
    let fmn = far - near;

    #[rustfmt::skip]
    let res = [
        2.0 / rml, 0.0, 0.0, 0.0,
        0.0, -2.0 / tmb, 0.0, 0.0,
        0.0, 0.0, -1.0 / fmn, 0.0,
        -(rpl / rml), -(tpb / tmb), -(near / fmn), 1.0
    ];

    res
}
