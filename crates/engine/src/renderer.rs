use crate::command_buffer::ImmediateSubmitData;
use crate::context::{QueueData, VkContext};
use crate::descriptor;
use crate::descriptor::{
    DescriptorSetLayout, DescriptorWriter, GrowableAllocator, LayoutBuilder, PoolSizeRatio,
};
use crate::frame::FrameData;
use crate::input::{ElementState, Key, NamedKey};
use crate::material::GltfMetallicRoughness;
use crate::pipeline::{ComputeEffect, ComputePushConstants, Pipeline, PipelineLayout};
use crate::primitives::{GPUSceneData, Vertex};
use crate::resources::{
    AllocatedBuffer, AllocatedImage, ImageCreateInfo, Sampler, upload_mesh_buffers,
};
use crate::stats::StatsHistory;
use crate::swapchain::Swapchain;
use crate::sync::{Fence, Semaphore};
use crate::ui::UiContext;
use ash::{Device, vk};
use itertools::Itertools;
use nalgebra_glm as glm;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;

pub(crate) const FRAME_OVERLAP: u32 = 2;

pub(crate) struct MeshEntry {
    pub buffers: crate::primitives::GPUMeshBuffers,
    pub bounds: crate::meshes::Bounds,
    pub index_count: u32,
}

pub(crate) struct TextureEntry {
    pub image: std::sync::Arc<crate::resources::AllocatedImage>,
    pub sampler: std::sync::Arc<crate::resources::Sampler>,
}

pub(crate) struct MaterialEntry {
    pub instance: std::sync::Arc<crate::material::MaterialInstance>,
    pub _data_buffer: crate::resources::AllocatedBuffer,
}

pub(crate) struct RendererResources {
    pub(crate) frame_number: u32,
    pub(crate) window_size: (u32, u32),
    pub(crate) render_scale: f32,
    pub(crate) render_size: (u32, u32),
    pub(crate) resize_requested: bool,
    pub(crate) ui_context: UiContext,
    pub(crate) gpu_alloc: Arc<vk_mem::Allocator>,
    pub(crate) swapchain: Swapchain,
    pub(crate) frames: Vec<FrameData>,
    pub(crate) graphics_queue: QueueData,
    pub(crate) descriptor_allocator: descriptor::Allocator,
    pub(crate) immediate_submit: ImmediateSubmitData,
    pub(crate) draw_image: AllocatedImage,
    pub(crate) draw_image_descriptors: vk::DescriptorSet,
    pub(crate) draw_image_descriptor_layout: DescriptorSetLayout,
    pub(crate) depth_image: AllocatedImage,
    pub(crate) effect_pipeline_layout: PipelineLayout,
    pub(crate) background_effects: Vec<ComputeEffect>,
    pub(crate) active_background_effect: usize,
    pub(crate) scene_data: GPUSceneData,
    pub(crate) scene_data_layout: DescriptorSetLayout,
    pub(crate) default_sampler_nearest: Arc<Sampler>,
    pub(crate) default_sampler_linear: Arc<Sampler>,
    pub(crate) white_image: Arc<AllocatedImage>,
    pub(crate) grey_image: Arc<AllocatedImage>,
    pub(crate) black_image: Arc<AllocatedImage>,
    pub(crate) checkerboard_image: Arc<AllocatedImage>,
    pub(crate) metal_rough_image: Arc<AllocatedImage>,
    pub(crate) metal_rough_material: GltfMetallicRoughness,
    pub(crate) stats: StatsHistory,
    pub(crate) show_stats: bool,
    pub(crate) show_controls: bool,
    pub(crate) mesh_registry: Vec<MeshEntry>,
    pub(crate) texture_registry: Vec<TextureEntry>,
    pub(crate) material_registry: Vec<MaterialEntry>,
    pub(crate) material_descriptor_allocator: descriptor::GrowableAllocator,
    pub(crate) default_white_texture: crate::TextureHandle,
    pub(crate) default_grey_texture: crate::TextureHandle,
    pub(crate) default_black_texture: crate::TextureHandle,
    pub(crate) default_checkerboard_texture: crate::TextureHandle,
    pub(crate) default_metal_rough_texture: crate::TextureHandle,
    pub(crate) gizmo_mesh: crate::MeshHandle,
    /// [0..2] bright R/G/B for positive arm + cap, [3..5] dim R/G/B for negative stubs, [6] origin cube
    pub(crate) gizmo_materials: [crate::MaterialHandle; 7],
    pub(crate) show_dev_overlay: bool,
    pub(crate) last_frame_start: Option<std::time::Instant>,
}

pub(crate) struct VulkanRenderer {
    pub(crate) resources: ManuallyDrop<RendererResources>,
    pub(crate) context: ManuallyDrop<VkContext>,
}

impl VulkanRenderer {
    pub(crate) fn initialize(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        window_size: (u32, u32),
        config: crate::EngineConfig,
    ) -> Self {
        let (context, graphics_queue, gpu_alloc) = VkContext::initialize(window, &config.app_name);

        let swapchain = Swapchain::create(&context, [window_size.0, window_size.1]);
        let ui = UiContext::initialize(&context.device, &gpu_alloc, swapchain.properties);

        let immediate_submit = Self::create_immediate_submit_data(&context, &graphics_queue);

        let frames = Self::create_framedata(&context, &gpu_alloc, &graphics_queue);

        let draw_image = AllocatedImage::create(
            &context,
            &gpu_alloc,
            config.max_render_resolution,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let depth_image = AllocatedImage::create(
            &context,
            &gpu_alloc,
            (2560, 1440),
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            false,
        );

        let (descriptor_allocator, draw_image_descriptor_layout, draw_image_descriptors) =
            Self::init_descriptors(&context, &draw_image);

        let scene_data_layout = Self::init_scene_data(&context);

        let metal_rough_material = crate::material::GltfMetallicRoughness::build_pipelines(
            &context,
            draw_image.format,
            depth_image.format,
            &scene_data_layout.layout,
        );

        let gradient_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/gradient.comp.spv");

        let gradient_color_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/gradient_color.comp.spv");

        let sky_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/sky.comp.spv");

        let (effects, effect_pipeline_layout) = Self::initialize_effect_pipelines(
            &context,
            &draw_image_descriptor_layout.layout,
            &[
                (gradient_shader_module, "color_grid"),
                (gradient_color_shader_module, "gradient"),
                (sky_shader_module, "sky"),
            ],
        );
        unsafe {
            context
                .device
                .destroy_shader_module(gradient_shader_module, None);
            context
                .device
                .destroy_shader_module(sky_shader_module, None);
            context
                .device
                .destroy_shader_module(gradient_color_shader_module, None);
        };

        // default data
        const WHITE: u32 = u32::from_le_bytes([255, 255, 255, 255]);
        const GREY: u32 = u32::from_le_bytes([128, 128, 128, 255]);
        const MAGENTA: u32 = u32::from_le_bytes([255, 0, 255, 255]);
        const BLACK: u32 = u32::from_le_bytes([0, 0, 0, 0]);
        // glTF metallic-roughness: R=unused, G=roughness(1.0), B=metallic(0.0)
        const METAL_ROUGH_DEFAULT: u32 = u32::from_le_bytes([0, 255, 0, 255]);

        let white_image = Arc::new(AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[WHITE],
            ImageCreateInfo {
                resolution: (1, 1),
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        ));

        let grey_image = Arc::new(AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[GREY],
            ImageCreateInfo {
                resolution: (1, 1),
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        ));

        let black_image = Arc::new(AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[BLACK],
            ImageCreateInfo {
                resolution: (1, 1),
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        ));

        let mut checkerboard_data = [0u32; 16 * 16];
        for x in 0..16 {
            for y in 0..16 {
                checkerboard_data[x + y * 16] = if ((x % 2) ^ (y % 2)) == 0 {
                    MAGENTA
                } else {
                    BLACK
                }
            }
        }

        let checkerboard_image = Arc::new(AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &checkerboard_data,
            ImageCreateInfo {
                resolution: (16, 16),
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        ));

        let metal_rough_image = Arc::new(AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[METAL_ROUGH_DEFAULT],
            ImageCreateInfo {
                resolution: (1, 1),
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        ));

        let sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let nearest_handle = unsafe { context.device.create_sampler(&sampler, None) }.unwrap();
        let default_sampler_nearest =
            Arc::new(Sampler::new(nearest_handle, context.device.clone()));

        let sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let linear_handle = unsafe { context.device.create_sampler(&sampler, None) }.unwrap();
        let default_sampler_linear = Arc::new(Sampler::new(linear_handle, context.device.clone()));

        let material_descriptor_allocator = descriptor::GrowableAllocator::init(
            &context.device,
            1000,
            &[
                descriptor::PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ratio: 1.0,
                },
                descriptor::PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ratio: 2.0,
                },
            ],
        );

        let mut renderer = Self {
            resources: ManuallyDrop::new(RendererResources {
                frame_number: 0,
                window_size,
                render_scale: config.initial_render_scale,
                render_size: (0, 0),
                resize_requested: false,
                ui_context: ui,
                gpu_alloc,
                swapchain,
                frames,
                graphics_queue,
                descriptor_allocator,
                immediate_submit,
                draw_image,
                draw_image_descriptors,
                draw_image_descriptor_layout,
                depth_image,
                effect_pipeline_layout,
                background_effects: effects,
                active_background_effect: 0,
                scene_data: GPUSceneData {
                    ambient_color: [0.5, 0.5, 0.5, 1.0],
                    sunlight_direction: [0.667, 0.667, 0.333, 1.0],
                    sunlight_color: [1.0, 1.0, 1.0, 2.5],
                    ..Default::default()
                },
                scene_data_layout,
                default_sampler_nearest,
                default_sampler_linear,
                white_image,
                grey_image,
                black_image,
                checkerboard_image,
                metal_rough_image,
                metal_rough_material,
                stats: StatsHistory::default(),
                show_stats: false,
                show_controls: false,
                mesh_registry: Vec::new(),
                texture_registry: Vec::new(),
                material_registry: Vec::new(),
                material_descriptor_allocator,
                default_white_texture: crate::TextureHandle(0),
                default_grey_texture: crate::TextureHandle(1),
                default_black_texture: crate::TextureHandle(2),
                default_checkerboard_texture: crate::TextureHandle(3),
                default_metal_rough_texture: crate::TextureHandle(4),
                gizmo_mesh: crate::MeshHandle(0),
                gizmo_materials: [crate::MaterialHandle(0); 7],
                show_dev_overlay: false,
                last_frame_start: None,
            }),
            context: ManuallyDrop::new(context),
        };

        // Register default textures into the registry so handles 0-4 are valid.
        // Reuse the shared default samplers — no need for per-texture sampler objects.
        let white_img = Arc::clone(&renderer.resources.white_image);
        let grey_img = Arc::clone(&renderer.resources.grey_image);
        let black_img = Arc::clone(&renderer.resources.black_image);
        let cb_img = Arc::clone(&renderer.resources.checkerboard_image);
        let mr_img = Arc::clone(&renderer.resources.metal_rough_image);
        let linear = Arc::clone(&renderer.resources.default_sampler_linear);
        let nearest = Arc::clone(&renderer.resources.default_sampler_nearest);

        let wh = renderer.register_texture(white_img, Arc::clone(&linear));
        let gr = renderer.register_texture(grey_img, Arc::clone(&linear));
        let bl = renderer.register_texture(black_img, Arc::clone(&linear));
        let cb = renderer.register_texture(cb_img, nearest);
        let mr = renderer.register_texture(mr_img, linear);
        renderer.resources.default_white_texture = wh;
        renderer.resources.default_grey_texture = gr;
        renderer.resources.default_black_texture = bl;
        renderer.resources.default_checkerboard_texture = cb;
        renderer.resources.default_metal_rough_texture = mr;

        // Create gizmo resources: a unit box mesh shared across three axis materials (R/G/B).
        let (gizmo_idx, gizmo_verts) = Self::unit_cube_mesh();
        let gizmo_mesh = renderer.upload_mesh(&gizmo_idx, &gizmo_verts);

        let make_gizmo_mat = |r: &mut VulkanRenderer, color: [f32; 4]| -> crate::MaterialHandle {
            let mut c = crate::material::MaterialConstants::default();
            c.color_factors = color;
            c.metal_rough_factors = [0.0, 1.0, 0.0, 0.0];
            r.create_material_colored(c, crate::material::MaterialPass::MainColor)
        };
        let x_pos = make_gizmo_mat(&mut renderer, [1.00, 0.00, 0.00, 1.0]);
        let y_pos = make_gizmo_mat(&mut renderer, [0.00, 1.00, 0.00, 1.0]);
        let z_pos = make_gizmo_mat(&mut renderer, [0.00, 0.00, 1.00, 1.0]);
        let x_neg = make_gizmo_mat(&mut renderer, [0.35, 0.00, 0.00, 1.0]);
        let y_neg = make_gizmo_mat(&mut renderer, [0.00, 0.35, 0.00, 1.0]);
        let z_neg = make_gizmo_mat(&mut renderer, [0.00, 0.00, 0.35, 1.0]);
        let origin = make_gizmo_mat(&mut renderer, [0.2, 0.2, 0.2, 1.0]);
        renderer.resources.gizmo_mesh = gizmo_mesh;
        renderer.resources.gizmo_materials = [x_pos, y_pos, z_pos, x_neg, y_neg, z_neg, origin];

        renderer
    }

    fn unit_cube_mesh() -> (Vec<u32>, Vec<Vertex>) {
        let positions: [[f32; 3]; 8] = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];
        let vertices = positions
            .iter()
            .map(|&p| Vertex {
                position: p,
                uv_x: 0.0,
                normal: [0.0, 1.0, 0.0],
                uv_y: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();
        let indices = vec![
            0u32, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
        ];
        (indices, vertices)
    }

    pub(crate) fn egui_context(&self) -> egui::Context {
        self.resources.ui_context.ctx.clone()
    }

    pub(crate) fn on_key_event(&mut self, key_event: (ElementState, Key)) {
        if !self.resources.ui_context.ctx.wants_keyboard_input() {
            if matches!(key_event, (ElementState::Pressed, Key::Named(NamedKey::F2))) {
                self.resources.show_controls = !self.resources.show_controls;
                return;
            }
            if matches!(key_event, (ElementState::Pressed, Key::Named(NamedKey::F3))) {
                self.resources.show_dev_overlay = !self.resources.show_dev_overlay;
                return;
            }
            if matches!(key_event, (ElementState::Pressed, Key::Named(NamedKey::F4))) {
                self.resources.show_stats = !self.resources.show_stats;
            }
        }
    }

    pub(crate) fn register_texture(
        &mut self,
        image: Arc<AllocatedImage>,
        sampler: Arc<Sampler>,
    ) -> crate::TextureHandle {
        let idx = self.resources.texture_registry.len() as u32;
        self.resources
            .texture_registry
            .push(TextureEntry { image, sampler });
        crate::TextureHandle(idx)
    }

    pub(crate) fn upload_mesh(
        &mut self,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> crate::MeshHandle {
        use crate::meshes::Bounds;
        let buffers = upload_mesh_buffers(
            &self.resources.gpu_alloc,
            &self.context,
            &self.resources.immediate_submit,
            &self.resources.graphics_queue,
            indices,
            vertices,
        );
        let mut min = glm::vec3(f32::MAX, f32::MAX, f32::MAX);
        let mut max = glm::vec3(f32::MIN, f32::MIN, f32::MIN);
        for v in vertices {
            min.x = min.x.min(v.position[0]);
            min.y = min.y.min(v.position[1]);
            min.z = min.z.min(v.position[2]);
            max.x = max.x.max(v.position[0]);
            max.y = max.y.max(v.position[1]);
            max.z = max.z.max(v.position[2]);
        }
        let origin = (min + max) * 0.5;
        let extents = (max - min) * 0.5;
        let sphere_radius = extents.norm();
        let bounds = Bounds {
            origin,
            extents,
            sphere_radius,
        };
        let index_count = indices.len() as u32;
        let idx = self.resources.mesh_registry.len() as u32;
        self.resources.mesh_registry.push(MeshEntry {
            buffers,
            bounds,
            index_count,
        });
        crate::MeshHandle(idx)
    }

    pub(crate) fn upload_texture(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        sampler_type: crate::SamplerType,
    ) -> crate::TextureHandle {
        let image = AllocatedImage::create_from_bytes(
            &self.resources.gpu_alloc,
            &self.context,
            &self.resources.immediate_submit,
            &self.resources.graphics_queue,
            data,
            ImageCreateInfo {
                resolution: (width, height),
                format: vk::Format::R8G8B8A8_SRGB,
                usage: vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                mip_mapped: false,
            },
        );
        let filter = match sampler_type {
            crate::SamplerType::Linear => vk::Filter::LINEAR,
            crate::SamplerType::Nearest => vk::Filter::NEAREST,
        };
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter);
        let vk_sampler =
            unsafe { self.context.device.create_sampler(&sampler_info, None) }.unwrap();
        let sampler = Arc::new(Sampler::new(vk_sampler, self.context.device.clone()));
        self.register_texture(Arc::new(image), sampler)
    }

    pub(crate) fn create_material(
        &mut self,
        color: crate::TextureHandle,
        metal_rough: crate::TextureHandle,
        constants: crate::material::MaterialConstants,
        pass: crate::material::MaterialPass,
    ) -> crate::MaterialHandle {
        use crate::material::MaterialResources;
        use std::mem::size_of;

        let data_buffer = AllocatedBuffer::create(
            &self.resources.gpu_alloc,
            size_of::<crate::material::MaterialConstants>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
            Some(
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
            ),
        );

        let dst = data_buffer.info.mapped_data;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&constants as *const crate::material::MaterialConstants).cast::<u8>(),
                dst.cast::<u8>(),
                size_of::<crate::material::MaterialConstants>(),
            )
        };

        let color_sampler = self.resources.texture_registry[color.0 as usize]
            .sampler
            .sampler;
        let mr_sampler = self.resources.texture_registry[metal_rough.0 as usize]
            .sampler
            .sampler;
        let color_image = Arc::clone(&self.resources.texture_registry[color.0 as usize].image);
        let mr_image = Arc::clone(&self.resources.texture_registry[metal_rough.0 as usize].image);

        let resources = MaterialResources {
            color_image,
            color_sampler,
            metal_rough_image: mr_image,
            metal_rough_sampler: mr_sampler,
            data_buffer: &data_buffer,
            data_buffer_offset: 0,
        };

        // Split disjoint field borrows through ManuallyDrop by going through &mut *self.resources.
        let instance = {
            let res = &mut *self.resources;
            res.metal_rough_material.write_material(
                &self.context.device,
                pass,
                &resources,
                &mut res.material_descriptor_allocator,
            )
        };

        let idx = self.resources.material_registry.len() as u32;
        self.resources.material_registry.push(MaterialEntry {
            instance: Arc::new(instance),
            _data_buffer: data_buffer,
        });
        crate::MaterialHandle(idx)
    }

    pub(crate) fn create_material_colored(
        &mut self,
        constants: crate::material::MaterialConstants,
        pass: crate::material::MaterialPass,
    ) -> crate::MaterialHandle {
        let white = self.resources.default_white_texture;
        let metal_rough = self.resources.default_metal_rough_texture;
        self.create_material(white, metal_rough, constants, pass)
    }

    fn create_immediate_submit_data(
        context: &VkContext,
        graphics_queue: &QueueData,
    ) -> ImmediateSubmitData {
        let commandpool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool =
            unsafe { context.device.create_command_pool(&commandpool_info, None) }.unwrap();

        let commandbuffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer =
            unsafe { context.device.allocate_command_buffers(&commandbuffer_info) }.unwrap()[0];

        ImmediateSubmitData::new(
            command_pool,
            command_buffer,
            Fence::new_signaled(&context.device),
            context.device.clone(),
        )
    }

    fn create_framedata(
        context: &VkContext,
        gpu_alloc: &Arc<vk_mem::Allocator>,
        graphics_queue: &QueueData,
    ) -> Vec<FrameData> {
        let commandpool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let mut frames = Vec::with_capacity(FRAME_OVERLAP as usize);

        for _ in 0..FRAME_OVERLAP {
            let pool =
                unsafe { context.device.create_command_pool(&commandpool_info, None) }.unwrap();

            let commandbuffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY);

            let buffer =
                unsafe { context.device.allocate_command_buffers(&commandbuffer_info) }.unwrap()[0];

            let swapchain_semaphore = Semaphore::new(&context.device);

            let render_fence = Fence::new_signaled(&context.device);

            // create descriptor allocator exclusive to this frame
            let pool_ratios = [
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ratio: 3.0,
                },
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ratio: 3.0,
                },
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ratio: 3.0,
                },
                PoolSizeRatio {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ratio: 4.0,
                },
            ];

            let desc_allocator = GrowableAllocator::init(&context.device, 1000, &pool_ratios);

            let scene_buffer = AllocatedBuffer::create(
                gpu_alloc,
                2 * size_of::<GPUSceneData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk_mem::MemoryUsage::Auto,
                Some(
                    vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ),
            );

            let frame = FrameData::new(
                pool,
                buffer,
                swapchain_semaphore,
                render_fence,
                desc_allocator,
                scene_buffer,
                context.device.clone(),
            );
            frames.push(frame);
        }
        frames
    }

    pub(crate) fn wait_gpu_idle(&self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();
    }

    pub(crate) fn resize(&mut self, size: (u32, u32)) {
        if self.resources.window_size.0 != size.0 || self.resources.window_size.1 != size.1 {
            self.resources.window_size = size;
            self.resources.resize_requested = true;
        }
    }

    fn init_descriptors(
        context: &VkContext,
        draw_image: &AllocatedImage,
    ) -> (
        descriptor::Allocator,
        DescriptorSetLayout,
        vk::DescriptorSet,
    ) {
        let pool_sizes = [PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1f32,
        }];
        let pool = descriptor::Allocator::init_pool(&context.device, 10, pool_sizes.to_vec());

        let mut builder = LayoutBuilder::new();
        builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);
        let layout = builder.build(&context.device, vk::ShaderStageFlags::COMPUTE, None);
        builder.clear();

        let draw_image_descriptors = pool.allocate(&context.device, layout.layout);

        let mut writer = DescriptorWriter::new();

        writer.write_image(
            0,
            draw_image.view,
            vk::Sampler::null(),
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );

        writer.update_set(&context.device, draw_image_descriptors);

        (pool, layout, draw_image_descriptors)
    }

    fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
        log::debug!("Loading shader file {}", path.as_ref().display());
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module<P: AsRef<Path>>(device: &Device, path: P) -> vk::ShaderModule {
        let source = Self::read_shader_from_file(path);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&source);
        unsafe { device.create_shader_module(&create_info, None) }.unwrap()
    }

    fn initialize_effect_pipelines(
        context: &VkContext,
        image_dsl: &vk::DescriptorSetLayout,
        shaders: &[(vk::ShaderModule, &'static str)],
    ) -> (Vec<ComputeEffect>, PipelineLayout) {
        let push_constants = &[vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(core::slice::from_ref(image_dsl))
            .push_constant_ranges(push_constants);

        let layout = unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        (
            shaders
                .iter()
                .map(|(shader, name)| {
                    let pipeline = Self::initialize_shader_pipeline(context, shader, &layout);
                    let mut effect = ComputeEffect {
                        name,
                        pipeline,
                        data: ComputePushConstants {
                            data1: [1.0, 0.0, 0.0, 1.0],
                            data2: [0.0, 0.0, 1.0, 1.0],
                            data3: [0.0; 4],
                            data4: [0.0; 4],
                        },
                    };
                    if *name == "sky" {
                        effect.data.data1 = [0.1, 0.2, 0.4, 0.97];
                    }
                    effect
                })
                .collect_vec(),
            PipelineLayout::new(layout, context.device.clone()),
        )
    }

    fn initialize_shader_pipeline(
        context: &VkContext,
        shader_module: &vk::ShaderModule,
        pipeline_layout: &vk::PipelineLayout,
    ) -> Pipeline {
        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader_module)
            .name(c"main");

        let pipeline_info = &[vk::ComputePipelineCreateInfo::default()
            .layout(*pipeline_layout)
            .stage(shader_stage_info)];

        let pipeline = unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), pipeline_info, None)
        }
        .unwrap()[0];

        // Layout is shared across all compute effects and owned by effect_pipeline_layout;
        // passing null here so Pipeline::drop skips layout destruction (no-op per Vulkan spec).
        Pipeline::new(pipeline, vk::PipelineLayout::null(), context.device.clone())
    }

    pub(crate) fn resize_swapchain(&mut self) -> bool {
        self.wait_gpu_idle();
        if self.resources.window_size.0 == 0 || self.resources.window_size.1 == 0 {
            return true;
        }

        let old_handle = self.resources.swapchain.swapchain;
        self.resources.swapchain = Swapchain::recreate(
            &self.context,
            [self.resources.window_size.0, self.resources.window_size.1],
            old_handle,
        );

        self.resources.resize_requested = false;
        false
    }

    fn init_scene_data(context: &VkContext) -> DescriptorSetLayout {
        let mut builder = LayoutBuilder::new();
        builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);
        builder.build(
            &context.device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            None,
        )
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::trace!("Start: Dropping renderer");
        unsafe {
            self.context.device.device_wait_idle().unwrap();
            self.resources
                .material_descriptor_allocator
                .destroy_pools(&self.context.device);
            ManuallyDrop::drop(&mut self.resources);
            ManuallyDrop::drop(&mut self.context); // device + instance destroyed last
        }
        log::trace!("End: Dropping renderer");
    }
}
