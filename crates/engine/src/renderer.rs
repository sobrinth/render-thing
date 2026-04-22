use crate::camera::Camera;
use crate::command_buffer::{CommandBuffer, Recording, Submitted};
use crate::context::{QueueData, VkContext};
use crate::descriptor;
use crate::descriptor::{
    DescriptorSetLayout, DescriptorWriter, GrowableAllocator, LayoutBuilder, PoolSizeRatio,
};
use crate::frame::FrameData;
use crate::input::{ElementState, Key, MouseButton};
use crate::meshes::{MeshAsset, load_gltf_meshes};
use crate::pipeline::{Pipeline, PipelineBuilder, PipelineLayout};
use crate::primitives::{GPUDrawPushConstants, GPUMeshBuffers, GPUSceneData, Vertex};
use crate::resources::{AllocatedBuffer, AllocatedImage, Sampler};
use crate::swapchain::Swapchain;
use crate::sync::{Fence, Semaphore};
use crate::ui::UiContext;
use ash::{Device, vk};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;

pub(crate) const FRAME_OVERLAP: u32 = 2;

pub(crate) struct RendererResources {
    pub(crate) frame_number: u32,
    pub(crate) window_size: (u32, u32),
    pub(crate) render_scale: f32,
    pub(crate) render_size: (u32, u32),
    pub(crate) resize_requested: bool,
    pub(crate) mouse_pos: (i32, i32),
    pub(crate) main_camera: Camera,
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
    pub(crate) mesh_pipeline: Pipeline,
    pub(crate) scene_data: GPUSceneData,
    pub(crate) scene_data_layout: DescriptorSetLayout,
    pub(crate) meshes: Option<Vec<MeshAsset>>,
    pub(crate) active_mesh: usize,
    pub(crate) default_sampler_nearest: Sampler,
    pub(crate) default_sampler_linear: Sampler,
    pub(crate) white_image: AllocatedImage,
    pub(crate) grey_image: AllocatedImage,
    pub(crate) black_image: AllocatedImage,
    pub(crate) checkerboard_image: AllocatedImage,
    pub(crate) single_image_layout: DescriptorSetLayout,
}

pub(crate) struct VulkanRenderer {
    pub(crate) resources: ManuallyDrop<RendererResources>,
    pub(crate) context: ManuallyDrop<VkContext>,
}

impl VulkanRenderer {
    pub(crate) fn initialize(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        window_size: (u32, u32),
    ) -> Self {
        let (context, graphics_queue, gpu_alloc) = VkContext::initialize(window);

        let swapchain = Swapchain::create(&context, [window_size.0, window_size.1]);
        let ui = UiContext::initialize(&context.device, &gpu_alloc, swapchain.properties);

        let immediate_submit = Self::create_immediate_submit_data(&context, &graphics_queue);

        let frames = Self::create_framedata(&context, &gpu_alloc, &graphics_queue);

        let draw_image = AllocatedImage::create(
            &context,
            &gpu_alloc,
            (2560, 1440),
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

        let mut dsl_builder = LayoutBuilder::new();
        dsl_builder.add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        let single_image_layout =
            dsl_builder.build(&context.device, vk::ShaderStageFlags::FRAGMENT, None);

        let mesh_pipeline = Self::initialize_mesh_pipeline(
            &context,
            &draw_image,
            &depth_image,
            &single_image_layout.layout,
        );

        // default data
        const WHITE: u32 = u32::from_le_bytes([255, 255, 255, 255]);
        const GREY: u32 = u32::from_le_bytes([128, 128, 128, 255]);
        const MAGENTA: u32 = u32::from_le_bytes([255, 0, 255, 255]);
        const BLACK: u32 = u32::from_le_bytes([0, 0, 0, 0]);

        let white_image = AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[WHITE],
            (1, 1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let grey_image = AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[GREY],
            (1, 1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let black_image = AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &[BLACK],
            (1, 1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            false,
        );

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

        let checkerboard_image = AllocatedImage::create_from_data(
            &gpu_alloc,
            &context,
            &immediate_submit,
            &graphics_queue,
            &checkerboard_data,
            (16, 16),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        //sampler here??
        let mut sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let nearest_handle = unsafe { context.device.create_sampler(&sampler, None) }.unwrap();
        let default_sampler_nearest = Sampler::new(nearest_handle, context.device.clone());

        sampler = sampler.mag_filter(vk::Filter::LINEAR);
        sampler = sampler.min_filter(vk::Filter::LINEAR);

        let linear_handle = unsafe { context.device.create_sampler(&sampler, None) }.unwrap();
        let default_sampler_linear = Sampler::new(linear_handle, context.device.clone());
        let main_camera = Camera::new();

        let mut renderer = Self {
            resources: ManuallyDrop::new(RendererResources {
                frame_number: 0,
                window_size,
                render_scale: 1f32,
                render_size: (0, 0),
                resize_requested: false,
                mouse_pos: (0, 0),
                main_camera,
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
                mesh_pipeline,
                scene_data: GPUSceneData::default(),
                scene_data_layout,
                meshes: None,
                active_mesh: 0,
                default_sampler_nearest,
                default_sampler_linear,
                white_image,
                grey_image,
                black_image,
                checkerboard_image,
                single_image_layout,
            }),
            context: ManuallyDrop::new(context),
        };
        renderer.resources.meshes = load_gltf_meshes(&renderer, "assets/models/basicmesh.glb");

        renderer
    }

    pub(crate) fn egui_context(&self) -> egui::Context {
        self.resources.ui_context.ctx.clone()
    }

    pub(crate) fn on_key_event(&mut self, key_event: (ElementState, Key)) {
        // TODO: UI should intercept if focused input and stuff
        self.resources.main_camera.handle_key_event(key_event);
    }

    pub(crate) fn on_mouse_event(&mut self, new_pos: (i32, i32)) {
        // TODO: UI should intercept if mouse is over it
        let old_pos = self.resources.mouse_pos;
        self.resources
            .main_camera
            .handle_mouse_event(old_pos, new_pos);
        self.resources.mouse_pos = new_pos;
    }

    pub(crate) fn on_mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        // TODO: UI should intercept if mouse is over it
        self.resources
            .main_camera
            .handle_mouse_button_event(button, state);
    }

    fn immediate_submit<F>(
        gpu: &Device,
        imm_data: &ImmediateSubmitData,
        graphics_queue: &QueueData,
        func: F,
    ) where
        F: FnOnce(&CommandBuffer<Recording>),
    {
        imm_data.fence.reset();

        let cmd = unsafe { CommandBuffer::<Submitted>::wrap(imm_data.command_buffer) };
        let cmd = cmd.reset(gpu);
        let cmd = cmd.begin(gpu, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        func(&cmd);

        let cmd = cmd.end(gpu);

        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd.handle())
            .device_mask(0)];

        let submit = &[vk::SubmitInfo2::default().command_buffer_infos(cmd_info)];

        unsafe { gpu.queue_submit2(graphics_queue.queue, submit, imm_data.fence.handle()) }
            .unwrap();

        cmd.into_submitted(); // type-level marker: buffer is now pending, no Vulkan call

        assert!(
            imm_data.fence.wait(1_000_000_000),
            "immediate submit fence timed out"
        );
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

        ImmediateSubmitData {
            command_buffer,
            command_pool,
            fence: Fence::new_signaled(&context.device),
            device: context.device.clone(),
        }
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
                size_of::<GPUSceneData>() as u64,
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

    fn initialize_mesh_pipeline(
        context: &VkContext,
        draw_image: &AllocatedImage,
        depth_image: &AllocatedImage,
        image_dsl: &vk::DescriptorSetLayout,
    ) -> Pipeline {
        let frag_module =
            Self::create_shader_module(&context.device, "assets/shaders/tex_image.frag.spv");
        let vert_module = Self::create_shader_module(
            &context.device,
            "assets/shaders/colored_triangle_mesh.vert.spv",
        );

        let buffer_range = &[vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let idsl = &[*image_dsl];

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(buffer_range)
            .set_layouts(idsl);

        let pipeline_layout =
            unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        let mut builder = PipelineBuilder::init();

        // use layout
        builder.pipeline_layout = pipeline_layout;
        // set shader modules
        builder.set_shaders(vert_module, frag_module);
        // Draw triangles
        builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        // Fill triangles
        builder.set_polygon_mode(vk::PolygonMode::FILL);
        builder.set_cull_mode(vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE);
        // no multisampling
        builder.set_multisampling_none();
        // no blending
        builder.disable_blending();
        // builder.enable_blending_additive();
        // no depth testing
        // builder.disable_depth_test();
        builder.enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL);

        // connect the image format from draw image
        builder.set_color_attachment_format(draw_image.format);
        builder.set_depth_format(depth_image.format);

        let pipeline = builder.build(&context.device);

        // clean up modules
        unsafe {
            context.device.destroy_shader_module(vert_module, None);
            context.device.destroy_shader_module(frag_module, None);
        };
        Pipeline::new(pipeline, pipeline_layout, context.device.clone())
    }

    pub(crate) fn upload_mesh(&self, indices: &[u32], vertices: &[Vertex]) -> GPUMeshBuffers {
        Self::upload_mesh_internal(
            &self.resources.gpu_alloc,
            &self.context,
            &self.resources.immediate_submit,
            &self.resources.graphics_queue,
            indices,
            vertices,
        )
    }

    fn upload_mesh_internal(
        gpu_alloc: &Arc<vk_mem::Allocator>,
        context: &VkContext,
        imm_data: &ImmediateSubmitData,
        graphics_queue: &QueueData,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> GPUMeshBuffers {
        let vertex_buffer_size = size_of_val(vertices);
        let index_buffer_size = size_of_val(indices);

        let vertex_buffer = AllocatedBuffer::create(
            gpu_alloc,
            vertex_buffer_size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
            None,
        );
        let index_buffer = AllocatedBuffer::create(
            gpu_alloc,
            index_buffer_size as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            None,
        );

        let device_address_info =
            vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let device_address = unsafe {
            context
                .device
                .get_buffer_device_address(&device_address_info)
        };

        let meshes = GPUMeshBuffers {
            vertex_buffer,
            index_buffer,
            vertex_buffer_address: device_address,
        };

        let mut staging = AllocatedBuffer::create(
            gpu_alloc,
            (vertex_buffer_size + index_buffer_size) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            Some(
                vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
        );

        let dst_data = unsafe { gpu_alloc.map_memory(&mut staging.allocation) }.unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                dst_data,
                vertex_buffer_size,
            );
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                dst_data.add(vertex_buffer_size),
                index_buffer_size,
            );
        };
        unsafe { gpu_alloc.unmap_memory(&mut staging.allocation) };

        Self::immediate_submit(&context.device, imm_data, graphics_queue, |cmd| {
            let vertex_copy = &[vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(0)
                .size(vertex_buffer_size as u64)];
            unsafe {
                context.device.cmd_copy_buffer(
                    cmd.handle(),
                    staging.buffer,
                    meshes.vertex_buffer.buffer,
                    vertex_copy,
                )
            }

            let index_copy = &[vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(vertex_buffer_size as u64)
                .size(index_buffer_size as u64)];

            unsafe {
                context.device.cmd_copy_buffer(
                    cmd.handle(),
                    staging.buffer,
                    meshes.index_buffer.buffer,
                    index_copy,
                )
            }
        });
        meshes
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
            ManuallyDrop::drop(&mut self.resources);
            ManuallyDrop::drop(&mut self.context); // device + instance destroyed last
        }
        log::trace!("End: Dropping renderer");
    }
}

impl AllocatedImage {
    pub(crate) fn create_from_data(
        gpu_alloc: &Arc<vk_mem::Allocator>,
        context: &VkContext,
        imm_data: &ImmediateSubmitData,
        graphics_queue: &QueueData,
        data: &[u32],
        image_resolution: (u32, u32),
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
        mip_mapped: bool,
    ) -> AllocatedImage {
        let extent = vk::Extent3D {
            width: image_resolution.0,
            height: image_resolution.1,
            depth: 1,
        };

        let data_size = size_of_val(data);

        let mut upload_buffer = AllocatedBuffer::create(
            gpu_alloc,
            data_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferDevice,
            Some(
                vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
        );

        let new_image = AllocatedImage::create(
            context,
            gpu_alloc,
            image_resolution,
            format,
            usage,
            aspect_flags,
            mip_mapped,
        );

        let dst_data = unsafe { gpu_alloc.map_memory(&mut upload_buffer.allocation) }.unwrap();
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst_data, data_size) };
        unsafe { gpu_alloc.unmap_memory(&mut upload_buffer.allocation) };

        VulkanRenderer::immediate_submit(&context.device, imm_data, graphics_queue, |cmd| {
            VulkanRenderer::transition_image(
                &context.device,
                cmd.handle(),
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: aspect_flags,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent);

            unsafe {
                context.device.cmd_copy_buffer_to_image(
                    cmd.handle(),
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                )
            }

            VulkanRenderer::transition_image(
                &context.device,
                cmd.handle(),
                new_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        new_image
    }
}

pub struct ImmediateSubmitData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: Fence,
    device: Device,
}

impl Drop for ImmediateSubmitData {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct ComputePushConstants {
    pub data1: [f32; 4],
    pub data2: [f32; 4],
    pub data3: [f32; 4],
    pub data4: [f32; 4],
}

pub(crate) struct ComputeEffect {
    pub name: &'static str,
    pub(crate) pipeline: Pipeline,
    pub data: ComputePushConstants,
}
