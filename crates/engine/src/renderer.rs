use crate::context::VkContext;
use crate::pipeline::PipelineBuilder;
use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::swapchain::Swapchain;
use crate::ui::UiContext;
use crate::{descriptor, ui};
use ash::{Device, vk};
use itertools::Itertools;
use std::mem;
use std::path::Path;
use std::sync::Arc;
use vk_mem::Alloc;
use winit::event::WindowEvent;
use winit::window::Window;

pub(crate) const FRAME_OVERLAP: u32 = 2;

pub(crate) struct VulkanRenderer {
    frame_number: u32,
    gpu_alloc: Arc<vk_mem::Allocator>,
    ui_context: UiContext,
    pub context: VkContext,

    swapchain: Swapchain,

    frames: Vec<FrameData>,
    graphics_queue: QueueData,

    descriptor_allocator: descriptor::Allocator,

    immediate_submit: ImmediateSubmitData,

    draw_image: AllocatedImage,
    draw_image_descriptors: vk::DescriptorSet,
    draw_image_descriptor_layout: vk::DescriptorSetLayout,

    effect_pipeline_layout: vk::PipelineLayout,
    background_effects: Vec<ComputeEffect>,
    active_background_effect: usize,

    triangle_pipeline: vk::Pipeline,
    triangle_pipeline_layout: vk::PipelineLayout,
}

impl<'a> VulkanRenderer {
    pub(crate) fn initialize(window: &'a Window) -> Self {
        let (context, graphics_queue, gpu_alloc) = VkContext::initialize(window);

        let swapchain = Swapchain::create(
            &context,
            [window.inner_size().width, window.inner_size().height],
        );
        let ui = UiContext::initialize(window, &context.device, &gpu_alloc, swapchain.properties);

        let immediate_submit = Self::create_immediate_submit_data(&context, &graphics_queue);

        let frames = Self::create_framedata(&context, &graphics_queue);

        let draw_image = Self::create_draw_image(
            &context,
            &gpu_alloc,
            (window.inner_size().width, window.inner_size().height),
        );

        let (descriptor_allocator, draw_image_descriptor_layout, draw_image_descriptors) =
            Self::init_descriptors(&context, &draw_image);

        let gradient_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/gradient.comp.spv");

        let gradient_color_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/gradient_color.comp.spv");

        let sky_shader_module =
            Self::create_shader_module(&context.device, "assets/shaders/sky.comp.spv");

        let (effects, effect_pipeline_layout) = Self::initialize_effect_pipelines(
            &context,
            &draw_image_descriptor_layout,
            &vec![
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

        let (triangle_pipeline, triangle_pipeline_layout) =
            Self::initialize_triangle_pipeline(&context, &draw_image);

        Self {
            frame_number: 0,
            gpu_alloc,
            context,
            ui_context: ui,
            swapchain,
            frames,
            immediate_submit,
            graphics_queue,
            descriptor_allocator,
            draw_image,
            draw_image_descriptors,
            draw_image_descriptor_layout,
            effect_pipeline_layout,
            background_effects: effects,
            active_background_effect: 0,
            triangle_pipeline,
            triangle_pipeline_layout,
        }
    }

    pub(crate) fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self
            .ui_context
            .state
            .as_mut()
            .unwrap()
            .on_window_event(window, event);
    }

    pub(crate) fn draw(&mut self, _window: &Window) {
        const ONE_SECOND: u64 = 1_000_000_000;
        let frame_index = self.frame_number % FRAME_OVERLAP;
        let mut frame = self.frames[frame_index as usize];

        let cmd = frame.main_command_buffer;
        let gpu = &self.context.device;

        unsafe {
            // wait for the GPU to have finished the last rendering of this frame.
            gpu.wait_for_fences(&[frame.render_fence], true, ONE_SECOND)
                .unwrap();
            gpu.reset_fences(&[frame.render_fence]).unwrap();
        }
        frame.clean_resources();

        let res = unsafe {
            self.swapchain.swapchain_fn.acquire_next_image(
                self.swapchain.swapchain,
                ONE_SECOND,
                frame.acquire_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match res {
            Ok((image_index, _)) => image_index as usize,
            Err(err) => panic!("Failed to acquire next image. Cause: {err}"),
        };

        // BEFORE FRAME
        let ui_primitives = ui::before_frame(
            &mut self.ui_context,
            _window,
            &self.graphics_queue,
            &frame,
            (
                &mut self.background_effects[self.active_background_effect],
                &mut self.active_background_effect,
            ),
        );

        // Reset and begin command buffer for the frame
        unsafe {
            gpu.reset_command_buffer(cmd, vk::CommandBufferResetFlags::default())
                .unwrap()
        }

        let cmd_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { gpu.begin_command_buffer(cmd, &cmd_begin_info).unwrap() }

        // transition the main draw image to Layout::GENERAL so we can draw into it.
        // we will overwrite the contents, so we don't care about the old layout
        Self::transition_image(
            gpu,
            cmd,
            self.draw_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        self.draw_background(cmd, gpu);

        Self::transition_image(
            gpu,
            cmd,
            self.draw_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        self.draw_geometry(cmd, gpu);

        // transition the draw image and the swapchain image into their correct transfer layouts.
        Self::transition_image(
            gpu,
            cmd,
            self.draw_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        // copy the draw image to the swapchain image
        Self::copy_image_to_image(
            gpu,
            cmd,
            self.draw_image.image,
            self.swapchain.images[image_index],
            vk::Extent2D {
                height: self.draw_image.extent.height,
                width: self.draw_image.extent.width,
            },
            self.swapchain.properties.extent,
        );

        // DO THE UI RENDER

        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.properties.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe { gpu.cmd_begin_rendering(cmd, &rendering_info) }

        ui::render(
            &mut self.ui_context,
            cmd,
            self.swapchain.properties.extent,
            ui_primitives,
        );

        unsafe { gpu.cmd_end_rendering(cmd) }

        // set the swapchain image to Layout::PRESENT so we can present it
        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        // finalize command buffer
        unsafe { gpu.end_command_buffer(cmd).unwrap() }

        // Prepare queue submission
        // we want to wait on the present_semaphore, as that is signaled when the swapchain is ready,
        // we will signal render_semaphore, to signal rendering has finished
        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd)
            .device_mask(0)];

        let wait_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.acquire_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .device_index(0)
            .value(1)];

        let signal_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.swapchain.semaphores[image_index])
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
            .device_index(0)
            .value(1)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_info)
            .signal_semaphore_infos(signal_info)
            .command_buffer_infos(cmd_info);

        // submit a command buffer to the queue and execute it.
        // render_fence will now block until the graphic commands finish execution
        unsafe {
            gpu.queue_submit2(
                self.graphics_queue.queue,
                &[submit_info],
                frame.render_fence,
            )
            .unwrap()
        }

        // Prepare presentation
        // this will put the image just rendered to into the visible window
        // wait on render_semaphore for that, as it's necessary that drawing commands have finished
        let swapchains = &[self.swapchain.swapchain];
        let wait_semaphores = &[self.swapchain.semaphores[image_index]];
        let image_indices = &[image_index as u32];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .wait_semaphores(wait_semaphores)
            .image_indices(image_indices);

        // TODO db: Maybe use `VK_EXT_swapchain_maintenance1` to be able to use a fence here and "circumvent" the semaphore per image
        unsafe {
            self.swapchain
                .swapchain_fn
                .queue_present(self.graphics_queue.queue, &present_info)
        }
        .unwrap();

        ui::after_frame(&mut self.ui_context);

        // increase the number of frames drawn
        self.frame_number += 1;
    }

    #[allow(dead_code)]
    fn immediate_submit<F>(&self, func: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        let gpu = &self.context.device;
        let imm_data = &self.immediate_submit;

        unsafe {
            gpu.reset_fences(&[imm_data.fence]).unwrap();
            gpu.reset_command_buffer(
                imm_data.command_buffer,
                vk::CommandBufferResetFlags::default(),
            )
            .unwrap();
        }

        let cmd_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            gpu.begin_command_buffer(imm_data.command_buffer, &cmd_begin_info)
                .unwrap()
        }

        func(imm_data.command_buffer);

        unsafe { gpu.end_command_buffer(imm_data.command_buffer).unwrap() }

        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(imm_data.command_buffer)
            .device_mask(0)];

        let submit = &[vk::SubmitInfo2::default().command_buffer_infos(cmd_info)];

        unsafe {
            gpu.queue_submit2(self.graphics_queue.queue, submit, imm_data.fence)
                .unwrap()
        }

        unsafe {
            gpu.wait_for_fences(&[imm_data.fence], true, 1_000_000_000)
                .unwrap()
        }
    }

    fn draw_background(&self, cmd: vk::CommandBuffer, gpu: &Device) {
        let active_background = self.background_effects[self.active_background_effect];

        // bind the gradient drawing compute pipeline
        unsafe {
            gpu.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                active_background.pipeline,
            )
        }

        // bind the descriptor set containing the draw image for the compute pipeline
        unsafe {
            gpu.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.effect_pipeline_layout,
                0,
                &[self.draw_image_descriptors],
                &[],
            )
        }

        unsafe {
            gpu.cmd_push_constants(
                cmd,
                self.effect_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &mem::transmute::<ComputePushConstants, [u8; 64]>(active_background.data),
            )
        }

        // execute the compute pipeline dispatch using a 16x16 workgroup size, so we divide by 16
        unsafe {
            gpu.cmd_dispatch(
                cmd,
                f64::ceil(self.draw_image.extent.width as f64 / 16.0) as u32,
                f64::ceil(self.draw_image.extent.height as f64 / 16.0) as u32,
                1,
            )
        }
    }

    fn draw_geometry(&self, cmd: vk::CommandBuffer, gpu: &Device) {
        // begin a render pass with the draw image
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.draw_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD) // maybe clear?
            .store_op(vk::AttachmentStoreOp::STORE);

        let render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    height: self.draw_image.extent.height,
                    width: self.draw_image.extent.width,
                },
            })
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&color_attachment));

        unsafe {
            gpu.cmd_begin_rendering(cmd, &render_info);
            gpu.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.triangle_pipeline);
        }

        // dynamic viewport and scissor
        let viewport = vk::Viewport {
            x: 0f32,
            y: 0f32,
            width: self.draw_image.extent.width as f32,
            height: self.draw_image.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        unsafe { gpu.cmd_set_viewport(cmd, 0, &[viewport]) }

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                height: self.draw_image.extent.height,
                width: self.draw_image.extent.width,
            },
        };

        unsafe { gpu.cmd_set_scissor(cmd, 0, &[scissor]) }

        // DRAW GEOMETRY
        unsafe {
            gpu.cmd_draw(cmd, 3, 1, 0, 0);
            gpu.cmd_end_rendering(cmd);
        }
    }

    fn transition_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        current_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if current_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_array_layer(0)
            .layer_count(vk::REMAINING_ARRAY_LAYERS);

        let image_barrier = vk::ImageMemoryBarrier2::default()
            // Using ALL_COMMANDS is inefficient as it will stop gpu commands completely when it arrives at the barrier
            // for more complex applications use correct stage masks
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(current_layout)
            .new_layout(new_layout)
            .subresource_range(subresource_range)
            .image(image);

        let barriers = [image_barrier];

        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) }
    }

    fn copy_image_to_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        dst: vk::Image,
        src_size: vk::Extent2D,
        dst_size: vk::Extent2D,
    ) {
        let blit_region = &[vk::ImageBlit2::default()
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_size.width as i32,
                    y: src_size.height as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_size.width as i32,
                    y: dst_size.height as i32,
                    z: 1,
                },
            ])
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )];

        let blit_info = vk::BlitImageInfo2::default()
            .src_image(src)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(dst)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .filter(vk::Filter::LINEAR)
            .regions(blit_region);

        unsafe { device.cmd_blit_image2(cmd, &blit_info) }
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

        let command_buffer = unsafe {
            context
                .device
                .allocate_command_buffers(&commandbuffer_info)
                .unwrap()[0]
        };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { context.device.create_fence(&fence_info, None).unwrap() };

        ImmediateSubmitData {
            command_buffer,
            command_pool,
            fence,
        }
    }

    fn create_framedata(context: &VkContext, graphics_queue: &QueueData) -> Vec<FrameData> {
        let commandpool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut frames = Vec::with_capacity(FRAME_OVERLAP as usize);

        for _ in 0..FRAME_OVERLAP {
            let pool =
                unsafe { context.device.create_command_pool(&commandpool_info, None) }.unwrap();

            let commandbuffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY);

            let buffer = unsafe {
                context
                    .device
                    .allocate_command_buffers(&commandbuffer_info)
                    .unwrap()[0]
            };

            let swapchain_semaphore = unsafe {
                context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            };

            let render_fence = unsafe { context.device.create_fence(&fence_info, None).unwrap() };

            let frame = FrameData {
                command_pool: pool,
                main_command_buffer: buffer,
                acquire_semaphore: swapchain_semaphore,
                render_fence,
            };
            frames.push(frame);
        }
        frames
    }

    pub(crate) fn wait_gpu_idle(&self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();
    }

    fn create_draw_image(
        context: &VkContext,
        gpu_alloc: &vk_mem::Allocator,
        window_size: (u32, u32),
    ) -> AllocatedImage {
        let extent = vk::Extent3D {
            width: window_size.0,
            height: window_size.1,
            depth: 1,
        };

        let format = vk::Format::R16G16B16A16_SFLOAT;
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (image, allocation) =
            unsafe { gpu_alloc.create_image(&create_info, &alloc_info) }.unwrap();

        let create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .image(image)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            );

        let view = unsafe { context.device.create_image_view(&create_info, None) }.unwrap();

        AllocatedImage {
            image,
            view,
            allocation,
            extent,
            format,
        }
    }

    fn init_descriptors(
        context: &VkContext,
        draw_image: &AllocatedImage,
    ) -> (
        descriptor::Allocator,
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
    ) {
        let pool_sizes = vec![descriptor::PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1f32,
        }];
        let pool = descriptor::Allocator::init_pool(&context.device, 10, pool_sizes);

        let mut builder = descriptor::LayoutBuilder::new();
        builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);
        let layout = builder.build(&context.device, vk::ShaderStageFlags::COMPUTE, None);
        builder.clear();

        let draw_image_descriptors = pool.allocate(&context.device, layout);

        let image_info = &[vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(draw_image.view)];

        let write_info = &[vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(draw_image_descriptors)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(image_info)];

        unsafe { context.device.update_descriptor_sets(write_info, &[]) }
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
        shaders: &Vec<(vk::ShaderModule, &'static str)>,
    ) -> (Vec<ComputeEffect>, vk::PipelineLayout) {
        let push_constants = &[vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let layouts = &[*image_dsl];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(layouts)
            .push_constant_ranges(push_constants);

        let layout = unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        (
            shaders
                .iter()
                .map(|(shader, name)| {
                    let pipeline = Self::initialize_shader_pipeline(context, shader, &layout);
                    let mut effect = ComputeEffect {
                        layout,
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
            layout,
        )
    }

    fn initialize_shader_pipeline(
        context: &VkContext,
        shader_module: &vk::ShaderModule,
        pipeline_layout: &vk::PipelineLayout,
    ) -> vk::Pipeline {
        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader_module)
            .name(c"main");

        let pipeline_info = &[vk::ComputePipelineCreateInfo::default()
            .layout(*pipeline_layout)
            .stage(shader_stage_info)];

        unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), pipeline_info, None)
        }
        .unwrap()[0]
    }

    fn initialize_triangle_pipeline(
        context: &VkContext,
        draw_image: &AllocatedImage,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let frag_module =
            Self::create_shader_module(&context.device, "assets/shaders/colored_triangle.frag.spv");
        let vert_module =
            Self::create_shader_module(&context.device, "assets/shaders/colored_triangle.vert.spv");

        let layout_info = vk::PipelineLayoutCreateInfo::default();

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
        // no backface culling
        builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE);
        // no multisampling
        builder.set_multisampling_none();
        // no blending
        builder.disable_blending();
        // no depth testing
        builder.disable_depth_test();

        // connect the image format from draw image
        builder.set_color_attachment_format(draw_image.format);
        builder.set_depth_format(vk::Format::UNDEFINED);

        let pipeline = builder.build(&context.device);

        // clean up modules
        unsafe {
            context.device.destroy_shader_module(vert_module, None);
            context.device.destroy_shader_module(frag_module, None);
        };
        (pipeline, pipeline_layout)
    }

    fn upload_mesh(&self, indices: &[u32], vertices: &[Vertex]) -> GPUMeshBuffers {
        let vertex_buffer_size = size_of_val(vertices);
        let index_buffer_size = size_of_val(indices);

        let vertex_buffer = AllocatedBuffer::create(
            &self.gpu_alloc,
            vertex_buffer_size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::GpuOnly,
        );
        let index_buffer = AllocatedBuffer::create(
            &self.gpu_alloc,
            index_buffer_size as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::GpuOnly,
        );

        let device_address_info =
            vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let device_address = unsafe {
            self.context
                .device
                .get_buffer_device_address(&device_address_info)
        };

        let meshes = GPUMeshBuffers {
            vertex_buffer,
            index_buffer,
            vertex_buffer_address: device_address,
        };

        let mut staging = AllocatedBuffer::create(
            &self.gpu_alloc,
            (vertex_buffer_size + index_buffer_size) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::CpuOnly,
        );

        unsafe {
            let dst_data = self.gpu_alloc.map_memory(&mut staging.allocation).unwrap();
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
            self.gpu_alloc.unmap_memory(&mut staging.allocation);
        };

        self.immediate_submit(|cmd| {
            let vertex_copy = &[vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(0)
                .size(vertex_buffer_size as u64)];
            unsafe {
                self.context.device.cmd_copy_buffer(
                    cmd,
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
                self.context.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    meshes.index_buffer.buffer,
                    index_copy,
                )
            }
        });
        staging.destroy(&self.gpu_alloc);

        meshes
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::debug!("Start: Dropping renderer");
        self.wait_gpu_idle();

        unsafe {
            self.context
                .device
                .destroy_pipeline(self.triangle_pipeline, None);

            self.context
                .device
                .destroy_pipeline_layout(self.triangle_pipeline_layout, None);

            self.context
                .device
                .destroy_pipeline_layout(self.effect_pipeline_layout, None);
        }
        self.background_effects.iter_mut().for_each(|effect| {
            effect.destroy(&self.context.device);
        });
        self.descriptor_allocator
            .clear_descriptors(&self.context.device);
        self.descriptor_allocator.destroy_pool(&self.context.device);
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(self.draw_image_descriptor_layout, None)
        }
        self.draw_image
            .destroy(&self.context.device, &self.gpu_alloc);

        self.frames.iter_mut().for_each(|frame| {
            frame.destroy(&self.context.device);
        });

        self.immediate_submit.destroy(&self.context.device);

        self.swapchain.destroy(&self.context.device);
        log::debug!("End: Dropping renderer");
    }
}

#[derive(Copy, Clone)]
pub struct FrameData {
    pub command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    acquire_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
}

impl FrameData {
    pub(crate) fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_semaphore(self.acquire_semaphore, None);
            device.destroy_fence(self.render_fence, None);
        }
        self.clean_resources()
    }

    pub(crate) fn clean_resources(&mut self) {}
}

#[derive(Copy, Clone)]
pub(crate) struct QueueData {
    pub(crate) queue: vk::Queue,
    pub(crate) family_index: u32,
}

pub(crate) struct AllocatedImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) extent: vk::Extent3D,
    pub(crate) format: vk::Format,
}

impl AllocatedImage {
    pub(crate) fn destroy(&mut self, device: &Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ImmediateSubmitData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl ImmediateSubmitData {
    pub(crate) fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.fence, None);
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

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct ComputeEffect {
    pub name: &'static str,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    pub data: ComputePushConstants,
}

impl ComputeEffect {
    pub fn destroy(&mut self, device: &Device) {
        unsafe { device.destroy_pipeline(self.pipeline, None) }
    }
}

#[derive(Debug, Clone)]
pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    info: vk_mem::AllocationInfo,
}

impl AllocatedBuffer {
    pub fn create(
        allocator: &vk_mem::Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        let alloc_create_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (buffer, allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_create_info) }.unwrap();
        let info = allocator.get_allocation_info(&allocation);

        AllocatedBuffer {
            buffer,
            allocation,
            info,
        }
    }

    pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        unsafe { allocator.destroy_buffer(self.buffer, &mut self.allocation) }
    }
}
