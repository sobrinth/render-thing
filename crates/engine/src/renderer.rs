use crate::context::VkContext;
use crate::descriptor;
use crate::swapchain::Swapchain;
use ash::{Device, vk};
use std::path::Path;
use vk_mem::Alloc;
use winit::window::Window;

const FRAME_OVERLAP: u32 = 2;

pub struct VulkanRenderer {
    frame_number: u32,
    pub context: VkContext,

    swapchain: Swapchain,

    frames: Vec<FrameData>,
    graphics_queue: QueueData,

    descriptor_allocator: descriptor::Allocator,

    draw_image: AllocatedImage,
    draw_image_descriptors: vk::DescriptorSet,
    draw_image_descriptor_layout: vk::DescriptorSetLayout,

    compute_shader: vk::ShaderModule,
    gradient_pipeline: vk::Pipeline,
    gradient_pipeline_layout: vk::PipelineLayout,
}

impl VulkanRenderer {
    pub fn initialize(window: &Window) -> Self {
        let (context, graphics_queue) = VkContext::initialize(window);

        let swapchain = Swapchain::create(
            &context,
            [window.inner_size().width, window.inner_size().height],
        );

        let frames = Self::create_framedata(&context, &graphics_queue);

        let draw_image = Self::create_draw_image(
            &context,
            (window.inner_size().width, window.inner_size().height),
        );

        let (descriptor_allocator, draw_image_descriptor_layout, draw_image_descriptors) =
            Self::init_descriptors(&context, &draw_image);

        let shader_code = Self::read_shader_from_file("assets/shaders/gradient.comp.spv");
        let shader_module = Self::create_shader_module(&context.device, &shader_code);

        let (gradient_pipeline, gradient_pipeline_layout) =
            Self::init_pipelines(&context, &draw_image_descriptor_layout, &shader_module);

        Self {
            frame_number: 0,
            context,
            swapchain,
            frames,
            graphics_queue,
            descriptor_allocator,
            draw_image,
            draw_image_descriptors,
            draw_image_descriptor_layout,
            compute_shader: shader_module,
            gradient_pipeline: gradient_pipeline,
            gradient_pipeline_layout: gradient_pipeline_layout,
        }
    }

    pub fn draw(&mut self) {
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
                frame.swapchain_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match res {
            Ok((image_index, _)) => image_index as usize,
            Err(err) => panic!("Failed to acquire next image. Cause: {err}"),
        };

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

        Self::draw_background(cmd, gpu, self.frame_number as f32, self.draw_image.image);

        // transition the draw image and the swapchain image into their correct transfer layouts.
        Self::transition_image(
            gpu,
            cmd,
            self.draw_image.image,
            vk::ImageLayout::GENERAL,
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

        // set the swapchain image to Layout::PRESENT so we can present it
        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        // finalize command buffer
        unsafe { gpu.end_command_buffer(cmd).unwrap() }

        // Prepare queue submission
        // we want to wait on the present_semaphore, as that is signaled when the swapchain is ready,
        // we will signal render_semaphore, to signal rendering has finished
        let cmd_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd)
            .device_mask(0);
        let cmd_infos = &[cmd_info];

        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.swapchain_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .device_index(0)
            .value(1);
        let wait_infos = &[wait_info];

        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.render_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
            .device_index(0)
            .value(1);
        let signal_infos = &[signal_info];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_infos)
            .signal_semaphore_infos(signal_infos)
            .command_buffer_infos(cmd_infos);

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
        let wait_semaphores = &[frame.render_semaphore];
        let image_indices = &[image_index as u32];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .wait_semaphores(wait_semaphores)
            .image_indices(image_indices);

        unsafe {
            self.swapchain
                .swapchain_fn
                .queue_present(self.graphics_queue.queue, &present_info)
        }
        .unwrap();

        // increase the number of frames drawn
        self.frame_number += 1;
    }

    fn draw_background(
        cmd: vk::CommandBuffer,
        gpu: &Device,
        frame_number: f32,
        draw_image: vk::Image,
    ) {
        // create a clear color based on the frame-number
        let flash = f32::abs(f32::sin(frame_number / 120.0));
        let clear_color = vk::ClearColorValue {
            float32: [0.0, 0.0, flash, 1.0],
        };

        let clear_subrange = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_array_layer(0)
            .layer_count(vk::REMAINING_ARRAY_LAYERS);

        // clear image
        let ranges = &[clear_subrange];
        unsafe {
            gpu.cmd_clear_color_image(
                cmd,
                draw_image,
                vk::ImageLayout::GENERAL,
                &clear_color,
                ranges,
            )
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
        let blit_region = vk::ImageBlit2::default()
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
            );
        let regions = &[blit_region];

        let blit_info = vk::BlitImageInfo2::default()
            .src_image(src)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(dst)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .filter(vk::Filter::LINEAR)
            .regions(regions);

        unsafe { device.cmd_blit_image2(cmd, &blit_info) }
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
            let render_semaphore = unsafe {
                context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            };
            let render_fence = unsafe { context.device.create_fence(&fence_info, None).unwrap() };

            let frame = FrameData {
                command_pool: pool,
                main_command_buffer: buffer,
                swapchain_semaphore,
                render_semaphore,
                render_fence,
            };
            frames.push(frame);
        }
        frames
    }

    pub fn wait_gpu_idle(&self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();
    }

    fn create_draw_image(context: &VkContext, window_size: (u32, u32)) -> AllocatedImage {
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
            unsafe { context.allocator.create_image(&create_info, &alloc_info) }.unwrap();

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
            _format: format,
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

    fn create_shader_module(device: &Device, shader_source: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(shader_source);
        unsafe { device.create_shader_module(&create_info, None) }.unwrap()
    }

    fn init_pipelines(
        context: &VkContext,
        image_dsl: &vk::DescriptorSetLayout,
        shader_module: &vk::ShaderModule,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        Self::init_background_pipeline(context, image_dsl, shader_module)
    }

    fn init_background_pipeline(
        context: &VkContext,
        image_dsl: &vk::DescriptorSetLayout,
        shader_module: &vk::ShaderModule,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let layouts = &[*image_dsl];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(layouts);

        let layout = unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader_module)
            .name(c"main");

        let pipeline_info = &[vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(shader_stage_info)];

        let compute_pipeline = unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), pipeline_info, None)
        }
        .unwrap()[0];

        (compute_pipeline, layout)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::debug!("Start: Dropping renderer");
        self.wait_gpu_idle();

        unsafe {
            self.context
                .device
                .destroy_pipeline(self.gradient_pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.gradient_pipeline_layout, None);
            self.context
                .device
                .destroy_shader_module(self.compute_shader, None)
        }
        self.descriptor_allocator.destroy_pool(&self.context.device);
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(self.draw_image_descriptor_layout, None)
        }
        self.draw_image
            .destroy(&self.context.device, &self.context.allocator);

        self.frames.iter_mut().for_each(|frame| {
            frame.destroy(&self.context.device);
        });

        unsafe { self.swapchain.destroy(&self.context.device) }
        log::debug!("End: Dropping renderer");
    }
}

#[derive(Copy, Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
}

impl FrameData {
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_semaphore(self.swapchain_semaphore, None);
            device.destroy_semaphore(self.render_semaphore, None);
            device.destroy_fence(self.render_fence, None);
        }
        self.clean_resources()
    }

    pub fn clean_resources(&mut self) {}
}

#[derive(Copy, Clone)]
pub struct QueueData {
    pub queue: vk::Queue,
    pub family_index: u32,
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk_mem::Allocation,
    pub extent: vk::Extent3D,
    pub _format: vk::Format,
}

impl AllocatedImage {
    pub fn destroy(&mut self, device: &Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}
