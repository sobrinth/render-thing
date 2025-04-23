use crate::context::VkContext;
use crate::swapchain::Swapchain;
use ash::{Device, vk};
use winit::window::Window;

const FRAME_OVERLAP: u32 = 2;

pub struct VulkanRenderer {
    frame_number: u32,
    pub context: VkContext,

    swapchain: Swapchain,

    frames: Vec<FrameData>,
    graphics_queue: QueueData,
}

impl VulkanRenderer {
    pub fn initialize(window: &Window) -> Self {
        let (context, graphics_queue) = VkContext::initialize(window);

        let swapchain = Swapchain::create(
            &context,
            [window.inner_size().width, window.inner_size().height],
        );

        let frames = Self::create_framedata(&context, &graphics_queue);

        Self {
            frame_number: 0,
            context,
            swapchain,
            frames,
            graphics_queue,
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

        // transition swapchain-image to writable layout before rendering
        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        // create a clear color based on the frame-number
        let flash = f32::abs(f32::sin(self.frame_number as f32 / 1000.0));
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
                self.swapchain.images[image_index],
                vk::ImageLayout::GENERAL,
                &clear_color,
                ranges,
            )
        }

        // make swapchain image presentable
        Self::transition_image(
            gpu,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::GENERAL,
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
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::debug!("Start: Dropping renderer");
        self.wait_gpu_idle();
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
    
    pub fn clean_resources(&mut self) {
    }
}

#[derive(Copy, Clone)]
pub struct QueueData {
    pub queue: vk::Queue,
    pub family_index: u32,
}
