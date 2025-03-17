use crate::engine::context::VkContext;
use crate::engine::swapchain::Swapchain;
use ash::vk::CommandBufferResetFlags;
use ash::{Device, vk};
use winit::window::Window;

const FRAME_OVERLAP: u32 = 2;

pub struct VulkanRenderer {
    frame_index: usize,
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
            frame_index: 0,
            context,
            swapchain,
            frames,
            graphics_queue,
        }
    }

    pub fn draw(&mut self) {
        let frame = self.frames[self.frame_index];
        self.frame_index = (self.frame_index + 1) % FRAME_OVERLAP as usize;

        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame.render_fence], true, 1_000_000_000)
                .unwrap();
            self.context
                .device
                .reset_fences(&[frame.render_fence])
                .unwrap();
        }

        let res = unsafe {
            self.swapchain.swapchain_fn.acquire_next_image(
                self.swapchain.swapchain,
                1_000_000_000,
                frame.swapchain_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match res {
            Ok((image_index, _)) => image_index,
            Err(err) => panic!("Failed to acquire next image. Cause: {err}"),
        };

        let cmd = frame.main_command_buffer;

        // Reset and begin command buffer for the frame
        unsafe {
            self.context
                .device
                .reset_command_buffer(cmd, CommandBufferResetFlags::default())
                .unwrap()
        }

        let cmd_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .unwrap()
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
    }
}

#[derive(Copy, Clone)]
pub struct QueueData {
    pub queue: vk::Queue,
    pub family_index: u32,
}
