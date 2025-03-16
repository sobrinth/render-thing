use crate::engine::context::VkContext;
use crate::engine::swapchain::Swapchain;
use ash::vk;
use winit::window::Window;

const FRAME_OVERLAP: u32 = 2;

pub struct VulkanRenderer {
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
            context,
            swapchain,
            frames,
            graphics_queue,
        }
    }

    fn create_framedata(context: &VkContext, graphics_queue: &QueueData) -> Vec<FrameData> {
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

            let buffer = unsafe {
                context
                    .device
                    .allocate_command_buffers(&commandbuffer_info)
                    .unwrap()[0]
            };

            let frame = FrameData {
                command_pool: pool,
                main_command_buffer: buffer,
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
        self.frames.iter().for_each(|frame| unsafe {
            self.context
                .device
                .destroy_command_pool(frame.command_pool, None)
        });

        unsafe { self.swapchain.destroy(&self.context.device) }
        log::debug!("End: Dropping renderer");
    }
}

struct FrameData {
    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
}

pub struct QueueData {
    pub queue: vk::Queue,
    pub family_index: u32,
}
