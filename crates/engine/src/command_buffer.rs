use crate::context::QueueData;
use crate::sync::Fence;
use ash::Device;
use ash::vk;
use std::marker::PhantomData;

mod sealed {
    pub trait CmdState {}
}

pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Submitted;

impl sealed::CmdState for Initial {}
impl sealed::CmdState for Recording {}
impl sealed::CmdState for Executable {}
impl sealed::CmdState for Submitted {}

pub(crate) struct CommandBuffer<S: sealed::CmdState> {
    handle: vk::CommandBuffer,
    _state: PhantomData<S>,
}

impl CommandBuffer<Submitted> {
    /// # Safety
    /// Caller must ensure:
    /// - `handle` was returned by `vkAllocateCommandBuffers` and its pool is still live.
    /// - No other `CommandBuffer<_>` wrapper exists for this handle — aliasing would
    ///   desynchronize the typestate from the real Vulkan state.
    /// - The buffer is safe to reset: either the associated fence has been waited on,
    ///   or the buffer was just allocated and has never been submitted.
    pub(crate) unsafe fn wrap(handle: vk::CommandBuffer) -> Self {
        Self {
            handle,
            _state: PhantomData,
        }
    }

    pub(crate) fn reset(self, device: &Device) -> CommandBuffer<Initial> {
        unsafe { device.reset_command_buffer(self.handle, vk::CommandBufferResetFlags::default()) }
            .unwrap();
        CommandBuffer {
            handle: self.handle,
            _state: PhantomData,
        }
    }
}

impl CommandBuffer<Initial> {
    pub(crate) fn begin(
        self,
        device: &Device,
        flags: vk::CommandBufferUsageFlags,
    ) -> CommandBuffer<Recording> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        unsafe { device.begin_command_buffer(self.handle, &begin_info) }.unwrap();
        CommandBuffer {
            handle: self.handle,
            _state: PhantomData,
        }
    }
}

impl CommandBuffer<Recording> {
    pub(crate) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub(crate) fn end(self, device: &Device) -> CommandBuffer<Executable> {
        unsafe { device.end_command_buffer(self.handle) }.unwrap();
        CommandBuffer {
            handle: self.handle,
            _state: PhantomData,
        }
    }
}

impl CommandBuffer<Executable> {
    pub(crate) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub(crate) fn into_submitted(self) -> CommandBuffer<Submitted> {
        CommandBuffer {
            handle: self.handle,
            _state: PhantomData,
        }
    }
}

pub struct ImmediateSubmitData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: Fence,
    device: Device,
}

impl ImmediateSubmitData {
    pub(crate) fn new(
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        fence: Fence,
        device: Device,
    ) -> Self {
        Self {
            command_pool,
            command_buffer,
            fence,
            device,
        }
    }

    pub(crate) fn submit<F>(&self, gpu: &Device, graphics_queue: &QueueData, func: F)
    where
        F: FnOnce(&CommandBuffer<Recording>),
    {
        self.fence.reset();

        let cmd = unsafe { CommandBuffer::<Submitted>::wrap(self.command_buffer) };
        let cmd = cmd.reset(gpu);
        let cmd = cmd.begin(gpu, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        func(&cmd);

        let cmd = cmd.end(gpu);

        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd.handle())
            .device_mask(0)];

        let submit_info = &[vk::SubmitInfo2::default().command_buffer_infos(cmd_info)];

        unsafe { gpu.queue_submit2(graphics_queue.queue, submit_info, self.fence.handle()) }
            .unwrap();

        cmd.into_submitted(); // type-level marker: buffer is now pending, no Vulkan call

        assert!(
            self.fence.wait(1_000_000_000),
            "immediate submit fence timed out"
        );
    }
}

impl Drop for ImmediateSubmitData {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub(crate) fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
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
