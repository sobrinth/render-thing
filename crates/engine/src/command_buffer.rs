use crate::context::QueueData;
use crate::resources::AllocatedBuffer;
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

enum BatchCmd {
    Idle(CommandBuffer<Submitted>),
    Recording(CommandBuffer<Recording>),
}

/// Accumulates upload commands into a single command buffer that is submitted
/// lazily: when pending staging memory crosses [`Self::FLUSH_THRESHOLD_BYTES`]
/// or when `flush` runs at the start of a frame. Staging buffers are owned by
/// the batch so their memory stays valid until the flush fence signals.
pub struct UploadBatch {
    command_pool: vk::CommandPool,
    cmd: Option<BatchCmd>,
    staging: Vec<AllocatedBuffer>,
    staging_bytes: vk::DeviceSize,
    fence: Fence,
    device: Device,
}

impl UploadBatch {
    /// Bounds peak host memory held by pending staging buffers.
    const FLUSH_THRESHOLD_BYTES: vk::DeviceSize = 64 * 1024 * 1024;

    pub(crate) fn new(
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        fence: Fence,
        device: Device,
    ) -> Self {
        Self {
            command_pool,
            // Safety: freshly allocated buffer, sole wrapper, never submitted.
            cmd: Some(BatchCmd::Idle(unsafe {
                CommandBuffer::wrap(command_buffer)
            })),
            staging: Vec::new(),
            staging_bytes: 0,
            fence,
            device,
        }
    }

    /// Records upload commands and takes ownership of the staging buffer they
    /// read from, keeping it alive until the batch is flushed.
    pub(crate) fn record_upload<F>(
        &mut self,
        gpu: &Device,
        graphics_queue: &QueueData,
        staging: AllocatedBuffer,
        func: F,
    ) where
        F: FnOnce(&CommandBuffer<Recording>),
    {
        let cmd = match self.cmd.take().unwrap() {
            BatchCmd::Recording(cmd) => cmd,
            BatchCmd::Idle(cmd) => cmd
                .reset(gpu)
                .begin(gpu, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        };

        func(&cmd);

        self.cmd = Some(BatchCmd::Recording(cmd));
        self.staging_bytes += staging.info.size;
        self.staging.push(staging);

        if self.staging_bytes >= Self::FLUSH_THRESHOLD_BYTES {
            self.flush(gpu, graphics_queue);
        }
    }

    /// Submits pending uploads and waits for completion; no-op when idle.
    pub(crate) fn flush(&mut self, gpu: &Device, graphics_queue: &QueueData) {
        let cmd = match self.cmd.take().unwrap() {
            BatchCmd::Recording(cmd) => cmd,
            idle @ BatchCmd::Idle(_) => {
                self.cmd = Some(idle);
                return;
            }
        };

        let cmd = cmd.end(gpu);

        self.fence.reset();

        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd.handle())
            .device_mask(0)];

        let submit_info = &[vk::SubmitInfo2::default().command_buffer_infos(cmd_info)];

        unsafe { gpu.queue_submit2(graphics_queue.queue, submit_info, self.fence.handle()) }
            .unwrap();

        self.cmd = Some(BatchCmd::Idle(cmd.into_submitted()));

        assert!(
            self.fence.wait(1_000_000_000),
            "upload batch fence timed out"
        );

        self.staging.clear();
        self.staging_bytes = 0;
    }
}

impl Drop for UploadBatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Stage + access masks for one side of an image barrier.
#[derive(Clone, Copy)]
pub(crate) struct BarrierScope {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
}

impl BarrierScope {
    pub(crate) const NONE: Self = Self {
        stage: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };
    pub(crate) const COMPUTE_STORAGE_RW: Self = Self {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
        ),
    };
    pub(crate) const COLOR_ATTACHMENT_RW: Self = Self {
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
        ),
    };
    pub(crate) const COLOR_ATTACHMENT_WRITE: Self = Self {
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };
    pub(crate) const DEPTH_ATTACHMENT_RW: Self = Self {
        stage: vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
        ),
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
        ),
    };
    pub(crate) const TRANSFER_READ: Self = Self {
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
    };
    pub(crate) const TRANSFER_WRITE: Self = Self {
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };
    pub(crate) const FRAGMENT_SAMPLED_READ: Self = Self {
        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };
    pub(crate) const INDIRECT_READ: Self = Self {
        stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
        access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
    };
}

pub(crate) fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src: BarrierScope,
    dst: BarrierScope,
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
        .src_stage_mask(src.stage)
        .src_access_mask(src.access)
        .dst_stage_mask(dst.stage)
        .dst_access_mask(dst.access)
        .old_layout(current_layout)
        .new_layout(new_layout)
        .subresource_range(subresource_range)
        .image(image);

    let barriers = [image_barrier];

    let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) }
}

pub(crate) fn memory_barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: BarrierScope,
    dst: BarrierScope,
) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(src.stage)
        .src_access_mask(src.access)
        .dst_stage_mask(dst.stage)
        .dst_access_mask(dst.access);
    let barriers = [barrier];
    let dependency_info = vk::DependencyInfo::default().memory_barriers(&barriers);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
}
