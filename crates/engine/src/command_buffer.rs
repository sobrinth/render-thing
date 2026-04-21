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
