use ash::{Device, vk};

pub(crate) struct Fence {
    handle: vk::Fence,
    device: Device,
}

impl Fence {
    pub(crate) fn new(device: &Device) -> Self {
        Self::create(device, &vk::FenceCreateInfo::default())
    }

    pub(crate) fn new_signaled(device: &Device) -> Self {
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        Self::create(device, &info)
    }

    fn create(device: &Device, info: &vk::FenceCreateInfo) -> Self {
        let handle = unsafe { device.create_fence(info, None) }.unwrap();
        Self {
            handle,
            device: device.clone(),
        }
    }

    pub(crate) fn handle(&self) -> vk::Fence {
        self.handle
    }

    #[must_use]
    pub(crate) fn wait(&self, timeout_ns: u64) -> bool {
        match unsafe {
            self.device
                .wait_for_fences(&[self.handle], true, timeout_ns)
        } {
            Ok(()) => true,
            Err(vk::Result::TIMEOUT) => false,
            Err(e) => panic!("wait_for_fences failed: {e}"),
        }
    }

    pub(crate) fn reset(&self) {
        unsafe { self.device.reset_fences(&[self.handle]) }.unwrap();
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.handle, None) }
    }
}

pub(crate) struct Semaphore {
    handle: vk::Semaphore,
    device: Device,
}

impl Semaphore {
    pub(crate) fn new(device: &Device) -> Self {
        let info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { device.create_semaphore(&info, None) }.unwrap();
        Self {
            handle,
            device: device.clone(),
        }
    }

    pub(crate) fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.handle, None) }
    }
}
