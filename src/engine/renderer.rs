use crate::engine::context::VkContext;
use crate::engine::swapchain::Swapchain;
use winit::window::Window;

pub struct VulkanRenderer {
    pub context: VkContext,

    swapchain: Swapchain,
}

impl VulkanRenderer {
    pub fn initialize(window: &Window) -> Self {
        let context = VkContext::initialize(window);

        let swapchain = Swapchain::create(
            &context,
            [window.inner_size().width, window.inner_size().height],
        );
        Self { context, swapchain }
    }

    pub fn wait_gpu_idle(&self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::debug!("Start: Dropping renderer");
        unsafe { self.swapchain.destroy(&self.context.device) }
        log::debug!("End: Dropping renderer");
    }
}
