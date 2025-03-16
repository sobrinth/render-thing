use crate::engine::context::VkContext;
use winit::window::Window;

pub struct VulkanRenderer {
    pub context: VkContext,
}

impl VulkanRenderer {
    pub fn initialize(window: &Window) -> Self {
        log::debug!("Creating vulkan context");

        let context = VkContext::initialize(window);
        Self {
            context,
        }
    }
}
