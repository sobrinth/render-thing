use ash::vk;
use ash::khr::surface;

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(device: vk::PhysicalDevice, surface: &surface::Instance, surface_khr: vk::SurfaceKHR) -> Self {
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .unwrap()
        };

        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(device, surface_khr)
                .unwrap()
        };

        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .unwrap()
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }
}