use crate::engine::context::VkContext;
use ash::khr::surface;
use ash::{Device, vk};
use itertools::Itertools;

pub struct Swapchain {
    properties: SwapchainProperties,
    pub swapchain_fn: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn create(vk_context: &VkContext, dimensions: [u32; 2]) -> Self {
        let details = SwapchainSupportDetails::new(
            vk_context.physical_device,
            &vk_context.surface_fn,
            vk_context.surface,
        );

        let swapchain_properties = details.get_ideal_swapchain_properties(dimensions);

        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {}",
            swapchain_properties.format.format,
            swapchain_properties.format.color_space,
            swapchain_properties.present_mode,
            swapchain_properties.extent,
            image_count,
        );

        let create_info = {
            let default = vk::SwapchainCreateInfoKHR::default()
                .surface(vk_context.surface)
                .min_image_count(image_count)
                .image_format(swapchain_properties.format.format)
                .image_color_space(swapchain_properties.format.color_space)
                .image_extent(swapchain_properties.extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            // TODO: db: Handle image sharing mode if graphics / present queue are different

            default
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(swapchain_properties.present_mode)
                .clipped(true)
        };

        let swapchain_fn =
            ash::khr::swapchain::Device::new(&vk_context.instance, &vk_context.device);
        let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None).unwrap() };
        let swapchain_images = unsafe { swapchain_fn.get_swapchain_images(swapchain).unwrap() };

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_properties.format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe { vk_context.device.create_image_view(&create_info, None) }.unwrap()
            })
            .collect_vec();

        Self {
            properties: swapchain_properties,
            swapchain_fn,
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views,
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        log::debug!("Start: Destroying swapchain");
        unsafe {
            self.image_views
                .iter()
                .for_each(|image_view| device.destroy_image_view(*image_view, None));
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
        log::debug!("End: Destroying swapchain");
    }
}

#[derive(Copy, Clone, Debug)]
struct SwapchainProperties {
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
}
pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: vk::PhysicalDevice,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Self {
        let capabilities = unsafe {
            surface_fn
                .get_physical_device_surface_capabilities(device, surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_fn
                .get_physical_device_surface_formats(device, surface)
                .unwrap()
        };

        let present_modes = unsafe {
            surface_fn
                .get_physical_device_surface_present_modes(device, surface)
                .unwrap()
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = Self::choose_swapchain_surface_format(&self.formats);
        let present_mode = Self::choose_swapchain_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    /// Choose the swapchain surface format.
    ///
    /// Will choose B8G8R8A8_UNORM/SRGB_NONLINEAR if possible or
    /// the first available otherwise.
    fn choose_swapchain_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
            return vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        *available_formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    /// Choose the swapchain present mode.
    ///
    /// Will favor MAILBOX (aka. Triple buffering) otherwise FIFO.
    /// If none is present it will fall back to IMMEDIATE.
    fn choose_swapchain_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if available_present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        }
    }

    /// Choose the swapchain extent.
    ///
    /// If a current extent is defined it will be returned.
    /// Otherwise, the surface extent clamped between the min
    /// and max image extent will be returned.
    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        preferred_dimensions: [u32; 2],
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;
        let width = preferred_dimensions[0].min(max.width).max(min.width);
        let height = preferred_dimensions[1].min(max.height).max(min.height);
        vk::Extent2D { width, height }
    }
}
