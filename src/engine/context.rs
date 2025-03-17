use crate::QueueFamilyIndices;
use crate::debug::{
    check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger,
};
use crate::engine::renderer::QueueData;
use crate::engine::{renderer, swapchain};
use ash::Instance;
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{Device, Entry, vk};
use itertools::Itertools;
use std::error::Error;
use std::ffi::CStr;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct VkContext {
    _vulkan_fn: Entry,
    pub instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
}

impl VkContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface_fn(&self) -> &surface::Instance {
        &self.surface_fn
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };

        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    pub fn initialize(window: &Window) -> (Self, QueueData) {
        log::debug!("Creating vulkan context");
        // TODO: db: Probably move reference to `winit` out of VkContext
        let vulkan_fn = unsafe { Entry::load().expect("Failed to create ash entrypoint") };
        let instance = Self::create_instance(&vulkan_fn, window).unwrap();

        let debug_report_callback = setup_debug_messenger(&vulkan_fn, &instance);

        let surface_fn = surface::Instance::new(&vulkan_fn, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &vulkan_fn,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .unwrap();

        let (physical_device, device, graphics_queue, _) =
            Self::initialize_vulkan_device(&instance, &surface_fn, surface);

        (
            Self {
                _vulkan_fn: vulkan_fn,
                instance,
                debug_report_callback,
                surface_fn,
                surface,
                physical_device,
                device,
            },
            graphics_queue,
        )
    }

    fn create_instance(vulkan_fn: &Entry, window: &Window) -> Result<Instance, Box<dyn Error>> {
        let app_name = c"Vulkan Application";
        let engine_name = c"No Engine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();

        if cfg!(debug_assertions) {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        Self::get_required_instance_extensions()
            .iter()
            .for_each(|ext| {
                extension_names.push(ext.as_ptr());
            });

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if cfg!(debug_assertions) {
            check_validation_layer_support(vulkan_fn);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { Ok(vulkan_fn.create_instance(&instance_create_info, None)?) }
    }

    fn initialize_vulkan_device(
        instance: &Instance,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, Device, QueueData, vk::Queue) {
        // Select physical device
        let available_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let selected_device = available_devices
            .into_iter()
            .find(|d| Self::is_device_suitable(instance, surface_fn, surface, *d))
            .expect("No suitable physical device found.");

        let props = unsafe { instance.get_physical_device_properties(selected_device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        // Queue families for graphics and present queue
        let (graphics, present) =
            Self::find_queue_families(instance, surface_fn, surface, selected_device);
        let queue_family_indices = QueueFamilyIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        // Create logical vulkan device
        let queue_priorities = [1.0_f32];
        let queue_create_infos = {
            // Vulkan spec does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to dedup it.
            let mut indices = vec![
                queue_family_indices.graphics_index,
                queue_family_indices.present_index,
            ];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                })
                .collect_vec()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect_vec();

        let device_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let mut device_features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features)
            .push_next(&mut device_features13);

        let device = unsafe { instance.create_device(selected_device, &device_create_info, None) }
            .expect("Failed to create logical device.");

        // graphics and present queue are created, but not retrieved yet
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_index, 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_index, 0) };

        let graphics_queue_data = renderer::QueueData {
            queue: graphics_queue,
            family_index: queue_family_indices.graphics_index,
        };

        (selected_device, device, graphics_queue_data, present_queue)
    }
    fn is_device_suitable(
        instance: &Instance,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) =
            Self::find_queue_families(instance, surface_fn, surface, physical_device);

        let extension_support = Self::check_device_extension_support(instance, physical_device);

        let is_swapchain_usable = {
            let details =
                swapchain::SwapchainSupportDetails::new(physical_device, surface_fn, surface);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(physical_device) };
        graphics.is_some()
            && present.is_some()
            && extension_support
            && is_swapchain_usable
            && features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let required_extension = Self::get_required_device_extensions();

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap();

        for extension in required_extension.iter() {
            let found_ext = extension_properties.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                extension == &ext_name
            });

            if !found_ext {
                return false;
            }
        }

        true
    }

    // TODO: db: What is actually needed here, now that I derped on the vk version...
    fn get_required_device_extensions() -> [&'static CStr; 11] {
        [
            c"VK_KHR_swapchain",
            c"VK_KHR_dynamic_rendering",
            c"VK_KHR_synchronization2",
            c"VK_KHR_create_renderpass2",
            c"VK_KHR_depth_stencil_resolve",
            c"VK_KHR_buffer_device_address",
            c"VK_EXT_descriptor_indexing",
            c"VK_KHR_multiview",
            c"VK_KHR_maintenance2",
            c"VK_KHR_maintenance3",
            c"VK_KHR_device_group",
        ]
    }

    // TODO: db: What is actually needed here, now that I derped on the vk version...
    fn get_required_instance_extensions() -> [&'static CStr; 2] {
        [
            c"VK_KHR_get_physical_device_properties2",
            c"VK_KHR_device_group_creation",
        ]
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface_fn.get_physical_device_surface_support(physical_device, index, surface)
            }
            .unwrap();

            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        log::debug!("Start: Dropping context");
        unsafe {
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
        log::debug!("End: Dropping context");
    }
}
