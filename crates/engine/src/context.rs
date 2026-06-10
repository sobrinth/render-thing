use crate::debug::{
    check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger,
};
use crate::swapchain;
use ash::Instance;
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{Device, Entry, vk};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::error::Error;
use std::ffi::CStr;
use std::sync::Arc;

// KHR promotions of the maintenance1 extensions (aliases of the EXT variants);
// ash 0.38 predates them, so the names are spelled out here.
const SURFACE_MAINTENANCE1_KHR: &CStr = c"VK_KHR_surface_maintenance1";
const SWAPCHAIN_MAINTENANCE1_KHR: &CStr = c"VK_KHR_swapchain_maintenance1";

pub(crate) struct VkContext {
    _vulkan_fn: Entry,
    pub(crate) instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub(crate) surface_fn: surface::Instance,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: Device,
}

impl VkContext {
    pub(crate) fn initialize(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        app_name: &str,
    ) -> (Self, QueueData, Arc<vk_mem::Allocator>) {
        log::trace!("Creating vulkan context");
        let vulkan_fn = unsafe { Entry::load().expect("Failed to create ash entrypoint") };
        let instance = Self::create_instance(&vulkan_fn, window, app_name).unwrap();

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

        let mut alloc_info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        alloc_info.vulkan_api_version = vk::make_api_version(0, 1, 3, 0);
        alloc_info.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe { vk_mem::Allocator::new(alloc_info) }.unwrap();

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
            Arc::new(allocator),
        )
    }

    fn create_instance(
        vulkan_fn: &Entry,
        window: &impl HasDisplayHandle,
        app_name: &str,
    ) -> Result<Instance, Box<dyn Error>> {
        let app_name = std::ffi::CString::new(app_name).unwrap_or_default();
        let engine_name = c"No Engine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
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

        // Instance-level dependencies of VK_(KHR|EXT)_swapchain_maintenance1.
        // Hard requirement: drivers are assumed end-of-2025 or newer (see spec).
        let available_extensions =
            unsafe { vulkan_fn.enumerate_instance_extension_properties(None) }.unwrap();
        let has_extension = |name: &CStr| {
            available_extensions.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                ext_name == name
            })
        };

        assert!(
            has_extension(ash::khr::get_surface_capabilities2::NAME),
            "VK_KHR_get_surface_capabilities2 is required (driver too old?)"
        );
        extension_names.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());

        // Enable every available spelling: on multi-ICD systems the instance-level
        // list is a union across drivers, and the selected device's variant must
        // find its dependency under the matching name.
        let surface_maintenance1 = [
            SURFACE_MAINTENANCE1_KHR,
            ash::ext::surface_maintenance1::NAME,
        ]
        .into_iter()
        .filter(|name| has_extension(name))
        .collect_vec();
        assert!(
            !surface_maintenance1.is_empty(),
            "VK_KHR/EXT_surface_maintenance1 is required (driver too old?)"
        );
        extension_names.extend(surface_maintenance1.iter().map(|name| name.as_ptr()));

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
        // Select a physical device
        let available_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let selected_device = available_devices
            .into_iter()
            .filter(|d| Self::is_device_suitable(instance, surface_fn, surface, *d))
            .max_by_key(|d| {
                let props = unsafe { instance.get_physical_device_properties(*d) };
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                }
            })
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
            // And since the family for graphics and presentation could be the same, we need to dedup it.
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

        let device_extensions = Self::select_device_extensions(instance, selected_device)
            .expect("selected device is missing required extensions");
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect_vec();

        let device_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let mut device_features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true);

        let mut device_features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut swapchain_maintenance1_features =
            vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                .swapchain_maintenance1(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features)
            .push_next(&mut device_features12)
            .push_next(&mut device_features13)
            .push_next(&mut swapchain_maintenance1_features);

        let device = unsafe { instance.create_device(selected_device, &device_create_info, None) }
            .expect("Failed to create logical device.");

        // graphics and present queue are created but not retrieved yet
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_index, 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_index, 0) };

        let graphics_queue_data = QueueData {
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

        let extension_support = Self::select_device_extensions(instance, physical_device).is_some();

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

    /// Device extensions to enable, or None if the device is missing any.
    /// swapchain_maintenance1 prefers the KHR name, falling back to the EXT
    /// alias (older Mesa advertises only EXT; NVIDIA advertises only KHR).
    fn select_device_extensions(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<Vec<&'static CStr>> {
        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap();
        let has_extension = |name: &CStr| {
            extension_properties.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                ext_name == name
            })
        };

        if !has_extension(ash::khr::swapchain::NAME) {
            return None;
        }
        let maintenance1 = [
            SWAPCHAIN_MAINTENANCE1_KHR,
            ash::ext::swapchain_maintenance1::NAME,
        ]
        .into_iter()
        .find(|name| has_extension(name))?;

        Some(vec![ash::khr::swapchain::NAME, maintenance1])
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
        log::trace!("Start: Dropping context");
        unsafe {
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
        log::trace!("End: Dropping context");
    }
}

#[derive(Copy, Clone)]
pub(crate) struct QueueData {
    pub(crate) queue: vk::Queue,
    pub(crate) family_index: u32,
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_index: u32,
    present_index: u32,
}
