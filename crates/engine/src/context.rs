use crate::debug::{self, DebugUtils, Validation, setup_debug_messenger};
use crate::features::FeatureChain;
use crate::swapchain;
use ash::Instance;
use ash::ext::{debug_utils, layer_settings};
use ash::khr::surface;
use ash::{Device, Entry, vk};
use eros::{Context, ensure};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::{CStr, c_char};
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
    pub(crate) debug: DebugUtils,
}

impl VkContext {
    pub(crate) fn initialize(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        app_name: &str,
        validation: Validation,
    ) -> eros::Result<(Self, QueueData, Arc<vk_mem::Allocator>)> {
        log::trace!("Creating vulkan context");
        let vulkan_fn = unsafe { Entry::load() }.context("Loading the Vulkan library")?;
        let (instance, validation, debug_utils_available) =
            Self::create_instance(&vulkan_fn, window, app_name, validation)
                .context("Creating Vulkan instance")?;

        let debug_report_callback = if validation != Validation::Off && debug_utils_available {
            Some(setup_debug_messenger(&vulkan_fn, &instance)?)
        } else {
            None
        };

        let surface_fn = surface::Instance::new(&vulkan_fn, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &vulkan_fn,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }
        .context("Creating window surface")?;

        let (physical_device, device, graphics_queue) =
            Self::initialize_vulkan_device(&instance, &surface_fn, surface)
                .context("Creating logical device")?;

        let debug = DebugUtils::new(
            debug_utils_available.then(|| debug_utils::Device::new(&instance, &device)),
        );

        let mut alloc_info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        alloc_info.vulkan_api_version = vk::make_api_version(0, 1, 3, 0);
        alloc_info.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator =
            unsafe { vk_mem::Allocator::new(alloc_info) }.context("Creating VMA allocator")?;

        Ok((
            Self {
                _vulkan_fn: vulkan_fn,
                instance,
                debug_report_callback,
                surface_fn,
                surface,
                physical_device,
                device,
                debug,
            },
            graphics_queue,
            Arc::new(allocator),
        ))
    }

    fn create_instance(
        vulkan_fn: &Entry,
        window: &impl HasDisplayHandle,
        app_name: &str,
        validation: Validation,
    ) -> eros::Result<(Instance, Validation, bool)> {
        let app_name = std::ffi::CString::new(app_name).unwrap_or_default();
        let engine_name = c"No Engine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut extension_names: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();

        // Instance-level dependencies of VK_(KHR|EXT)_swapchain_maintenance1.
        // Hard requirement: drivers are assumed end-of-2025 or newer (see spec).
        let available_extensions =
            unsafe { vulkan_fn.enumerate_instance_extension_properties(None) }?;
        let has_extension = |name: &CStr| {
            available_extensions.iter().any(|ext| {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                ext_name == name
            })
        };

        ensure!(
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
        ensure!(
            !surface_maintenance1.is_empty(),
            "VK_KHR/EXT_surface_maintenance1 is required (driver too old?)"
        );
        extension_names.extend(surface_maintenance1.iter().map(|name| name.as_ptr()));

        // Always useful (also release). Names show in RenderDoc / Nsight
        let debug_utils_available = has_extension(debug_utils::NAME);
        if debug_utils_available {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        let validation = if validation != Validation::Off
            && !debug::has_layer(vulkan_fn, debug::VALIDATION_LAYER)?
        {
            log::warn!("Validation requested, but VK_LAYER_KHRONOS_validation is not installed");
            Validation::Off
        } else {
            validation
        };
        let layer_names_ptrs: Vec<*const c_char> = if validation != Validation::Off {
            vec![debug::VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let layer_settings = if validation != Validation::Off
            && debug::layer_has_extension(vulkan_fn, debug::VALIDATION_LAYER, layer_settings::NAME)?
        {
            extension_names.push(layer_settings::NAME.as_ptr());
            debug::layer_settings(validation)
        } else {
            if validation != Validation::Off {
                log::warn!(
                    "VK_EXT_layer_settings unavailable; validation runs with layer defaults"
                );
            }
            Vec::new()
        };
        let mut layer_settings_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&layer_settings);

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names_ptrs);
        if !layer_settings.is_empty() {
            instance_create_info = instance_create_info.push_next(&mut layer_settings_info);
        }
        log::info!("Validation: {validation:?}");

        let instance = unsafe { vulkan_fn.create_instance(&instance_create_info, None) }?;
        Ok((instance, validation, debug_utils_available))
    }

    fn initialize_vulkan_device(
        instance: &Instance,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> eros::Result<(vk::PhysicalDevice, Device, QueueData)> {
        let candidate = pick_physical_device(instance, surface_fn, surface)?;

        let props = unsafe { instance.get_physical_device_properties(candidate.physical_device) };
        log::debug!(
            "Selected physical device: {:?}",
            props.device_name_as_c_str().unwrap_or(c"<unknown>")
        );

        let queue_priorities = [1.0_f32];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(candidate.queue_family)
            .queue_priorities(&queue_priorities)];

        let device_extensions_ptrs = candidate
            .extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect_vec();

        let mut features = FeatureChain::required();
        let mut features2 = features.features2();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .push_next(&mut features2);

        let device = unsafe {
            instance.create_device(candidate.physical_device, &device_create_info, None)
        }?;

        let graphics_queue = QueueData {
            queue: unsafe { device.get_device_queue(candidate.queue_family, 0) },
            family_index: candidate.queue_family,
        };

        Ok((candidate.physical_device, device, graphics_queue))
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

struct Candidate {
    physical_device: vk::PhysicalDevice,
    /// It is assumed the queue supports both graphics and present
    queue_family: u32,
    extensions: Vec<&'static CStr>,
    score: u32,
}

fn pick_physical_device(
    instance: &Instance,
    surface_fn: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> eros::Result<Candidate> {
    let mut rejections = Vec::new();
    let mut best: Option<Candidate> = None;

    for physical_device in unsafe { instance.enumerate_physical_devices() }? {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = properties
            .device_name_as_c_str()
            .unwrap_or(c"<unknown>")
            .to_string_lossy()
            .into_owned();

        match evaluate(instance, surface_fn, surface, physical_device, &properties)? {
            Ok(candidate) => {
                log::debug!("Suitable device: {name}");
                if best.as_ref().is_none_or(|b| candidate.score > b.score) {
                    best = Some(candidate);
                }
            }
            Err(reason) => {
                log::debug!("Rejected device {name}: {reason}");
                rejections.push(format!("{name}: {reason}"));
            }
        }
    }

    best.ok_or_else(|| eros::error!("No suitable GPU found:\n  {}", rejections.join("\n  ")))
}

fn evaluate(
    instance: &Instance,
    surface_fn: &surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
) -> eros::Result<Result<Candidate, String>> {
    // One family that does graphics and present: presenting happens on the graphics
    // queue and swapchain images are EXCLUSIVE, so split families aren't supported.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut queue_family = None;
    for (index, family) in families.iter().enumerate() {
        let index = index as u32;
        if family.queue_count == 0 || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            continue;
        }
        let presents = unsafe {
            surface_fn.get_physical_device_surface_support(physical_device, index, surface)
        }?;
        if presents {
            queue_family = Some(index);
            break;
        }
    }
    let Some(queue_family) = queue_family else {
        return Ok(Err(
            "no queue family supports both graphics and present".into()
        ));
    };

    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
    let has_extension = |name: &CStr| {
        available
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(name))
    };
    if !has_extension(ash::khr::swapchain::NAME) {
        return Ok(Err("VK_KHR_swapchain missing".into()));
    }
    // Prefer the KHR name, falling back to the EXT alias
    let Some(swapchain_maintenance1) = [
        SWAPCHAIN_MAINTENANCE1_KHR,
        ash::ext::swapchain_maintenance1::NAME,
    ]
    .into_iter()
    .find(|name| has_extension(name)) else {
        return Ok(Err("VK_KHR/EXT_swapchain_maintenance1 missing".into()));
    };

    let details = swapchain::SwapchainSupportDetails::new(physical_device, surface_fn, surface)?;
    if details.formats.is_empty() || details.present_modes.is_empty() {
        return Ok(Err("surface reports no formats or present modes".into()));
    }

    let missing = FeatureChain::supported(instance, physical_device).missing_required();
    if !missing.is_empty() {
        return Ok(Err(format!("missing features: {}", missing.join(", "))));
    }

    let score = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    };

    Ok(Ok(Candidate {
        physical_device,
        queue_family,
        extensions: vec![ash::khr::swapchain::NAME, swapchain_maintenance1],
        score,
    }))
}
