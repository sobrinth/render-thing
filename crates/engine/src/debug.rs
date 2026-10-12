use ash::ext::debug_utils;
use ash::{Entry, Instance, vk};
use eros::{Context, context, ensure};
use itertools::Itertools;
use std::ffi::{CStr, c_char, c_void};

pub(crate) const REQUIRED_LAYERS: [&CStr; 1] = [c"VK_LAYER_KHRONOS_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    match flag {
        Flag::VERBOSE => log::trace!("{:?} - {:?}", typ, message),
        Flag::INFO => log::debug!("{:?} - {:?}", typ, message),
        Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
        _ => log::error!("{:?} - {:?}", typ, message),
    }
    vk::FALSE
}

/// Pointers to the validation layer names; they point into `'static` C strings.
pub(crate) fn layer_name_pointers() -> Vec<*const c_char> {
    REQUIRED_LAYERS.iter().map(|n| n.as_ptr()).collect_vec()
}

/// Check that every layer in `REQUIRED_LAYERS` is supported by the Vulkan instance.
#[context("Checking validation layer support")]
pub(crate) fn check_validation_layer_support(entry: &Entry) -> eros::Result<()> {
    let supported_layers = unsafe { entry.enumerate_instance_layer_properties() }?;
    for required in REQUIRED_LAYERS {
        let found = supported_layers.iter().any(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
            name == required
        });
        ensure!(found, "Validation layer not supported: {:?}", required);
    }
    Ok(())
}

/// Set up the debug messenger if validation layers are enabled
pub(crate) fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> eros::Result<Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
    if !cfg!(debug_assertions) {
        return Ok(None);
    }

    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_utils_messenger =
        unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }
            .context("Creating debug utils messenger")?;

    Ok(Some((debug_utils, debug_utils_messenger)))
}
