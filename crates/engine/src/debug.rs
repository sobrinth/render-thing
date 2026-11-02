use ash::ext::debug_utils;
use ash::{Entry, Instance, vk};
use eros::Context;
use std::ffi::{CStr, CString, c_void};

pub(crate) const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation {
    /// No validation layer.
    Off,
    /// Core checks, synchronization validation and best practices.
    Full,
    /// GPU-assisted validation only: instruments shaders to catch out-of-bounds
    /// bindless indices and buffer-device-address / indirect buffer accesses. Core,
    /// sync and best practices are off (the layer warns combining them is very slow),
    /// so fix `Full` errors first, then switch to this.
    GpuAssisted,
}

impl Validation {
    fn settings(self) -> &'static [(&'static CStr, bool)] {
        match self {
            Validation::Off => &[],
            Validation::Full => &[
                (c"validate_core", true),
                (c"validate_sync", true),
                (c"validate_best_practices", true),
                (c"gpuav_enable", false),
            ],
            Validation::GpuAssisted => &[
                (c"validate_core", false),
                (c"validate_sync", false),
                (c"validate_best_practices", false),
                (c"gpuav_enable", true),
            ],
        }
    }
}

static BOOL_TRUE: vk::Bool32 = vk::TRUE;
static BOOL_FALSE: vk::Bool32 = vk::FALSE;

pub(crate) fn layer_settings(mode: Validation) -> Vec<vk::LayerSettingEXT<'static>> {
    mode.settings()
        .iter()
        .map(|&(name, enabled)| {
            let mut setting = vk::LayerSettingEXT::default()
                .layer_name(VALIDATION_LAYER)
                .setting_name(name)
                .ty(vk::LayerSettingTypeEXT::BOOL32);
            // Set by hand: ash 0.38's `values()` takes bytes and stores the byte count
            // as value_count, which is 4x too many for a BOOL32 (fixed in ash master).
            // CHECK (ash 0.39): Fix landed?
            setting.value_count = 1;
            setting.p_values = if enabled { &BOOL_TRUE } else { &BOOL_FALSE } as *const vk::Bool32
                as *const c_void;
            setting
        })
        .collect()
}

pub(crate) fn has_layer(entry: &Entry, name: &CStr) -> eros::Result<bool> {
    let layers = unsafe { entry.enumerate_instance_layer_properties() }?;
    Ok(layers
        .iter()
        .any(|layer| layer.layer_name_as_c_str() == Ok(name)))
}

pub(crate) fn layer_has_extension(
    entry: &Entry,
    layer: &CStr,
    extension: &CStr,
) -> eros::Result<bool> {
    let extensions = unsafe { entry.enumerate_instance_extension_properties(Some(layer)) }?;
    Ok(extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(extension)))
}

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    // SAFETY: the loader passes a valid callback-data struct for the duration of the
    // call; both pointers are null-checked anyway.
    let message = unsafe { data.as_ref() }
        .and_then(|data| unsafe { data.message_as_c_str() })
        .map(CStr::to_string_lossy)
        .unwrap_or_default();

    match severity {
        Severity::ERROR => log::error!("[{kind:?}] {message}"),
        Severity::WARNING => log::warn!("[{kind:?}] {message}"),
        Severity::INFO => log::debug!("[{kind:?}] {message}"),
        _ => log::trace!("[{kind:?}] {message}"),
    }
    vk::FALSE
}

pub(crate) fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> eros::Result<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
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

    Ok((debug_utils, debug_utils_messenger))
}

/// Object names and command buffer labels for validation messages and GPU tools
/// (RenderDoc, Nsight).
#[derive(Clone)]
pub(crate) struct DebugUtils(Option<debug_utils::Device>);

impl DebugUtils {
    pub(crate) fn new(device: Option<debug_utils::Device>) -> Self {
        Self(device)
    }

    pub(crate) fn set_object_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let Some(debug_utils) = &self.0 else {
            return;
        };
        let Ok(name) = CString::new(name) else {
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);
        if let Err(err) = unsafe { debug_utils.set_debug_utils_object_name(&info) } {
            log::warn!("Naming {name:?} failed: {err}");
        }
    }

    /// NOTE: Labels can nest
    pub(crate) fn begin_label(&self, cmd: vk::CommandBuffer, name: &CStr) {
        if let Some(debug_utils) = &self.0 {
            let label = vk::DebugUtilsLabelEXT::default().label_name(name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(cmd, &label) };
        }
    }

    pub(crate) fn end_label(&self, cmd: vk::CommandBuffer) {
        if let Some(debug_utils) = &self.0 {
            unsafe { debug_utils.cmd_end_debug_utils_label(cmd) };
        }
    }
}
