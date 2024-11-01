mod debug;

use std::error::Error;
use std::ffi::{CStr, CString};
use ash::{vk, Device, Entry, Instance};
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::vk::SurfaceKHR;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};
use crate::debug::{check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS};

struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    _physical_device: vk::PhysicalDevice,
    device: Device,
    _graphics_queue: vk::Queue,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    _present_queue: vk::Queue,
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<VulkanContext>,
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

impl VulkanContext {
    fn new(window: &Window) -> Self {
        log::debug!("Creating Vulkan context.");

        let entry = unsafe { Entry::load().expect("Failed to create entry.") };
        let instance = Self::create_instance(&entry, window).unwrap();

        let surface = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            ).unwrap()
        };

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let physical_device = Self::pick_physical_device(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) = Self::create_logical_device_with_graphics_queue(&instance, &surface, surface_khr, physical_device);

        Self {
            _entry: entry,
            instance,
            debug_report_callback,
            _physical_device: physical_device,
            device,
            _graphics_queue: graphics_queue,
            surface,
            surface_khr,
            _present_queue: present_queue,
        }
    }

    fn run(&mut self) {
        log::info!("Running vulkan application");
    }

    fn create_instance(entry: &Entry, window: &Window) -> Result<Instance, Box<dyn Error>> {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let mut extension_names = ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
            .unwrap().to_vec();

        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();


        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { Ok(entry.create_instance(&instance_create_info, None)?) }
    }

    /// Pick the first physical device that supports the graphics queue families.
    fn pick_physical_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
    ) -> vk::PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        graphics.is_some() && present.is_some()
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };

        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support =
                unsafe { surface.get_physical_device_surface_support(device, index, surface_khr).unwrap() };

            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    /// Create the logical device to interact with `device`, a graphics queue
    /// and a presentation queue.
    ///
    /// # Returns
    ///
    /// Return a tuple containing the logical device, the graphics queue and the presentation queue.
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Device, vk::Queue, vk::Queue) {
        let (graphics_family_index, present_family_index) =
            Self::find_queue_families(instance, surface, surface_khr, device);
        let graphics_family_index = graphics_family_index.unwrap();
        let present_family_index = present_family_index.unwrap();

        let queue_priorities = [1.0_f32];

        let queue_create_infos = {
            // Vulkan spec does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to dedup it.
            let mut indices = vec![graphics_family_index, present_family_index];
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
                .collect::<Vec<_>>()
        };

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features);

        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (device, graphics_queue, present_queue)
    }
}
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan App with Ash")
                    .with_inner_size(PhysicalSize::new(800, 600)),
            )
            .unwrap();

        self.vulkan = Some(VulkanContext::new(&window));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        }
    }
}
