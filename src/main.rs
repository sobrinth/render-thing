mod debug;
mod swapchain;

use crate::debug::*;
use crate::swapchain::*;

use ash::ext::debug_utils;
use ash::khr::{surface, swapchain as khr_swapchain};
use ash::{vk, Device, Entry, Instance};
use std::error::Error;
use std::ffi::{CStr, CString};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    _physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    present_queue: vk::Queue,
    swapchain: khr_swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    _swapchain_properties: SwapchainProperties,
    _images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
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
            )
            .unwrap()
        };

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let physical_device = Self::pick_physical_device(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                &surface,
                surface_khr,
                physical_device,
            );

        let (swapchain, swapchain_khr, swapchain_properties, images) =
            Self::create_swapchain_and_images(
                &instance,
                physical_device,
                &device,
                &surface,
                surface_khr,
            );
        let swapchain_image_views =
            Self::create_swapchain_image_views(&device, &images, swapchain_properties);

        let render_pass = Self::create_render_pass(&device, swapchain_properties);

        let (pipeline, layout) = Self::create_pipeline(&device, swapchain_properties, render_pass);

        let framebuffers = Self::create_framebuffers(
            &device,
            &swapchain_image_views,
            render_pass,
            swapchain_properties,
        );

        let command_pool =
            Self::create_command_pool(&device, &instance, &surface, surface_khr, physical_device);

        let command_buffers = Self::create_and_register_command_buffers(
            &device,
            command_pool,
            &framebuffers,
            render_pass,
            swapchain_properties,
            pipeline,
        );

        let (image_available_semaphore, render_finished_semaphore) =
            Self::create_semaphores(&device);

        Self {
            _entry: entry,
            instance,
            debug_report_callback,
            _physical_device: physical_device,
            device,
            graphics_queue,
            surface,
            surface_khr,
            present_queue,
            swapchain,
            swapchain_khr,
            _swapchain_properties: swapchain_properties,
            _images: images,
            swapchain_image_views,
            pipeline_layout: layout,
            render_pass,
            pipeline,
            swapchain_framebuffers: framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
        }
    }

    pub fn draw_frame(&mut self) {
        log::trace!("Drawing frame.");

        let image_index = unsafe {
            self.swapchain
                .acquire_next_image(
                    self.swapchain_khr,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        };

        let wait_semaphores = [self.image_available_semaphore];
        let signal_semaphores = [self.render_finished_semaphore];

        // submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            let submit_infos = [submit_info];

            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain_khr];
        let image_indices = [image_index];

        // Present queue
        {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe {
                self.swapchain
                    .queue_present(self.present_queue, &present_info)
                    .unwrap();
            }
        }
    }

    pub fn wait_gpu_idle(&self) {
        unsafe { self.device.device_wait_idle().unwrap() }
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

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();

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

    /// Pick the first suitable physical device.
    ///
    /// # Requirements
    /// - At least one queue family with one queue supporting graphics.
    /// - At least one queue family with one queue supporting presentation to `surface_khr`.
    /// - Swapchain extension support. (VK_KHR_swapchain)
    fn pick_physical_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
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
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);

        let extension_support = Self::check_device_extension_support(instance, device);

        let is_swapchain_usable = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        graphics.is_some() && present.is_some() && extension_support && is_swapchain_usable
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extension = Self::get_required_device_extensions();

        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

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

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [khr_swapchain::NAME]
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
        surface_khr: vk::SurfaceKHR,
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

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };

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
        surface_khr: vk::SurfaceKHR,
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

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
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

    /// Create the swapchain with optimal settings possible with `device`
    fn create_swapchain_and_images(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (
        khr_swapchain::Device,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let details = SwapchainSupportDetails::new(physical_device, surface, surface_khr);
        let swapchain_properties = details.get_ideal_swapchain_properties([WIDTH, HEIGHT]);

        let format = swapchain_properties.format;
        let present_mode = swapchain_properties.present_mode;
        let extent = swapchain_properties.extent;

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
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let (graphics, present) =
            Self::find_queue_families(instance, surface, surface_khr, physical_device);
        let families_indices = [graphics.unwrap(), present.unwrap()];

        let create_info = {
            let mut default = vk::SwapchainCreateInfoKHR::default()
                .surface(surface_khr)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            default = match (graphics, present) {
                (Some(graphics), Some(present)) if graphics != present => default
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices),
                (Some(_), Some(_)) => default.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
                _ => panic!("Failed to create swapchain."),
            };

            default
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let swapchain = khr_swapchain::Device::new(instance, device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, swapchain_properties, images)
    }

    /// Create one image view for each image of the swapchain.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_format: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_format.format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        // Vertex & Fragment Shaders
        let vertex_source = Self::read_shader_from_file("assets/shaders/shader.vert.spv");
        let fragment_source = Self::read_shader_from_file("assets/shaders/shader.frag.spv");

        let vertex_shader_module = Self::create_shader_module(device, &vertex_source);
        let fragment_shader_module = Self::create_shader_module(device, &fragment_source);

        // Vertex input & topology
        let entry_point_name = CString::new("main").unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name);
        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name);
        let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        // .vertex_binding_descriptions() Default for now, because the vertices are hardcoded in shader
        // .vertex_attribute_descriptions() Same

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport & Scissors
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor];

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // Rasterizer
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        // Multisampling
        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            // .sample_mask() // null
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        // Color blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend_attachments = [color_blend_attachment];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // Pipeline layout
        let layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::default();
            // .set_layouts() //there are no uniforms yet
            // .push_constant_ranges() // no push constants yet

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            // .depth_stencil_state()
            .color_blend_state(&color_blending_info)
            // .dynamic_state()
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }
        (pipeline, layout)
    }

    fn read_shader_from_file<P: AsRef<std::path::Path>>(path: P) -> Vec<u32> {
        log::debug!("Loading shader file {}", path.as_ref().display());
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module(device: &Device, shader_source: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(shader_source);
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
    ) -> vk::RenderPass {
        let attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let attachment_descs = [attachment_desc];

        let attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let attachment_refs = [attachment_ref];

        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_refs);
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps);

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        // Create a framebuffer for each image view
        image_views
            .iter()
            .map(|v| [*v])
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect()
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _);

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers
            .iter()
            .zip(framebuffers.iter())
            .for_each(|(buffer, framebuffer)| {
                let buffer = *buffer;

                // begin command buffer
                {
                    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                    // .inheritance_info() // null since it's a primary command buffer

                    unsafe {
                        device
                            .begin_command_buffer(buffer, &command_buffer_begin_info)
                            .unwrap()
                    }
                }

                // begin render pass
                {
                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: swapchain_properties.extent,
                        })
                        .clear_values(&clear_values);

                    unsafe {
                        device.cmd_begin_render_pass(
                            buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );
                    };
                }

                // bind pipeline
                unsafe {
                    device.cmd_bind_pipeline(
                        buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphics_pipeline,
                    )
                };

                // draw
                unsafe { device.cmd_draw(buffer, 3, 1, 0, 0) };

                // end render pass
                unsafe { device.cmd_end_render_pass(buffer) };

                // end command buffer
                unsafe { device.end_command_buffer(buffer).unwrap() }
            });

        buffers
    }

    fn create_command_pool(
        device: &Device,
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> vk::CommandPool {
        let (graphics_family, _) =
            Self::find_queue_families(instance, surface, surface_khr, physical_device);

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::empty());

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_semaphores(device: &Device) -> (vk::Semaphore, vk::Semaphore) {
        let image_available = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };
        let render_finished = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };
        (image_available, render_finished)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| self.device.destroy_framebuffer(*f, None));
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views
                .iter()
                .for_each(|v| self.device.destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
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
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.vulkan = Some(VulkanContext::new(&window));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let app = self.vulkan.as_mut().unwrap();

        app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.vulkan.as_ref().unwrap().wait_gpu_idle();
    }
}
