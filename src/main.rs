mod camera;
mod context;
mod debug;
mod math;
mod primitives;
mod swapchain;
mod texture;

use crate::{camera::*, context::*, debug::*, primitives::*, swapchain::*, texture::*};

use ash::ext::debug_utils;
use ash::khr::{surface, swapchain as khr_swapchain};
use ash::{Device, Entry, Instance, vk};
use cgmath::{Deg, Matrix4, Point3, Vector3, vec3};
use std::error::Error;
use std::ffi::CStr;
use std::path::Path;
use itertools::Itertools;
use tobj::GPU_LOAD_OPTIONS;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

struct VulkanApplication {
    resize_dimensions: Option<[u32; 2]>,
    dirty_swapchain: bool,

    vk_context: VkContext,

    camera: Camera,
    is_left_clicked: bool,
    cursor_position: [i32; 2],
    cursor_delta: Option<[i32; 2]>,
    wheel_delta: Option<f32>,

    queue_family_indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain: khr_swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,

    texture: Texture,
    model_index_count: usize,
    depth_texture: Texture,
    depth_format: vk::Format,
    color_texture: Texture,
    msaa_samples: vk::SampleCountFlags,

    mesh_buffers: MeshBuffers,
    uniform_buffers: Vec<AllocatedBuffer>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<VulkanApplication>,
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

impl VulkanApplication {
    fn new(window: &Window) -> Self {
        log::debug!("Creating vulkan context");

        let entry = unsafe { Entry::load().expect("Failed to create ash entrypoint") };
        let instance = Self::create_instance(&entry, window).unwrap();

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .unwrap();

        let (physical_device, device, _, _) =
            Self::initialize_vulkan_device(&instance, &surface, surface_khr);

        let vk_context = VkContext::new(
            entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
        );

        todo!()
    }
    fn new_12(_window: &Window) -> Self {
        /*
        // This is here so I can kinda check off what still needs to be moved over
        let (swapchain, swapchain_khr, swapchain_properties, swapchain_images) =
            Self::create_swapchain_and_images(&vk_context, queue_family_indices, [WIDTH, HEIGHT]);
        let swapchain_image_views = Self::create_swapchain_image_views(
            vk_context.device(),
            &swapchain_images,
            swapchain_properties,
        );

        let msaa_samples = vk_context.get_max_usable_sample_count();
        let depth_format = Self::find_depth_format(&vk_context);

        let render_pass = Self::create_render_pass(
            vk_context.device(),
            swapchain_properties,
            msaa_samples,
            depth_format,
        );

        let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device());

        let (pipeline, layout) = Self::create_pipeline(
            vk_context.device(),
            swapchain_properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
        );

        let command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_family_indices,
            vk::CommandPoolCreateFlags::empty(),
        );

        let transient_command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_family_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        let color_texture = Self::create_color_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            swapchain_properties,
            msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            depth_format,
            swapchain_properties.extent,
            msaa_samples,
        );

        let framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            swapchain_properties,
        );

        let texture = Self::create_texture_image(&vk_context, command_pool, graphics_queue);

        let (vertices, indices) = Self::load_model();

        let vertex_buffer = Self::create_vertex_buffer(
            &vk_context,
            transient_command_pool,
            graphics_queue,
            &vertices,
        );

        let index_buffer = Self::create_index_buffer(
            &vk_context,
            transient_command_pool,
            graphics_queue,
            &indices,
        );

        let mesh_buffers = MeshBuffers::new(vertex_buffer, index_buffer);

        let uniform_buffers = Self::create_uniform_buffers(&vk_context, swapchain_images.len());

        let descriptor_pool =
            Self::create_descriptor_pool(vk_context.device(), swapchain_images.len() as _);
        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            texture,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            vk_context.device(),
            command_pool,
            &framebuffers,
            render_pass,
            swapchain_properties,
            mesh_buffers,
            indices.len(),
            layout,
            &descriptor_sets,
            pipeline,
        );

        let in_flight_frames = Self::create_sync_objects(vk_context.device());

        Self {
            resize_dimensions: None,
            dirty_swapchain: false,
            vk_context,
            camera: Camera::default(),
            is_left_clicked: false,
            cursor_position: [0, 0],
            cursor_delta: None,
            wheel_delta: None,
            queue_family_indices,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_khr,
            swapchain_properties,
            swapchain_images,
            swapchain_image_views,
            pipeline_layout: layout,
            render_pass,
            descriptor_set_layout,
            pipeline,
            swapchain_framebuffers: framebuffers,
            command_pool,
            transient_command_pool,
            texture,
            color_texture,
            msaa_samples,
            model_index_count: indices.len(),
            depth_texture,
            depth_format,
            mesh_buffers,
            uniform_buffers,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            in_flight_frames,
        }
         */
        todo!()
    }

    pub(crate) fn draw_frame(&mut self) -> bool {
        log::trace!("Drawing frame.");

        let sync_objects = self.in_flight_frames.next().unwrap();

        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        // wait for available fence
        unsafe {
            self.vk_context
                .device()
                .wait_for_fences(&wait_fences, true, u64::MAX)
        }
        .unwrap();

        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.vk_context.device().reset_fences(&wait_fences) }.unwrap();

        self.update_uniform_buffers(image_index);

        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

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
                self.vk_context.device().queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    in_flight_fence,
                )
            }
            .unwrap();
        }

        let swapchains = [self.swapchain_khr];
        let image_indices = [image_index];

        // Present queue
        {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = unsafe {
                self.swapchain
                    .queue_present(self.present_queue, &present_info)
            };
            match result {
                Ok(true) => {
                    self.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        if self.resize_dimensions.is_some() {
            self.recreate_swapchain();
        }

        false
    }

    pub(crate) fn wait_gpu_idle(&self) {
        unsafe { self.vk_context.device().device_wait_idle() }.unwrap();
    }

    fn create_instance(entry: &Entry, window: &Window) -> Result<Instance, Box<dyn Error>> {
        let app_name = c"Vulkan Application";
        let engine_name = c"No Engine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
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
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extension_support
            && is_swapchain_usable
            && features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extension = Self::get_required_device_extensions();

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(device) }.unwrap();

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

    fn get_required_device_extensions() -> [&'static CStr; 7] {
        [
            c"VK_KHR_swapchain",
            c"VK_KHR_dynamic_rendering",
            c"VK_KHR_synchronization2",
            c"VK_KHR_create_renderpass2",
            c"VK_KHR_depth_stencil_resolve",
            c"VK_KHR_buffer_device_address",
            c"VK_EXT_descriptor_indexing",
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

            let present_support =
                unsafe { surface.get_physical_device_surface_support(device, index, surface_khr) }
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

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let ubo_binding = CameraUBO::get_descriptor_set_layout_bindings();
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [ubo_binding, sampler_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe { device.create_descriptor_set_layout(&layout_info, None) }.unwrap()
    }
    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);

        unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap()
    }

    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[AllocatedBuffer],
        texture: Texture,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect_vec();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap();

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, buffer)| {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(size_of::<CameraUBO>() as vk::DeviceSize);
                let buffer_infos = [buffer_info];

                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view)
                    .sampler(texture.sampler.unwrap());
                let image_infos = [image_info];

                let ubo_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos);

                let sampler_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos);

                let descriptor_writes = [ubo_descriptor_write, sampler_descriptor_write];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }
            });

        descriptor_sets
    }

    /// Create the swapchain with optimal settings possible with `device`
    fn create_swapchain_and_images(
        vk_context: &VkContext,
        queue_family_indices: QueueFamilyIndices,
        dimensions: [u32; 2],
    ) -> (
        khr_swapchain::Device,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let details = SwapchainSupportDetails::new(
            vk_context.physical_device(),
            vk_context.surface(),
            vk_context.surface_khr(),
        );
        let swapchain_properties = details.get_ideal_swapchain_properties(dimensions);

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

        let graphics = queue_family_indices.graphics_index;
        let present = queue_family_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut default = vk::SwapchainCreateInfoKHR::default()
                .surface(vk_context.surface_khr())
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            default = if graphics != present {
                default
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                default.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            default
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let swapchain = khr_swapchain::Device::new(vk_context.instance(), vk_context.device());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, swapchain_properties, images)
    }

    /// Create one image view for each image of the swapchain.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                Self::create_image_view(
                    device,
                    *image,
                    swapchain_properties.format.format,
                    1,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect_vec()
    }

    fn create_image_view(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        mip_levels: u32,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { device.create_image_view(&create_info, None) }.unwrap()
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        // Vertex & Fragment Shaders
        let vertex_source = Self::read_shader_from_file("assets/shaders/shader.vert.spv");
        let fragment_source = Self::read_shader_from_file("assets/shaders/shader.frag.spv");

        let vertex_shader_module = Self::create_shader_module(device, &vertex_source);
        let fragment_shader_module = Self::create_shader_module(device, &fragment_source);

        // Vertex input & topology
        let entry_point_name = c"main";
        let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(entry_point_name);
        let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(entry_point_name);
        let shader_stage_infos = [vertex_shader_stage_info, fragment_shader_stage_info];

        let vertex_binding_descs = [Vertex::get_binding_description()];
        let vertex_attribute_descs = Vertex::get_attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);

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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        // Multisampling
        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .min_sample_shading(1.0)
            // .sample_mask() // null
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        // Depth stencil
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default());

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
            let layouts = [descriptor_set_layout];
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<Matrix4<f32>>() as _,
            };
            let push_constant_ranges = [push_constant_range];
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_ranges);

            unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            // .dynamic_state()
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        }
        .unwrap()[0];

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }
        (pipeline, layout)
    }

    fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
        log::debug!("Loading shader file {}", path.as_ref().display());
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module(device: &Device, shader_source: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(shader_source);
        unsafe { device.create_shader_module(&create_info, None) }.unwrap()
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachment_desc = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let resolve_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_descs = [
            color_attachment_desc,
            depth_attachment_desc,
            resolve_attachment_desc,
        ];

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .resolve_attachments(&resolve_attachment_refs);
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

        unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap()
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        // Create a framebuffer for each image view
        image_views
            .iter()
            .map(|view| [color_texture.view, depth_texture.view, *view])
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None) }.unwrap()
            })
            .collect()
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        mesh_buffers: MeshBuffers,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _);

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap();

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                // .inheritance_info() // null since it's a primary command buffer

                unsafe { device.begin_command_buffer(buffer, &command_buffer_begin_info) }.unwrap()
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
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
                device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline)
            };

            // bind vertex buffer
            let vertex_buffers = [mesh_buffers.vertex.buffer];
            let offsets = [0];
            unsafe { device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets) };

            // bind index buffer
            unsafe {
                device.cmd_bind_index_buffer(
                    buffer,
                    mesh_buffers.index.buffer,
                    0,
                    vk::IndexType::UINT32,
                )
            };

            // bind descriptor set
            unsafe {
                let null = [];
                device.cmd_bind_descriptor_sets(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[i..=i],
                    &null,
                )
            };

            // Render objects
            let base_rot = Matrix4::from_angle_x(Deg(270.0));
            let transform_0 = Matrix4::from_translation(vec3(0.1, 0.0, -1.0)) * base_rot;
            let transform_1 = Matrix4::from_translation(vec3(-0.1, 0.0, 1.0)) * base_rot;

            Self::cmd_draw_object(device, buffer, index_count, pipeline_layout, transform_0);

            Self::cmd_draw_object(device, buffer, index_count, pipeline_layout, transform_1);

            // end render pass
            unsafe { device.cmd_end_render_pass(buffer) };

            // end command buffer
            unsafe { device.end_command_buffer(buffer) }.unwrap()
        });

        buffers
    }

    fn cmd_draw_object(
        device: &Device,
        buffer: vk::CommandBuffer,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        transform: Matrix4<f32>,
    ) {
        // Push constants
        unsafe {
            device.cmd_push_constants(
                buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                any_as_u8_slice(&transform),
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(buffer, index_count as _, 1, 0, 0, 0) };
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        // Move camera while holding left mouse-button
        if self.is_left_clicked && self.cursor_delta.is_some() {
            let delta = self.cursor_delta.take().unwrap();
            let x_ratio = delta[0] as f32 / self.swapchain_properties.extent.width as f32;
            let y_ratio = delta[1] as f32 / self.swapchain_properties.extent.height as f32;
            let theta = x_ratio * 180.0f32.to_radians();
            let phi = y_ratio * 90.0f32.to_radians();
            self.camera.rotate(theta, phi);
        }

        // Move camera forward/backwards with mouse wheel
        if let Some(wheel_data) = self.wheel_delta {
            self.camera.forward(wheel_data * 0.3);
        }

        let aspect = self.swapchain_properties.extent.width as f32
            / self.swapchain_properties.extent.height as f32;

        let ubo = CameraUBO {
            view: Matrix4::look_at_rh(
                self.camera.position(),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: math::perspective(Deg(45.0), aspect, 0.1, 10.0),
        };
        let ubos = [ubo];

        let buffer_mem = self.uniform_buffers[current_image as usize].memory;
        let size = size_of::<CameraUBO>() as vk::DeviceSize;

        unsafe {
            let data_ptr = self
                .vk_context
                .device()
                .map_memory(buffer_mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<f32>() as _, size);
            align.copy_from_slice(&ubos);
            self.vk_context.device().unmap_memory(buffer_mem);
        }
    }

    fn create_command_pool(
        device: &Device,
        queue_family_indices: QueueFamilyIndices,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_indices.graphics_index)
            .flags(create_flags);

        unsafe { device.create_command_pool(&command_pool_info, None) }.unwrap()
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap()
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap()
            };

            let in_flight_fence = {
                let fence_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_info, None) }.unwrap()
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects);
        }
        InFlightFrames::new(sync_objects_vec)
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximised. This is because width or height of 0 is not legal
    fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain.");

        self.wait_gpu_idle();

        self.cleanup_swapchain();

        let dimensions = self.resize_dimensions.unwrap_or([
            self.swapchain_properties.extent.width,
            self.swapchain_properties.extent.height,
        ]);

        let (swapchain, swapchain_khr, properties, images) = Self::create_swapchain_and_images(
            &self.vk_context,
            self.queue_family_indices,
            dimensions,
        );
        let swapchain_image_views =
            Self::create_swapchain_image_views(self.vk_context.device(), &images, properties);

        let render_pass = Self::create_render_pass(
            self.vk_context.device(),
            properties,
            self.msaa_samples,
            self.depth_format,
        );

        let (pipeline, layout) = Self::create_pipeline(
            self.vk_context.device(),
            self.swapchain_properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
        );

        let color_texture = Self::create_color_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            properties,
            self.msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            self.depth_format,
            properties.extent,
            self.msaa_samples,
        );

        let framebuffers = Self::create_framebuffers(
            self.vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            self.vk_context.device(),
            self.command_pool,
            &framebuffers,
            render_pass,
            properties,
            self.mesh_buffers,
            self.model_index_count,
            layout,
            &self.descriptor_sets,
            pipeline,
        );

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.swapchain_images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_layout = layout;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.swapchain_framebuffers = framebuffers;
        self.command_buffers = command_buffers;
        self.resize_dimensions = None;
    }

    fn create_texture_image(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
    ) -> Texture {
        let device = vk_context.device();

        let image = image::open("assets/images/chalet.jpg").unwrap().flipv();
        let image_as_rgb = image.to_rgba8();

        let height = image_as_rgb.height();
        let width = image_as_rgb.width();

        let max_mip_levels = ((width.min(height) as f32).log2().floor() + 1.0) as u32;

        let extent = vk::Extent2D { width, height };
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8>()) as vk::DeviceSize;

        let mut image_buffer = Self::create_buffer(
            vk_context,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = device
                .map_memory(
                    image_buffer.memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, image_buffer.size);
            align.copy_from_slice(&pixels);
            device.unmap_memory(image_buffer.memory);
        }

        let (image, image_memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            Self::transition_image_layout(
                device,
                command_pool,
                transition_queue,
                image,
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            Self::copy_buffer_to_image(
                device,
                command_pool,
                transition_queue,
                image_buffer.buffer,
                image,
                extent,
            );

            Self::create_mipmaps(
                vk_context,
                command_pool,
                transition_queue,
                image,
                extent,
                vk::Format::R8G8B8A8_UNORM,
                max_mip_levels,
            );
        }
        image_buffer.destroy(device);

        let image_view = Self::create_image_view(
            device,
            image,
            vk::Format::R8G8B8A8_UNORM,
            max_mip_levels,
            vk::ImageAspectFlags::COLOR,
        );

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(max_mip_levels as _);

            unsafe { device.create_sampler(&sampler_info, None) }.unwrap()
        };

        Texture::new(image, image_memory, image_view, Some(sampler))
    }

    fn load_model() -> (Vec<Vertex>, Vec<u32>) {
        log::debug!("Loading model.");
        let (models, _) =
            tobj::load_obj(Path::new("assets/models/chalet.obj"), &GPU_LOAD_OPTIONS).unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let u = coords[i * 2];
            let v = coords[i * 2 + 1];

            let vertex = Vertex {
                pos: [x, y, z],
                color: [1.0, 1.0, 1.0],
                coords: [u, v],
            };
            vertices.push(vertex);
        }
        (vertices, mesh.indices.clone())
    }

    fn create_image(
        vk_context: &VkContext,
        mem_properties: vk::MemoryPropertyFlags,
        extent: vk::Extent2D,
        mip_levels: u32,
        sample_count: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let device = vk_context.device();
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .tiling(tiling)
            .format(format)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(vk::ImageCreateFlags::empty());

        let image = unsafe { device.create_image(&image_info, None) }.unwrap();
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = Self::find_memory_type(
            mem_requirements,
            vk_context.get_mem_properties(),
            mem_properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            let mem = device.allocate_memory(&alloc_info, None).unwrap();
            device.bind_image_memory(image, mem, 0).unwrap();
            mem
        };

        (image, memory)
    }

    fn transition_image_layout(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        Self::execute_one_time_command(device, command_pool, transition_queue, |buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    ),
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    _ => panic!(
                        "Unsupported layout transition({:?} => {:?}).",
                        old_layout, new_layout
                    ),
                };

            let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = vk::ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format) {
                    mask |= vk::ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask);
            let barriers = [barrier];

            unsafe {
                device.cmd_pipeline_barrier(
                    buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                )
            };
        });
    }

    // TODO DB: Consider using AllocatedBuffer instead of vk::Buffer
    fn copy_buffer_to_image(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        extent: vk::Extent2D,
    ) {
        Self::execute_one_time_command(device, command_pool, transition_queue, |cmd_buffer| {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            let regions = [region];

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        });
    }

    /// Create a one time use command buffer and pass it to `executor`.
    fn execute_one_time_command<F: FnOnce(vk::CommandBuffer)>(
        device: &Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        executor: F,
    ) {
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0]
        };
        let command_buffers = [command_buffer];

        // begin recording
        {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();
        }

        executor(command_buffer);

        // end recording
        unsafe { device.end_command_buffer(command_buffer) }.unwrap();

        // submit and wait
        {
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(queue, &submit_infos, vk::Fence::null())
                    .unwrap();
                device.queue_wait_idle(queue).unwrap();
            };
        }

        // free
        unsafe { device.free_command_buffers(command_pool, &command_buffers) };
    }

    fn create_vertex_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        vertices: &[Vertex],
    ) -> AllocatedBuffer {
        Self::create_device_local_buffer_with_data::<u32, _>(
            vk_context,
            command_pool,
            transfer_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        )
    }

    fn create_index_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        indices: &[u32],
    ) -> AllocatedBuffer {
        Self::create_device_local_buffer_with_data::<u16, _>(
            vk_context,
            command_pool,
            transfer_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            indices,
        )
    }

    /// Create a buffer and its gpu memory and fill it.
    ///
    /// This function internally creates a host visible staging buffer and
    /// a device local buffer. The vertex data is first copied from the cpu
    /// to the staging buffer. Then we copy vertex data from the staging buffer
    /// to the final buffer using a one-time command buffer.
    ///
    /// # Inputs
    /// A - Memory alignment of data
    /// T - Type of data
    fn create_device_local_buffer_with_data<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> AllocatedBuffer {
        let device = vk_context.device();
        let size = size_of_val(data) as vk::DeviceSize;
        let mut staging_buffer = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_buffer.size);
            align.copy_from_slice(data);
            device.unmap_memory(staging_buffer.memory);
        };

        let device_buffer = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer.buffer,
            device_buffer.buffer,
            size,
        );

        staging_buffer.destroy(device);
        device_buffer
    }

    fn create_uniform_buffers(vk_context: &VkContext, count: usize) -> Vec<AllocatedBuffer> {
        let size = size_of::<CameraUBO>() as vk::DeviceSize;
        let mut buffers = Vec::new();

        for _ in 0..count {
            let uniform_buffer = Self::create_buffer(
                vk_context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(uniform_buffer);
        }
        buffers
    }

    /// Create a buffer and allocate its memory.
    ///
    /// # Returns
    ///
    /// The buffer, its memory and the actual size in bytes of the
    /// allocated memory since it may differ from the requested size.
    fn create_buffer(
        vk_context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> AllocatedBuffer {
        let device = vk_context.device();
        let buffer = {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { device.create_buffer(&buffer_info, None) }.unwrap()
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type = Self::find_memory_type(
                mem_requirements,
                vk_context.get_mem_properties(),
                mem_properties,
            );

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type);
            unsafe { device.allocate_memory(&alloc_info, None) }.unwrap()
        };

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.unwrap();

        AllocatedBuffer::new(buffer, memory, mem_requirements.size)
    }

    /// Copy the `size` first bytes of `src` into `dst`
    ///
    /// It's done using a command allocated from
    /// `command_pool`. The command buffer is submitted to
    /// `transfer_queue`.
    fn copy_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        Self::execute_one_time_command(device, command_pool, transfer_queue, |buffer| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            let regions = [region];

            unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) }
        });
    }

    /// Find a memory type in `mem_properties` that is suitable
    /// for `requirements` and supports `required_properties`.
    ///
    /// # Returns
    ///
    /// The index of the memory type from `mem_properties`.
    fn find_memory_type(
        requirements: vk::MemoryRequirements,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        required_properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        for i in 0..mem_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type.")
    }

    fn find_depth_format(vk_context: &VkContext) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        vk_context
            .find_supported_format(
                &candidates,
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
            .expect("Failed to find a supported depth format.")
    }

    /// Create the depth texture
    ///
    /// This function also transitions the image to be ready to be used
    /// as a depth/stencil attachment.
    fn create_depth_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        format: vk::Format,
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let (image, mem) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        Self::transition_image_layout(
            vk_context.device(),
            command_pool,
            transition_queue,
            image,
            1,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let view = Self::create_image_view(
            vk_context.device(),
            image,
            format,
            1,
            vk::ImageAspectFlags::DEPTH,
        );

        Texture::new(image, mem, view, None)
    }

    fn create_color_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let format = swapchain_properties.format.format;
        let (image, memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            swapchain_properties.extent,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        );

        Self::transition_image_layout(
            vk_context.device(),
            command_pool,
            transition_queue,
            image,
            1,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let view = Self::create_image_view(
            vk_context.device(),
            image,
            format,
            1,
            vk::ImageAspectFlags::COLOR,
        );

        Texture::new(image, memory, view, None)
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_mipmaps(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_format_properties(vk_context.physical_device(), format)
        };

        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear blitting is not supported for format {:?}.", format)
        }

        Self::execute_one_time_command(
            vk_context.device(),
            command_pool,
            transfer_queue,
            |buffer| {
                let mut barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        level_count: 1,
                        ..Default::default()
                    });

                let mut mip_width = extent.width as i32;
                let mut mip_height = extent.height as i32;
                for level in 1..mip_levels {
                    let next_mip_width = if mip_width > 1 {
                        mip_width / 2
                    } else {
                        mip_width
                    };
                    let next_mip_height = if mip_height > 1 {
                        mip_height / 2
                    } else {
                        mip_height
                    };

                    barrier.subresource_range.base_mip_level = level - 1;
                    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                    barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
                    let barriers = [barrier];

                    unsafe {
                        vk_context.device().cmd_pipeline_barrier(
                            buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &barriers,
                        )
                    };

                    let blit = vk::ImageBlit::default()
                        .src_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: mip_width,
                                y: mip_height,
                                z: 1,
                            },
                        ])
                        .src_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: level - 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .dst_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: next_mip_width,
                                y: next_mip_height,
                                z: 1,
                            },
                        ])
                        .dst_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: level,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    let blits = [blit];

                    unsafe {
                        vk_context.device().cmd_blit_image(
                            buffer,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &blits,
                            vk::Filter::LINEAR,
                        )
                    };

                    barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                    let barriers = [barrier];

                    unsafe {
                        vk_context.device().cmd_pipeline_barrier(
                            buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::FRAGMENT_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &barriers,
                        )
                    };

                    mip_width = next_mip_width;
                    mip_height = next_mip_height;
                }

                barrier.subresource_range.base_mip_level = mip_levels - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                let barriers = [barrier];

                unsafe {
                    vk_context.device().cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
                };
            },
        );
    }

    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            self.depth_texture.destroy(device);
            self.color_texture.destroy(device);
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None));
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }

    fn initialize_vulkan_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, Device, vk::Queue, vk::Queue) {
        // Select physical device
        let available_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let selected_device = available_devices
            .into_iter()
            .find(|d| Self::is_device_suitable(instance, surface, surface_khr, *d))
            .expect("No suitable physical device found.");

        let props = unsafe { instance.get_physical_device_properties(selected_device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        // Queue families for graphics and present queue
        let (graphics, present) =
            Self::find_queue_families(instance, surface, surface_khr, selected_device);
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

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);

        let device = unsafe { instance.create_device(selected_device, &device_create_info, None) }
            .expect("Failed to create logical device.");

        // graphics and present queue are created, but not retrieved yet
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_index, 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_index, 0) };

        (selected_device, device, graphics_queue, present_queue)
    }
}

impl Drop for VulkanApplication {
    fn drop(&mut self) {
        log::debug!("Dropping application");
        self.cleanup_swapchain();
        let device = self.vk_context.device();
        self.in_flight_frames.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.uniform_buffers.iter_mut().for_each(|u| {
                u.destroy(device);
            });
            self.mesh_buffers.destroy(device);
            self.texture.destroy(device);
            device.destroy_command_pool(self.transient_command_pool, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        if let Some(app) = self.vulkan.as_mut() {
            app.wheel_delta = None;
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan App with Ash")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.vulkan = Some(VulkanApplication::new(&window));
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
            WindowEvent::Resized { .. } => {
                self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.vulkan.as_mut().unwrap().is_left_clicked =
                    state == ElementState::Pressed && button == MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let app = self.vulkan.as_mut().unwrap();

                let position: (i32, i32) = position.into();
                app.cursor_delta = Some([
                    app.cursor_position[0] - position.0,
                    app.cursor_position[1] - position.1,
                ]);
                app.cursor_position = [position.0, position.1];
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, v_lines),
                ..
            } => {
                self.vulkan.as_mut().unwrap().wheel_delta = Some(v_lines);
            }
            _ => (),
        }
    }

    /// This is not the ideal place to drive rendering from.
    /// Should really be done with the RedrawRequested Event, but here we are for now.
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let app = self.vulkan.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();

        if app.dirty_swapchain {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                app.recreate_swapchain();
            } else {
                return;
            }
        }
        app.dirty_swapchain = app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.vulkan.as_ref().unwrap().wait_gpu_idle();
    }
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_index: u32,
    present_index: u32,
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}
/// Return a `&[u8]` for any sized object passed in.
unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, size_of::<T>()) }
}
