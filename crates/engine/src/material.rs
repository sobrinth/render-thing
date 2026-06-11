use crate::context::VkContext;
use crate::pipeline::{Pipeline, PipelineBuilder, PipelineLayout};
use ash::vk;
use std::mem::size_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPass {
    MainColor,
    Transparent,
}

/// Per-material constants uploaded to the GPU.
///
/// `color_factors`: RGBA multiplier on the colour texture. [r, g, b, a].
///
/// `metal_rough_factors`: scales the metallic-roughness texture channels.
///   - x: metallic scale  (multiplies texture B; 0 = fully dielectric, 1 = fully metallic)
///   - y: roughness scale (multiplies texture G; 0 = mirror, 1 = fully diffuse)
///   - z, w: unused
#[repr(C)]
pub struct MaterialConstants {
    pub color_factors: [f32; 4],
    pub metal_rough_factors: [f32; 4],
}

impl Default for MaterialConstants {
    fn default() -> Self {
        Self {
            color_factors: [1.0, 1.0, 1.0, 1.0],
            metal_rough_factors: [1.0, 0.5, 0.0, 0.0],
        }
    }
}

pub(crate) struct GltfMetallicRoughness {
    pub opaque_pipeline: Pipeline,
    pub transparent_pipeline: Pipeline,
    pub pipeline_layout: PipelineLayout,
}

impl GltfMetallicRoughness {
    pub(crate) fn build_pipelines(
        context: &VkContext,
        draw_image_format: vk::Format,
        depth_image_format: vk::Format,
        scene_data_layout: &vk::DescriptorSetLayout,
        bindless_layout: &vk::DescriptorSetLayout,
    ) -> Self {
        let load_spv = |path: &str, device: &ash::Device| -> vk::ShaderModule {
            let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing spv: {path}"));
            let mut cursor = std::io::Cursor::new(bytes);
            let words = ash::util::read_spv(&mut cursor).expect("invalid SPIR-V");
            let info = vk::ShaderModuleCreateInfo::default().code(&words);
            unsafe { device.create_shader_module(&info, None) }.unwrap()
        };
        let vert_module = load_spv("assets/shaders/mesh.vert.spv", &context.device);
        let frag_module = load_spv("assets/shaders/mesh.frag.spv", &context.device);

        let push_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<crate::primitives::GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let set_layouts = [*scene_data_layout, *bindless_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));

        let pipeline_layout =
            unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        let opaque_vk_pipeline = {
            let mut builder = PipelineBuilder::init();
            builder.pipeline_layout = pipeline_layout;
            builder.set_shaders(vert_module, frag_module);
            builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            builder.set_polygon_mode(vk::PolygonMode::FILL);
            builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE);
            builder.set_multisampling_none();
            builder.disable_blending();
            builder.enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL);
            builder.set_color_attachment_format(draw_image_format);
            builder.set_depth_format(depth_image_format);
            builder.build(&context.device)
        };

        let transparent_vk_pipeline = {
            let mut builder = PipelineBuilder::init();
            builder.pipeline_layout = pipeline_layout;
            builder.set_shaders(vert_module, frag_module);
            builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            builder.set_polygon_mode(vk::PolygonMode::FILL);
            builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE);
            builder.set_multisampling_none();
            builder.enable_blending_alpha_blend();
            builder.enable_depth_test(false, vk::CompareOp::GREATER_OR_EQUAL);
            builder.set_color_attachment_format(draw_image_format);
            builder.set_depth_format(depth_image_format);
            builder.build(&context.device)
        };

        unsafe {
            context.device.destroy_shader_module(vert_module, None);
            context.device.destroy_shader_module(frag_module, None);
        }

        // The pipelines share one layout owned by `pipeline_layout`; passing null
        // here so Pipeline::drop skips layout destruction (no-op per Vulkan spec).
        GltfMetallicRoughness {
            opaque_pipeline: Pipeline::new(
                opaque_vk_pipeline,
                vk::PipelineLayout::null(),
                context.device.clone(),
            ),
            transparent_pipeline: Pipeline::new(
                transparent_vk_pipeline,
                vk::PipelineLayout::null(),
                context.device.clone(),
            ),
            pipeline_layout: PipelineLayout::new(pipeline_layout, context.device.clone()),
        }
    }
}
