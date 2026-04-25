use crate::context::VkContext;
use crate::descriptor::{DescriptorSetLayout, DescriptorWriter, GrowableAllocator, LayoutBuilder};
use crate::pipeline::{Pipeline, PipelineBuilder};
use crate::resources::{AllocatedBuffer, AllocatedImage};
use ash::{Device, vk};
use std::mem::size_of;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPass {
    MainColor,
    Transparent,
}

#[repr(C)]
pub struct MaterialConstants {
    pub color_factors: [f32; 4],
    pub metal_rough_factors: [f32; 4],
    pub _padding: [[f32; 4]; 14],
}

impl Default for MaterialConstants {
    fn default() -> Self {
        Self {
            color_factors: [1.0, 1.0, 1.0, 1.0],
            metal_rough_factors: [1.0, 0.5, 0.0, 0.0],
            _padding: [[0.0; 4]; 14],
        }
    }
}

pub(crate) struct MaterialResources<'a> {
    pub color_image: Arc<AllocatedImage>,
    pub color_sampler: vk::Sampler,
    pub metal_rough_image: Arc<AllocatedImage>,
    pub metal_rough_sampler: vk::Sampler,
    pub data_buffer: &'a AllocatedBuffer,
    pub data_buffer_offset: u32,
}

pub(crate) struct MaterialInstance {
    pub pipeline: Arc<Pipeline>,
    pub material_set: vk::DescriptorSet,
    pub pass_type: MaterialPass,
}

pub(crate) struct GltfMetallicRoughness {
    pub opaque_pipeline: Arc<Pipeline>,
    pub transparent_pipeline: Arc<Pipeline>,
    pub material_layout: DescriptorSetLayout,
}

impl GltfMetallicRoughness {
    pub(crate) fn build_pipelines(
        context: &VkContext,
        draw_image_format: vk::Format,
        depth_image_format: vk::Format,
        scene_data_layout: &vk::DescriptorSetLayout,
    ) -> Self {
        // a) Build set 1 descriptor layout (3 bindings for VERTEX | FRAGMENT)
        let mut layout_builder = LayoutBuilder::new();
        layout_builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);
        layout_builder.add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        layout_builder.add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        let material_layout = layout_builder.build(
            &context.device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            None,
        );

        // b) Load vertex + fragment shader modules
        let load_spv = |path: &str, device: &ash::Device| -> vk::ShaderModule {
            let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing spv: {path}"));
            let mut cursor = std::io::Cursor::new(bytes);
            let words = ash::util::read_spv(&mut cursor).expect("invalid SPIR-V");
            let info = vk::ShaderModuleCreateInfo::default().code(&words);
            unsafe { device.create_shader_module(&info, None) }.unwrap()
        };
        let vert_module = load_spv("assets/shaders/mesh.vert.spv", &context.device);
        let frag_module = load_spv("assets/shaders/mesh.frag.spv", &context.device);

        // c) Push constant range
        let push_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<crate::primitives::GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let set_layouts = [*scene_data_layout, material_layout.layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));

        // Create two separate pipeline layouts (structurally identical but separate Vk objects)
        let opaque_pipeline_layout =
            unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();
        let transparent_pipeline_layout =
            unsafe { context.device.create_pipeline_layout(&layout_info, None) }.unwrap();

        // d) Opaque pipeline
        let opaque_vk_pipeline = {
            let mut builder = PipelineBuilder::init();
            builder.pipeline_layout = opaque_pipeline_layout;
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

        // e) Transparent pipeline
        let transparent_vk_pipeline = {
            let mut builder = PipelineBuilder::init();
            builder.pipeline_layout = transparent_pipeline_layout;
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

        // f) Destroy shader modules after pipeline creation
        unsafe {
            context.device.destroy_shader_module(vert_module, None);
            context.device.destroy_shader_module(frag_module, None);
        }

        // g) Return the struct
        GltfMetallicRoughness {
            opaque_pipeline: Arc::new(Pipeline::new(
                opaque_vk_pipeline,
                opaque_pipeline_layout,
                context.device.clone(),
            )),
            transparent_pipeline: Arc::new(Pipeline::new(
                transparent_vk_pipeline,
                transparent_pipeline_layout,
                context.device.clone(),
            )),
            material_layout,
        }
    }

    pub(crate) fn write_material(
        &self,
        device: &Device,
        pass_type: MaterialPass,
        resources: &MaterialResources,
        allocator: &mut GrowableAllocator,
    ) -> MaterialInstance {
        let pipeline = match pass_type {
            MaterialPass::MainColor => Arc::clone(&self.opaque_pipeline),
            MaterialPass::Transparent => Arc::clone(&self.transparent_pipeline),
        };

        let material_set = allocator.allocate(device, self.material_layout.layout);

        let mut writer = DescriptorWriter::new();
        writer.write_image(
            1,
            resources.color_image.view,
            resources.color_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        writer.write_image(
            2,
            resources.metal_rough_image.view,
            resources.metal_rough_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        writer.write_buffer(
            0,
            resources.data_buffer.buffer,
            size_of::<MaterialConstants>() as u64,
            resources.data_buffer_offset as u64,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        writer.update_set(device, material_set);

        MaterialInstance {
            pipeline,
            material_set,
            pass_type,
        }
    }
}
