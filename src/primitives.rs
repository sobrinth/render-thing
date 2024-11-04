use crate::static_data::VERTEX_SIZE;
use ash::vk;
use cgmath::Matrix4;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 2],
    pub(crate) color: [f32; 3],
}

impl Vertex {
    pub(crate) fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(VERTEX_SIZE as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub(crate) fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(8);
        [position_desc, color_desc]
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct UniformBufferObject {
    pub(crate) model: Matrix4<f32>,
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
}

impl UniformBufferObject {
    pub(crate) fn get_descriptor_set_layout_bindings<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
    }
}