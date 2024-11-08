use ash::vk;
use cgmath::Matrix4;
use std::mem::offset_of;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 3],
    pub(crate) color: [f32; 3],
    pub(crate) coords: [f32; 2],
}

impl Vertex {
    pub(crate) fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub(crate) fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as _);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as _);
        let coord_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, coords) as _);
        [position_desc, color_desc, coord_desc]
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct CameraUBO {
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
}

impl CameraUBO {
    pub(crate) fn get_descriptor_set_layout_bindings<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
    }
}

#[derive(Clone, Copy)]
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl AllocatedBuffer {
    pub fn new(buffer: vk::Buffer, memory: vk::DeviceMemory, size: vk::DeviceSize) -> Self {
        Self {
            buffer,
            memory,
            size,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct MeshBuffers {
    pub index: AllocatedBuffer,
    pub vertex: AllocatedBuffer,
}

impl MeshBuffers {
    pub fn new(vertex_buffer: AllocatedBuffer, index_buffer: AllocatedBuffer) -> Self {
        Self {
            vertex: vertex_buffer,
            index: index_buffer,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.vertex.destroy(device);
        self.index.destroy(device);
    }
}
