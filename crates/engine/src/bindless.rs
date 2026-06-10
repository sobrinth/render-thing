use crate::descriptor::{DescriptorSetLayout, LayoutBuilder};
use crate::resources::AllocatedBuffer;
use ash::{Device, vk};
use std::sync::Arc;

/// Must match the array size in assets/shaders/input_structures.glsl.
pub(crate) const MAX_TEXTURES: u32 = 4096;
pub(crate) const MAX_MATERIALS: u32 = 1024;

/// GPU-side material record (std430-compatible, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GPUMaterial {
    pub color_factors: [f32; 4],
    pub metal_rough_factors: [f32; 4],
    pub color_tex_index: u32,
    pub metal_rough_tex_index: u32,
    pub _pad: [u32; 2],
}

/// One global descriptor set holding every texture and all material constants.
///
/// Binding 0: COMBINED_IMAGE_SAMPLER[MAX_TEXTURES], PARTIALLY_BOUND | UPDATE_AFTER_BIND —
/// slots are written incrementally as textures register; unwritten slots are never read.
/// Binding 1: STORAGE_BUFFER with MAX_MATERIALS GPUMaterial records, written once here;
/// only the buffer *contents* change afterwards (host-visible, persistently mapped).
pub(crate) struct BindlessResources {
    pub(crate) layout: DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    material_buffer: AllocatedBuffer,
    device: Device,
}

impl BindlessResources {
    pub(crate) fn new(device: &Device, gpu_alloc: &Arc<vk_mem::Allocator>) -> Self {
        let mut builder = LayoutBuilder::new();
        builder.add_binding_array(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_TEXTURES);
        builder.add_binding(1, vk::DescriptorType::STORAGE_BUFFER);
        let layout = builder.build_with_binding_flags(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            &[
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::empty(),
            ],
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
        );

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TEXTURES),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(core::slice::from_ref(&layout.layout));
        let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap()[0];

        let material_buffer = AllocatedBuffer::create(
            gpu_alloc,
            (MAX_MATERIALS as usize * size_of::<GPUMaterial>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::Auto,
            Some(
                vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
        );

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: material_buffer.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(core::slice::from_ref(&buffer_info));
        unsafe { device.update_descriptor_sets(core::slice::from_ref(&write), &[]) };

        Self {
            layout,
            pool,
            set,
            material_buffer,
            device: device.clone(),
        }
    }

    /// Write texture `index` into the global array. Safe to call while the set is
    /// bound in in-flight command buffers (UPDATE_AFTER_BIND): fresh slots are never
    /// referenced by pending draws.
    pub(crate) fn write_texture(&self, index: u32, view: vk::ImageView, sampler: vk::Sampler) {
        assert!(
            index < MAX_TEXTURES,
            "texture registry exceeded MAX_TEXTURES ({MAX_TEXTURES})"
        );
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(core::slice::from_ref(&image_info));
        unsafe {
            self.device
                .update_descriptor_sets(core::slice::from_ref(&write), &[])
        };
    }

    /// Write material record `index` into the mapped SSBO. Materials are immutable
    /// after creation, so fresh slots are never read by in-flight frames.
    pub(crate) fn write_material(&self, index: u32, material: GPUMaterial) {
        assert!(
            index < MAX_MATERIALS,
            "material registry exceeded MAX_MATERIALS ({MAX_MATERIALS})"
        );
        unsafe {
            let dst =
                (self.material_buffer.info.mapped_data as *mut GPUMaterial).add(index as usize);
            dst.write(material);
        }
    }
}

impl Drop for BindlessResources {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}
