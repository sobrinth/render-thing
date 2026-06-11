use ash::Device;
use ash::vk;
use itertools::Itertools;

pub(crate) struct DescriptorSetLayout {
    pub(crate) layout: vk::DescriptorSetLayout,
    device: Device,
}

impl DescriptorSetLayout {
    pub(crate) fn new(layout: vk::DescriptorSetLayout, device: Device) -> Self {
        Self { layout, device }
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_set_layout(self.layout, None) }
    }
}

pub(crate) struct LayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl LayoutBuilder<'_> {
    pub(crate) fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }
    pub(crate) fn add_binding(&mut self, binding: u32, descriptor_type: vk::DescriptorType) {
        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(1);
        self.bindings.push(layout_binding);
    }

    pub(crate) fn add_binding_array(
        &mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        count: u32,
    ) {
        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(count);
        self.bindings.push(layout_binding);
    }

    pub(crate) fn clear(&mut self) {
        self.bindings.clear();
    }

    pub(crate) fn build(
        &mut self,
        device: &Device,
        stage_flags: vk::ShaderStageFlags,
        create_flags: Option<vk::DescriptorSetLayoutCreateFlags>,
    ) -> DescriptorSetLayout {
        self.bindings
            .iter_mut()
            .for_each(|binding| binding.stage_flags |= stage_flags);

        let mut info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        if let Some(flags) = create_flags {
            info = info.flags(flags);
        }

        let layout = unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap();
        DescriptorSetLayout::new(layout, device.clone())
    }

    pub(crate) fn build_with_binding_flags(
        &mut self,
        device: &Device,
        stage_flags: vk::ShaderStageFlags,
        binding_flags: &[vk::DescriptorBindingFlags],
        create_flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> DescriptorSetLayout {
        assert_eq!(
            binding_flags.len(),
            self.bindings.len(),
            "one binding-flags entry per binding required"
        );
        self.bindings
            .iter_mut()
            .for_each(|binding| binding.stage_flags |= stage_flags);

        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(create_flags)
            .push_next(&mut flags_info);

        let layout = unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap();
        DescriptorSetLayout::new(layout, device.clone())
    }
}

pub(crate) struct Allocator {
    pool: vk::DescriptorPool,
    device: Device,
}

impl Allocator {
    pub(crate) fn init_pool(
        device: &Device,
        max_sets: u32,
        pool_ratios: Vec<PoolSizeRatio>,
    ) -> Self {
        let pool_sizes = pool_ratios
            .iter()
            .map(|p| {
                vk::DescriptorPoolSize::default()
                    .ty(p.descriptor_type)
                    .descriptor_count((max_sets as f32 * p.ratio) as u32)
            })
            .collect_vec();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::default())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();

        Self {
            pool,
            device: device.clone(),
        }
    }

    pub(crate) fn allocate(
        &self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(core::slice::from_ref(&layout));

        unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap()[0]
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct PoolSizeRatio {
    pub(crate) descriptor_type: vk::DescriptorType,
    pub(crate) ratio: f32,
}

pub struct DescriptorWriter<'a> {
    image_infos: Vec<(vk::DescriptorImageInfo, vk::WriteDescriptorSet<'a>)>,
    buffer_infos: Vec<(vk::DescriptorBufferInfo, vk::WriteDescriptorSet<'a>)>,
}

impl<'a> DescriptorWriter<'a> {
    pub fn new() -> Self {
        Self {
            image_infos: Vec::new(),
            buffer_infos: Vec::new(),
        }
    }
    pub(crate) fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        offset: vk::DeviceSize,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range: size,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_count(1)
            .descriptor_type(descriptor_type);

        self.buffer_infos.push((buffer_info, write));
    }

    pub fn write_image(
        &mut self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: layout,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_count(1)
            .descriptor_type(descriptor_type);

        self.image_infos.push((image_info, write));
    }

    pub fn clear(&mut self) {
        self.buffer_infos.clear();
        self.image_infos.clear();
    }

    pub fn update_set(&mut self, device: &Device, desc_set: vk::DescriptorSet) {
        let mut writes = Vec::new();

        for (buffer_info, write) in self.buffer_infos.iter_mut() {
            write.dst_set = desc_set;
            write.p_buffer_info = buffer_info;
            writes.push(*write);
        }

        for (image_info, write) in self.image_infos.iter_mut() {
            write.dst_set = desc_set;
            write.p_image_info = image_info;
            writes.push(*write);
        }

        unsafe { device.update_descriptor_sets(&writes, &[]) }
    }
}
