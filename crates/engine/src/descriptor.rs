use ash::vk;
use itertools::Itertools;

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

    pub(crate) fn clear(&mut self) {
        self.bindings.clear();
    }

    pub(crate) fn build(
        &mut self,
        device: &ash::Device,
        stage_flags: vk::ShaderStageFlags,
        create_flags: Option<vk::DescriptorSetLayoutCreateFlags>,
    ) -> vk::DescriptorSetLayout {
        self.bindings
            .iter_mut()
            .for_each(|binding| binding.stage_flags |= stage_flags);

        let mut info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        if let Some(flags) = create_flags {
            info = info.flags(flags);
        }

        unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap()
    }
}

pub(crate) struct Allocator {
    pool: vk::DescriptorPool,
}

impl Allocator {
    pub(crate) fn init_pool(
        device: &ash::Device,
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

        Self { pool }
    }

    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let layouts = &[layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap()[0]
    }

    pub(crate) fn clear_descriptors(&self, device: &ash::Device) {
        unsafe { device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()) }
            .unwrap();
    }

    pub(crate) fn destroy_pool(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct PoolSizeRatio {
    pub(crate) descriptor_type: vk::DescriptorType,
    pub(crate) ratio: f32,
}
