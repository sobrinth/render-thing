use ash::{Device, vk};
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
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(core::slice::from_ref(&layout));

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

pub(crate) struct GrowableAllocator {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl GrowableAllocator {
    fn init(device: &Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Self {
        let pool = Self::create_pool(device, max_sets, pool_ratios.clone());

        Self {
            ratios: pool_ratios,
            full_pools: Vec::new(),
            ready_pools: vec![pool],
            sets_per_pool: (max_sets as f32 * 1.5) as u32,
        }
    }

    fn clear_pools(&mut self, device: &Device) {
        for poll in self.ready_pools.drain(..) {
            unsafe {
                device
                    .reset_descriptor_pool(poll, vk::DescriptorPoolResetFlags::empty())
                    .unwrap()
            }
        }

        for pool in self.full_pools.drain(..) {
            unsafe {
                device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap()
            }
            self.ready_pools.push(pool);
        }
    }

    fn destroy_pools(&mut self, device: &Device) {
        for poll in self.ready_pools.drain(..) {
            unsafe { device.destroy_descriptor_pool(poll, None) }
        }

        for pool in self.full_pools.drain(..) {
            unsafe { device.destroy_descriptor_pool(pool, None) }
        }
    }

    fn allocate(&mut self, device: &Device, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let mut pool_to_use = self.get_pool(device);

        // do I need pnext?
        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool_to_use)
            .set_layouts(core::slice::from_ref(&layout));

        let res = unsafe { device.allocate_descriptor_sets(&alloc_info) };
        let ds = match res {
            Ok(sets) => sets[0],
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool_to_use);

                pool_to_use = self.get_pool(device);
                alloc_info = alloc_info.descriptor_pool(pool_to_use);
                unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap()[0] }
            }
            Err(e) => panic!("Failed to allocate descriptor set: {:?}", e),
        };

        self.ready_pools.push(pool_to_use);
        ds
    }

    fn get_pool(&mut self, device: &Device) -> vk::DescriptorPool {
        if !self.ready_pools.is_empty() {
            self.ready_pools.pop().unwrap()
        } else {
            let c_pool = Self::create_pool(device, self.sets_per_pool, self.ratios.clone());

            self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
            if self.sets_per_pool > 4092 {
                self.sets_per_pool = 4092;
            }
            c_pool
        }
    }

    fn create_pool(
        device: &Device,
        set_count: u32,
        pool_ratios: Vec<PoolSizeRatio>,
    ) -> vk::DescriptorPool {
        let mut pool_sizes = vec![];
        for ratio in &pool_ratios {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: ratio.descriptor_type,
                descriptor_count: (ratio.ratio * set_count as f32) as u32,
            })
        }

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe { device.create_descriptor_pool(&create_info, None) }.unwrap()
    }
}
