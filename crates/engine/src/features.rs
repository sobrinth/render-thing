use ash::vk;

#[derive(Default)]
pub(crate) struct FeatureChain {
    pub(crate) core: vk::PhysicalDeviceFeatures,
    pub(crate) v12: vk::PhysicalDeviceVulkan12Features<'static>,
    pub(crate) v13: vk::PhysicalDeviceVulkan13Features<'static>,
    pub(crate) swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>,
}

type FeatureFlag = for<'c> fn(&'c mut FeatureChain) -> &'c mut vk::Bool32;

const REQUIRED: &[(&str, FeatureFlag)] = &[
    ("samplerAnisotropy", |c| &mut c.core.sampler_anisotropy),
    ("multiDrawIndirect", |c| &mut c.core.multi_draw_indirect),
    ("drawIndirectFirstInstance", |c| {
        &mut c.core.draw_indirect_first_instance
    }),
    ("bufferDeviceAddress", |c| &mut c.v12.buffer_device_address),
    ("descriptorIndexing", |c| &mut c.v12.descriptor_indexing),
    ("descriptorBindingPartiallyBound", |c| {
        &mut c.v12.descriptor_binding_partially_bound
    }),
    ("descriptorBindingSampledImageUpdateAfterBind", |c| {
        &mut c.v12.descriptor_binding_sampled_image_update_after_bind
    }),
    ("timelineSemaphore", |c| &mut c.v12.timeline_semaphore),
    ("drawIndirectCount", |c| &mut c.v12.draw_indirect_count),
    ("dynamicRendering", |c| &mut c.v13.dynamic_rendering),
    ("synchronization2", |c| &mut c.v13.synchronization2),
    ("swapchainMaintenance1", |c| {
        &mut c.swapchain_maintenance1.swapchain_maintenance1
    }),
];

impl FeatureChain {
    pub(crate) fn required() -> Self {
        let mut chain = Self::default();
        for (_, flag) in REQUIRED {
            *flag(&mut chain) = vk::TRUE;
        }
        chain
    }

    pub(crate) fn supported(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let mut chain = Self::default();
        let mut features2 = chain.features2();
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let core = features2.features;
        chain.core = core;
        chain
    }

    pub(crate) fn missing_required(&mut self) -> Vec<&'static str> {
        REQUIRED
            .iter()
            .filter(|(_, flag)| *flag(self) == vk::FALSE)
            .map(|(name, _)| *name)
            .collect()
    }

    pub(crate) fn features2(&mut self) -> vk::PhysicalDeviceFeatures2<'_> {
        // `push_next` follows the pushed struct's existing `p_next`, so clear links
        // left over from an earlier chaining (they'd dangle after a move).
        self.v12.p_next = std::ptr::null_mut();
        self.v13.p_next = std::ptr::null_mut();
        self.swapchain_maintenance1.p_next = std::ptr::null_mut();
        vk::PhysicalDeviceFeatures2::default()
            .features(self.core)
            .push_next(&mut self.v12)
            .push_next(&mut self.v13)
            .push_next(&mut self.swapchain_maintenance1)
    }
}
