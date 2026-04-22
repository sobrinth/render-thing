use crate::context::VkContext;
use ash::{Device, vk};
use std::sync::Arc;
use vk_mem::Alloc;

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub info: vk_mem::AllocationInfo,
    allocator: Arc<vk_mem::Allocator>,
}

impl std::fmt::Debug for AllocatedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocatedBuffer")
            .field("buffer", &self.buffer)
            .field("info", &self.info)
            .finish()
    }
}

impl AllocatedBuffer {
    pub fn create(
        allocator: &Arc<vk_mem::Allocator>,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: Option<vk_mem::AllocationCreateFlags>,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        let alloc_create_info = if let Some(flags) = flags {
            vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                flags,
                ..Default::default()
            }
        } else {
            vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                ..Default::default()
            }
        };

        let (buffer, allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &alloc_create_info) }.unwrap();
        let info = allocator.get_allocation_info(&allocation);

        AllocatedBuffer {
            buffer,
            allocation,
            info,
            allocator: Arc::clone(allocator),
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation)
        }
    }
}

pub(crate) struct AllocatedImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) extent: vk::Extent3D,
    pub(crate) format: vk::Format,
    device: Device,
    allocator: Arc<vk_mem::Allocator>,
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.allocator
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}

impl AllocatedImage {
    pub fn create(
        context: &VkContext,
        gpu_alloc: &Arc<vk_mem::Allocator>,
        image_resolution: (u32, u32),
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
        mip_mapped: bool,
    ) -> AllocatedImage {
        let extent = vk::Extent3D {
            width: image_resolution.0,
            height: image_resolution.1,
            depth: 1,
        };

        let mip_levels = if mip_mapped {
            u32::ilog2(u32::max(image_resolution.0, image_resolution.1)) + 1
        } else {
            1u32
        };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (image, allocation) =
            unsafe { gpu_alloc.create_image(&create_info, &alloc_info) }.unwrap();

        let view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .image(image)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1)
                    .aspect_mask(aspect_flags),
            );

        let view = unsafe { context.device.create_image_view(&view_create_info, None) }.unwrap();

        AllocatedImage {
            image,
            view,
            allocation,
            extent,
            format,
            device: context.device.clone(),
            allocator: Arc::clone(gpu_alloc),
        }
    }
}

pub(crate) struct Sampler {
    pub(crate) sampler: vk::Sampler,
    device: Device,
}

impl Sampler {
    pub(crate) fn new(sampler: vk::Sampler, device: Device) -> Self {
        Self { sampler, device }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.destroy_sampler(self.sampler, None) }
    }
}
