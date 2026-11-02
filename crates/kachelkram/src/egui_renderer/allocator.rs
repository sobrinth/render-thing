use super::RendererResult;
use ash::{Device, vk};
use std::sync::Arc;
use vk_mem::{
    Alloc, AllocationCreateFlags, AllocationCreateInfo, Allocator as VkMemAllocator, MemoryUsage,
};

pub(crate) type Memory = vk_mem::Allocation;

pub(crate) struct Allocator {
    allocator: Arc<VkMemAllocator>,
}

impl Allocator {
    pub(crate) fn new(allocator: Arc<VkMemAllocator>) -> Self {
        Self { allocator }
    }

    pub(crate) fn create_buffer(
        &mut self,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer_alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            self.allocator
                .create_buffer(&buffer_info, &buffer_alloc_info)?
        };

        Ok((buffer, allocation))
    }

    pub(crate) fn create_image(
        &mut self,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty());

        let image_alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            self.allocator
                .create_image(&image_info, &image_alloc_info)?
        };

        Ok((image, allocation))
    }

    pub(crate) fn destroy_buffer(&mut self, buffer: vk::Buffer, mut memory: Memory) {
        unsafe { self.allocator.destroy_buffer(buffer, &mut memory) };
    }

    pub(crate) fn destroy_image(&mut self, image: vk::Image, mut memory: Memory) {
        unsafe { self.allocator.destroy_image(image, &mut memory) };
    }

    pub(crate) fn update_buffer<T: Copy>(
        &mut self,
        _device: &Device,
        memory: &mut Memory,
        data: &[T],
    ) -> RendererResult<()> {
        let size = std::mem::size_of_val(data) as _;

        let data_ptr = unsafe { self.allocator.map_memory(memory)? as *mut std::ffi::c_void };
        let mut align =
            unsafe { ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as _, size) };
        align.copy_from_slice(data);
        unsafe { self.allocator.unmap_memory(memory) };

        Ok(())
    }
}
