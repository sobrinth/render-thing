use crate::command_buffer::{BarrierScope, ImmediateSubmitData, transition_image};
use crate::context::{QueueData, VkContext};
use crate::primitives::{GPUMeshBuffers, Vertex};
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

pub(crate) struct ImageCreateInfo {
    pub(crate) resolution: (u32, u32),
    pub(crate) format: vk::Format,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) aspect_flags: vk::ImageAspectFlags,
    pub(crate) mip_mapped: bool,
}

pub(crate) struct AllocatedImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) extent: vk::Extent3D,
    pub(crate) format: vk::Format,
    pub(crate) mip_levels: u32,
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
            mip_levels,
            device: context.device.clone(),
            allocator: Arc::clone(gpu_alloc),
        }
    }
}

/// Fill mips 1..n by successively blitting each level down to the next.
/// Expects all mip levels in TRANSFER_DST; leaves the whole image in SHADER_READ_ONLY.
fn generate_mipmaps(device: &Device, cmd: vk::CommandBuffer, image: &AllocatedImage) {
    let mut mip_width = image.extent.width as i32;
    let mut mip_height = image.extent.height as i32;

    for mip in 0..image.mip_levels {
        // Transition the current level (just written by the copy or previous blit)
        // to TRANSFER_SRC so it can be read by the next blit.
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image.image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

        if mip + 1 == image.mip_levels {
            break;
        }

        let next_width = i32::max(mip_width / 2, 1);
        let next_height = i32::max(mip_height / 2, 1);

        let blit_region = vk::ImageBlit2::default()
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: mip_width,
                    y: mip_height,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: next_width,
                    y: next_height,
                    z: 1,
                },
            ])
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(mip)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(mip + 1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let blit_info = vk::BlitImageInfo2::default()
            .src_image(image.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(image.image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .filter(vk::Filter::LINEAR)
            .regions(std::slice::from_ref(&blit_region));

        unsafe { device.cmd_blit_image2(cmd, &blit_info) };

        mip_width = next_width;
        mip_height = next_height;
    }

    // All levels are now TRANSFER_SRC; make the whole image sampleable.
    transition_image(
        device,
        cmd,
        image.image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        BarrierScope::TRANSFER_READ,
        BarrierScope::FRAGMENT_SAMPLED_READ,
    );
}

impl AllocatedImage {
    pub(crate) fn create_from_data(
        gpu_alloc: &Arc<vk_mem::Allocator>,
        context: &VkContext,
        imm_data: &ImmediateSubmitData,
        graphics_queue: &QueueData,
        data: &[u32],
        info: ImageCreateInfo,
    ) -> AllocatedImage {
        let extent = vk::Extent3D {
            width: info.resolution.0,
            height: info.resolution.1,
            depth: 1,
        };

        let data_size = size_of_val(data);

        let upload_buffer = AllocatedBuffer::create(
            gpu_alloc,
            data_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            Some(
                vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
        );

        let new_image = AllocatedImage::create(
            context,
            gpu_alloc,
            info.resolution,
            info.format,
            info.usage,
            info.aspect_flags,
            info.mip_mapped,
        );

        let dst_data = upload_buffer.info.mapped_data as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst_data, data_size) };

        imm_data.submit(&context.device, graphics_queue, |cmd| {
            transition_image(
                &context.device,
                cmd.handle(),
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                BarrierScope::NONE,
                BarrierScope::TRANSFER_WRITE,
            );

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: info.aspect_flags,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent);

            unsafe {
                context.device.cmd_copy_buffer_to_image(
                    cmd.handle(),
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                )
            }

            if new_image.mip_levels > 1 {
                generate_mipmaps(&context.device, cmd.handle(), &new_image);
            } else {
                transition_image(
                    &context.device,
                    cmd.handle(),
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    BarrierScope::TRANSFER_WRITE,
                    BarrierScope::FRAGMENT_SAMPLED_READ,
                );
            }
        });

        new_image
    }
}

impl AllocatedImage {
    /// Upload raw RGBA8 bytes (4 components per pixel, one byte each).
    /// Use this for gltf::image::Data::pixels which are laid out [R,G,B,A,R,G,B,A,...].
    pub(crate) fn create_from_bytes(
        gpu_alloc: &Arc<vk_mem::Allocator>,
        context: &VkContext,
        imm_data: &ImmediateSubmitData,
        graphics_queue: &QueueData,
        data: &[u8],
        info: ImageCreateInfo,
    ) -> AllocatedImage {
        let extent = vk::Extent3D {
            width: info.resolution.0,
            height: info.resolution.1,
            depth: 1,
        };

        let upload_buffer = AllocatedBuffer::create(
            gpu_alloc,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            Some(
                vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
        );

        let new_image = AllocatedImage::create(
            context,
            gpu_alloc,
            info.resolution,
            info.format,
            info.usage,
            info.aspect_flags,
            info.mip_mapped,
        );

        let dst_data = upload_buffer.info.mapped_data as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst_data, data.len()) };

        imm_data.submit(&context.device, graphics_queue, |cmd| {
            transition_image(
                &context.device,
                cmd.handle(),
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                BarrierScope::NONE,
                BarrierScope::TRANSFER_WRITE,
            );

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: info.aspect_flags,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent);

            unsafe {
                context.device.cmd_copy_buffer_to_image(
                    cmd.handle(),
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                )
            }

            if new_image.mip_levels > 1 {
                generate_mipmaps(&context.device, cmd.handle(), &new_image);
            } else {
                transition_image(
                    &context.device,
                    cmd.handle(),
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    BarrierScope::TRANSFER_WRITE,
                    BarrierScope::FRAGMENT_SAMPLED_READ,
                );
            }
        });

        new_image
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

pub(crate) fn upload_mesh_buffers(
    gpu_alloc: &Arc<vk_mem::Allocator>,
    context: &VkContext,
    imm_data: &ImmediateSubmitData,
    graphics_queue: &QueueData,
    indices: &[u32],
    vertices: &[Vertex],
) -> GPUMeshBuffers {
    let vertex_buffer_size = size_of_val(vertices);
    let index_buffer_size = size_of_val(indices);

    let vertex_buffer = AllocatedBuffer::create(
        gpu_alloc,
        vertex_buffer_size as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk_mem::MemoryUsage::AutoPreferDevice,
        None,
    );
    let index_buffer = AllocatedBuffer::create(
        gpu_alloc,
        index_buffer_size as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk_mem::MemoryUsage::AutoPreferDevice,
        None,
    );

    let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
    let device_address = unsafe {
        context
            .device
            .get_buffer_device_address(&device_address_info)
    };

    let meshes = GPUMeshBuffers {
        vertex_buffer,
        index_buffer,
        vertex_buffer_address: device_address,
    };

    let staging = AllocatedBuffer::create(
        gpu_alloc,
        (vertex_buffer_size + index_buffer_size) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk_mem::MemoryUsage::AutoPreferHost,
        Some(
            vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ),
    );

    let dst_data = staging.info.mapped_data as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(vertices.as_ptr() as *const u8, dst_data, vertex_buffer_size);
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            dst_data.add(vertex_buffer_size),
            index_buffer_size,
        );
    };

    imm_data.submit(&context.device, graphics_queue, |cmd| {
        let vertex_copy = &[vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(0)
            .size(vertex_buffer_size as u64)];
        unsafe {
            context.device.cmd_copy_buffer(
                cmd.handle(),
                staging.buffer,
                meshes.vertex_buffer.buffer,
                vertex_copy,
            )
        }

        let index_copy = &[vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(vertex_buffer_size as u64)
            .size(index_buffer_size as u64)];

        unsafe {
            context.device.cmd_copy_buffer(
                cmd.handle(),
                staging.buffer,
                meshes.index_buffer.buffer,
                index_copy,
            )
        }
    });

    meshes
}
