# Module: resources.rs

## Purpose
GPU memory allocation wrappers. Manages buffer and image lifetimes via vk-mem allocator. Provides typed allocations with RAII cleanup and upload helpers for mesh/image data.

## Key Types
- **AllocatedBuffer**: Holds vk::Buffer, vk_mem::Allocation, allocation info. Implements Drop to destroy buffer/allocation.
- **AllocatedImage**: Holds vk::Image, vk::ImageView, allocation. Implements Drop with proper ordering (view then image).
- **ImageCreateInfo**: Builder struct for image parameters (resolution, format, usage, aspect, mipmaps).
- **Sampler**: Wrapper for vk::Sampler with device ownership for drop.

## Notable Patterns
- **Arc<Allocator> cloning**: Allocator stored as Arc to avoid premature deallocation
- **Immediate submit pattern**: create_from_data() uses ImmediateSubmitData to upload texture data
- **Two-stage mesh upload**: Staging buffer (host-visible) → device buffer via immediate submit
- **Mapped memory management**: Manual map/unmap pairing with ptr::copy_nonoverlapping; lifetime scoped

## Unsafe Blocks
- Lines 48, 136, 203: allocator.create_buffer/image/map_memory - safe per vk-mem docs; allocation lifetime managed
- Line 204: ptr::copy_nonoverlapping - valid while mapped; data_size computed from slice; no overlap
- Lines 322-328: Multiple ptr::copy_nonoverlapping calls for vertex + index data - safe offsets (vertex_buffer_size boundary known)
- Line 298: get_buffer_device_address - safe; device address valid for command buffer lifetime

## Dependencies
- command_buffer (ImmediateSubmitData, transition_image)
- context (VkContext, QueueData)
- primitives (Vertex, GPUMeshBuffers)
- vk_mem (allocator trait)
