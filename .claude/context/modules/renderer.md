# Module: renderer.rs

## Purpose
Core Vulkan renderer orchestration. Manages frame submission, pipeline creation, resource initialization, and GPU state. Central hub connecting context, frame data, pipelines, and synchronization.

## Key Types
- **VulkanRenderer**: Holds context, resources (wrapped in ManuallyDrop for custom drop order)
- **RendererResources**: All mutable GPU state (frame_number, window_size, render_scale, camera, UI context, swapchain, frames, pipelines, meshes, descriptor allocators, images, samplers)

## Notable Patterns
- **ManuallyDrop wrapper**: Enforces drop order (resources before context)
- **Builder pattern**: PipelineBuilder used to configure graphics pipeline
- **Factory methods**: Static functions for creating immediate submit data, frame data, pipelines, descriptor sets
- **Lazy mesh loading**: Meshes loaded after renderer initialization via load_gltf_meshes

## Unsafe Blocks
Shader module loads and destroys (lines 131-141): once created, lifetime is owned by pipeline; safe to destroy after pipeline creation. Vulkan spec guarantees shader modules can be destroyed post-compile.

## Dependencies
- frame (FrameData, frame loop implementation in impl blocks)
- context (VkContext, QueueData)
- descriptor (DescriptorSetLayout, DescriptorWriter, GrowableAllocator)
- pipeline (Pipeline, PipelineLayout, PipelineBuilder, ComputeEffect)
- resources (AllocatedBuffer, AllocatedImage, Sampler, upload_mesh_buffers)
- meshes (MeshAsset, load_gltf_meshes)
- sync (Fence, Semaphore)
- camera (Camera)
- ui (UiContext)
- primitives (Vertex, GPUDrawPushConstants, GPUSceneData, GPUMeshBuffers)
