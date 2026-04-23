# Module: primitives.rs

## Purpose
GPU-facing data structures. Defines vertex layout, push constants, and scene data as repr(C) structs for direct GPU transmission.

## Key Types
- **Vertex**: Position (f32 x3), UV (f32 x2 split), Normal (f32 x3), Color (f32 x4). Interleaved layout.
- **GPUMeshBuffers**: Index buffer, vertex buffer, vertex device address (for shader indexing).
- **GPUDrawPushConstants**: World matrix (4x4) + vertex buffer device address. Push constant data per draw.
- **GPUSceneData**: View, proj, view_proj matrices; ambient color; sunlight direction/color. Uniform buffer data.

## Notable Patterns
- **repr(C) for FFI**: All structs layout-compatible with GPU expectations
- **Default derive**: Vertex and GPUSceneData derive Default for zero-initialization
- **Device address storage**: vertex_buffer_address enables shader buffer-device-address extension

## Unsafe Blocks
None; pure data definitions.

## Dependencies
- resources (AllocatedBuffer type reuse)
- ash::vk (vk::DeviceAddress type)
