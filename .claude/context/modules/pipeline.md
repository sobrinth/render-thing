# Module: pipeline.rs

## Purpose
Graphics pipeline construction via builder pattern. Encapsulates Vulkan pipeline creation state and provides configuration methods (shaders, topology, blending, depth test, etc.). Also defines compute effect types.

## Key Types
- **PipelineBuilder**: Accumulates shader stages, rasterization, color blend, depth stencil settings. Builds final vk::Pipeline on demand.
- **Pipeline**: Wraps vk::Pipeline + vk::PipelineLayout. Destructor cleans up both (or null layout for owned-elsewhere layouts).
- **PipelineLayout**: Wrapper for vk::PipelineLayout only (layout-only resource, no pipeline).
- **ComputeEffect**: Holds shader name, Pipeline handle, push constant data (4 x [f32; 4]).
- **ComputePushConstants**: repr(C) push constant block (16 floats total).

## Notable Patterns
- **Builder accumulation**: Methods return &mut Self for chaining; build() consumes and creates pipeline
- **VK_FALSE/VK_TRUE constants**: Bool32 type requires explicit u32 constants (lines 40-41)
- **Render info lifetime management**: render_info.p_color_attachment_formats points to color_attachment_format field (line 183) - lifetime managed by builder scope
- **Pipeline layout ownership**: Pipeline Drop only destroys layout if not null; effect_pipeline_layout owns shared layouts

## Unsafe Blocks
None in this module; all Vulkan calls wrapped in safe ash APIs.

## Dependencies
- ash::vk (Vulkan types)
- No renderer/frame/resource dependencies
