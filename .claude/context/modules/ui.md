# Module: ui.rs

## Purpose
egui integration layer. Manages egui-ash-renderer, handles before/after frame UI state, and integrates UI primitives into render loop with texture management.

## Key Types
- **UiContext**: Holds egui-ash Renderer, egui::Context, scale factor, pending texture frees
- **UiState**: Mutable references to active effect, mesh, render scale, effective resolution for UI bindings

## Notable Patterns
- **before_frame/render/after_frame split**: before_frame() builds UI, render() draws, after_frame() cleans up textures
- **Texture lifetime management**: Delta textures queued in before_frame(); freed in after_frame()
- **Slider-based controls**: UI exposes effect index, mesh index, render scale, push constants as sliders/drag values
- **Scale factor**: pixels_per_point from egui context passed to renderer for DPI-aware drawing
- **Soundness question (line 181): Comment notes potential issue with multiple frames in flight and texture cleanup timing

## Unsafe Blocks
None in this module; egui-ash-renderer API is safe.

## Dependencies
- context (QueueData for texture upload queue)
- frame (FrameData for command pool)
- meshes (MeshAsset names displayed)
- pipeline (ComputeEffect for UI state binding)
- swapchain (SwapchainProperties for format)
- egui/egui-ash-renderer (UI framework)
