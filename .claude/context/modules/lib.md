# Module: lib.rs

## Purpose
Public API entry point for the engine crate. Exports `Engine` struct which wraps `VulkanRenderer` and provides a clean interface to the application layer.

## Key Types
- **Engine**: Facade struct containing a single `VulkanRenderer` field, implements public methods

## Notable Patterns
- **Facade pattern**: Engine is deliberately thin; all heavy lifting delegated to VulkanRenderer
- **Input abstraction**: Engine re-exports `input` module (ElementState, Key, MouseButton) for type-safe event handling
- **Public/private split**: Engine exposes only 6 methods (initialize, egui_context, draw, resize, on_key_press, on_mouse_event, on_mouse_button_event, stop); everything else is internal

## Unsafe Blocks
None at this level.

## Dependencies
- renderer (VulkanRenderer, FRAME_OVERLAP constant)
- input (reexported)
- nalgebra_glm as glm (prelude import)
- raw_window_handle (trait types for window integration)
