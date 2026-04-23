# Module: swapchain.rs

## Purpose
Swapchain creation, recreation, and format selection. Handles swapchain lifecycle, image view creation, and semaphore allocation per image.

## Key Types
- **Swapchain**: Holds swapchain handle, images, image views, semaphores (one per image), function pointers, properties
- **SwapchainProperties**: Format (KHR), present mode (KHR), extent (2D)
- **SwapchainSupportDetails**: Capabilities, formats, present modes for device suitability check

## Notable Patterns
- **Create vs. recreate**: create() passes null old_swapchain; recreate() passes previous handle for retirement
- **Format selection**: Prefers B8G8R8A8_UNORM/SRGB_NONLINEAR; falls back to first available
- **Present mode selection**: Prefers FIFO (vsync); falls back to IMMEDIATE
- **Semaphore per image**: Each swapchain image gets its own semaphore for synchronization

## Unsafe Blocks
- Lines 86, 87, 104: create_swapchain, get_swapchain_images, create_image_view - safe per ash

## Dependencies
- context (VkContext, physical device, surface)
- sync (Semaphore)
- ash (Vulkan swapchain extension)
- itertools (collect_vec)
