# Module: context.rs

## Purpose
Vulkan context initialization and lifetime management. Handles instance creation, physical device selection, logical device creation, and surface setup. Central Vulkan initialization orchestrator.

## Key Types
- **VkContext**: Holds Entry, Instance, Device, Surface (KHR), PhysicalDevice, optional debug messenger
- **QueueData**: Graphics queue handle + family index pair
- **QueueFamilyIndices**: Internal struct for graphics/present family lookups

## Notable Patterns
- **Dedup queue family indices**: Graphics and present queue may be same family; deduped to avoid invalid device creation (lines 137-153)
- **Feature chaining**: PhysicalDeviceVulkan12Features and PhysicalDeviceVulkan13Features chained via push_next (lines 163-176)
- **Validation layer gate**: Debug layer only enabled in debug builds (cfg!(debug_assertions))
- **Device suitability predicate**: is_device_suitable() checks queue families, extensions, swapchain support, features in one place

## Unsafe Blocks
- Lines 30-45: Entry::load(), create_surface, enumerate_physical_devices - safe per ash docs
- Lines 30, 105, 178: FFI calls properly wrapped; errors propagated via Result
- Lines 113, 261: get_physical_device_* APIs - safe enumeration with bounds checking implicit

## Dependencies
- debug (check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger)
- swapchain (SwapchainSupportDetails for device suitability check)
- ash/ash-window (Vulkan bindings)
- raw_window_handle (display/window handle traits)
