# Module: descriptor.rs

## Purpose
Descriptor set allocation and layout management. Implements both simple Allocator (single pool) and GrowableAllocator (expanding pools on exhaustion). Includes DescriptorWriter for descriptor updates.

## Key Types
- **DescriptorSetLayout**: Wraps vk::DescriptorSetLayout with device for cleanup.
- **LayoutBuilder**: Accumulates bindings, builds layout with shader stage flags.
- **Allocator**: Simple fixed-pool allocator (single vk::DescriptorPool).
- **GrowableAllocator**: Maintains ready/full pool lists, auto-expands when pool exhausted. Clears pools per-frame.
- **DescriptorWriter**: Accumulates image/buffer writes, batches update_descriptor_sets in one call.
- **PoolSizeRatio**: Configuration (descriptor type + ratio) for pool sizing.

## Notable Patterns
- **Ratio-based pool sizing**: Pools sized as max_sets * ratio per descriptor type; avoids guessing
- **Pool reset not destroy**: clear_pools() resets (recycles) full pools instead of destroying; destroy_pools() called on drop
- **Growable expansion**: sets_per_pool *= 1.5 (capped at 4096) when creating new pool; adapts to demand
- **Transient descriptor capture**: DescriptorWriter stores both info pointers and writes; update_set() finalizes writes with dst_set

## Unsafe Blocks
- Line 107, 185, 225: allocate_descriptor_sets, reset_descriptor_pool, create_descriptor_pool - safe per ash wrapper
- Line 304: update_descriptor_sets - safe; info pointers valid for lifetime of writes

## Dependencies
- ash::vk (Vulkan descriptor types)
- itertools (Itertools trait for collect_vec)
