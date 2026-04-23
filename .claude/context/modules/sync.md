# Module: sync.rs

## Purpose
Lightweight synchronization primitives (Fence, Semaphore) with RAII cleanup. Wraps vk handles with device ownership for automatic destruction.

## Key Types
- **Fence**: Holds vk::Fence and device. wait() returns bool (false on timeout). reset() unsignals.
- **Semaphore**: Holds vk::Semaphore and device. No direct wait (Vulkan doesn't expose that).

## Notable Patterns
- **RAII cleanup**: Drop impls call vk::destroy_* automatically
- **new_signaled()**: Creates fence in pre-signaled state (useful for first frame)
- **Timeout in nanoseconds**: wait() takes timeout_ns; ONE_SECOND = 1_000_000_000
- **Panic on error**: wait() panics on failure (not just timeout), ensuring invariants

## Unsafe Blocks
- Lines 19, 61: create_fence, create_semaphore - safe per ash
- Line 34: wait_for_fences - safe; timeout errors handled gracefully (not panicked)

## Dependencies
- ash::vk (Vulkan synchronization types)
