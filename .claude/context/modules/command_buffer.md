# Module: command_buffer.rs

## Purpose
Type-state command buffer pattern for compile-time state machine enforcement. Prevents illegal state transitions (e.g., resetting while submitted). Includes ImmediateSubmitData for one-off GPU work.

## Key Types
- **CommandBuffer<S>**: Generic over sealed state trait (Initial, Recording, Executable, Submitted). PhantomData ensures zero-cost type distinction.
- **ImmediateSubmitData**: Encapsulates one-off command pool/buffer/fence for synchronous GPU operations (uploads, transitions).
- Helper function **transition_image**: Image layout barrier construction with aspect mask selection (depth vs. color).

## Notable Patterns
- **Sealed trait pattern**: CmdState trait in sealed module prevents external implementation
- **Typestate state machine**: wrap() → reset() → begin() → handle() [recording] → end() → into_submitted()
- **Zero-cost abstraction**: All states fit in same vk::CommandBuffer + PhantomData<S>; no runtime overhead
- **Immediate submit closure**: submit() takes FnOnce closure; closure receives CommandBuffer<Recording> guaranteeing only valid calls

## Unsafe Blocks
- Line 34: CommandBuffer::wrap() - caller guarantees fence waited, no aliasing, pool live (safety contract documented)
- Line 42, 58, 72: Device API calls (reset, begin, end) - safe per ash; typestate ensures preconditions met
- Line 135, 191: Queue submit, cmd_pipeline_barrier2 - safe; handles valid (not aliased)

## Dependencies
- context (QueueData for queue reference)
- sync (Fence, Semaphore)
- ash::vk (Vulkan command and barrier types)
