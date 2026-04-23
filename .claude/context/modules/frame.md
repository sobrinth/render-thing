# Module: frame.rs

## Purpose
Frame submission pipeline. Implements VulkanRenderer::draw() and companion functions (draw_background, draw_geometry, copy_image_to_image). Orchestrates per-frame rendering: command recording, synchronization, image transitions, and present.

## Key Types
- **FrameData**: Stores per-frame GPU resources (command pool, fence, semaphores, descriptor allocator, scene buffer). Dropped automatically, cleaning up pools.
- Impl block on VulkanRenderer with draw() as main entry point

## Notable Patterns
- **State machine flow**: Resize → acquire → sync fence → reset/begin → compute pass → geometry pass → copy → UI render → submit → present
- **Image layout choreography**: Transitions (UNDEFINED→GENERAL→COLOR_ATTACHMENT→TRANSFER_SRC→TRANSFER_DST→PRESENT) managed inline
- **Frame index modulo**: `frame_index = frame_number % FRAME_OVERLAP` selects active frame's command buffer
- **Immediate returns on error**: OUT_OF_DATE_KHR sets resize flag; misses present in that frame

## Unsafe Blocks
- Line 95: CommandBuffer::wrap() - requires fence wait guarantee (ensured above at line 44)
- Line 317: Push constants cast to byte slice - static lifetime, repr(C) struct, safe cast
- Line 486-491: std::ptr::copy_nonoverlapping for scene data - short-lived mapped pointer, paired with lifetime scope

## Dependencies
- command_buffer (CommandBuffer state machine, transition_image helper, ImmediateSubmitData)
- descriptor (DescriptorWriter)
- ui (before_frame, render, after_frame)
- resources (AllocatedBuffer)
- primitives (GPUDrawPushConstants, GPUSceneData)
