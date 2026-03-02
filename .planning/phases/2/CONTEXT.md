---
phase: 02-handle-safety
requirements: [CRASH-04, CRASH-05]
---

# Phase 2: Handle Safety — Context

## Phase Goal

Remove `Clone` from Vulkan handle wrappers (`AllocatedBuffer`, `FrameData`) to make double-free hazards a compile error rather than latent runtime risk.

## Why This Matters

Both `AllocatedBuffer` and `FrameData` wrap Vulkan GPU resource handles and both derive `Clone`. Cloning either struct duplicates the raw handle values without duplicating the underlying GPU resources. If a clone is later destroyed (or drops), the `destroy` method is called on the same handle a second time — a double-free that corrupts the allocator and triggers validation errors or crashes.

Neither type is actually `.clone()`d anywhere in the codebase today, so the fix is low-risk: removing the `Clone` derive is a pure subtraction with no call-site updates required.

## Requirements

### CRASH-04
`FrameData` derives `Clone` (renderer.rs:1316). Cloning duplicates `vk::CommandPool`, `vk::Semaphore`, `vk::Fence` handles without duplicating the underlying GPU resources. The `destroy` method would then destroy the same handles twice. **Remove the `Clone` derive.**

### CRASH-05
`AllocatedBuffer` derives `Clone` (renderer.rs:1407). Cloning duplicates the `vk::Buffer` handle and `vk_mem::Allocation`. `GPUMeshBuffers` (primitives.rs:27) contains `AllocatedBuffer` but does NOT currently derive `Clone` — no cascade needed there. `FrameData` contains `AllocatedBuffer` and is addressed by CRASH-04. **Remove `Clone` from `AllocatedBuffer`.**

## Key Files

| File | Relevance |
|------|-----------|
| `crates/engine/src/renderer.rs:1316` | `FrameData` — `#[derive(Clone)]` to remove |
| `crates/engine/src/renderer.rs:1407` | `AllocatedBuffer` — `#[derive(Debug, Clone)]` → `#[derive(Debug)]` |
| `crates/engine/src/primitives.rs:27` | `GPUMeshBuffers` — already `#[derive(Debug)]` only, no change needed |

## Current Code State

```rust
// renderer.rs:1316
// maybe no clone?
#[derive(Clone)]
pub struct FrameData {
    pub command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    acquire_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    descriptors: GrowableAllocator,
    scene_buffer: AllocatedBuffer,
}

// renderer.rs:1407
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    info: vk_mem::AllocationInfo,
}

// primitives.rs:27 — already clean
#[derive(Debug)]
#[allow(dead_code)]
pub struct GPUMeshBuffers {
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}
```

## Scope Constraints

- **Only** remove `Clone` derives — no other refactoring
- Do NOT remove `Clone` from other types (`QueueData`, `ComputePushConstants`, `ComputeEffect`, etc.) — those contain only primitive/Copy values and are safe to clone
- `ImmediateSubmitData` derives `Copy + Clone` and contains Vulkan handles — it is **out of scope** for this phase (it's used as a short-lived local, not stored persistently)
- Do NOT touch `GrowableAllocator` — it is not defined in scope for this phase

## Decisions

| Decision | Rationale |
|----------|-----------|
| GPUMeshBuffers needs no change | Already `#[derive(Debug)]` only — cascade already resolved |
| ImmediateSubmitData out of scope | Short-lived local usage; different risk profile |
| No `.clone()` call-site updates needed | Neither type is cloned anywhere in the codebase |

## Success Criteria

1. `AllocatedBuffer` does not derive `Clone` — `grep "derive.*Clone" crates/engine/src/renderer.rs` does not match the `AllocatedBuffer` definition
2. `FrameData` does not derive `Clone` — same grep does not match the `FrameData` definition
3. `cargo build` exits 0 with no new errors
4. No call sites use `.clone()` on `AllocatedBuffer` or `FrameData` (already true, but verify)
