# Phase 2: Handle Safety - Research

**Researched:** 2026-03-02
**Domain:** Rust ownership semantics, Vulkan handle wrappers, vk-mem 0.5.0
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

| Decision | Rationale |
|----------|-----------|
| GPUMeshBuffers needs no change | Already `#[derive(Debug)]` only — cascade already resolved |
| ImmediateSubmitData out of scope | Short-lived local usage; different risk profile |
| No `.clone()` call-site updates needed | Neither type is cloned anywhere in the codebase |

### Claude's Discretion

None specified — the entire change is fully prescribed.

### Deferred Ideas (OUT OF SCOPE)

- Any refactoring beyond removing the two `Clone` derives
- Removing `Clone` from other types (`QueueData`, `ComputePushConstants`, `ComputeEffect`, `ImmediateSubmitData`)
- Touching `GrowableAllocator`
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CRASH-04 | Remove `Clone` from `FrameData` (renderer.rs:1316) — cloning duplicates `vk::CommandPool`, `vk::Semaphore`, `vk::Fence` without duplicating GPU resources, enabling double-free in `destroy` | Confirmed: `#[derive(Clone)]` is on line 1316; no `.clone()` call sites exist; GrowableAllocator is NOT relied on for FrameData's Clone impl after the derive is removed |
| CRASH-05 | Remove `Clone` from `AllocatedBuffer` (renderer.rs:1407) — cloning duplicates `vk::Buffer` + `vk_mem::Allocation` without allocating new GPU memory, enabling double-free in `destroy` | Confirmed: `#[derive(Debug, Clone)]` on line 1407; vk_mem::Allocation implements Clone/Copy (it's a pointer wrapper) but AllocatedBuffer must not — cascade to GPUMeshBuffers is a non-issue (already Debug-only) |
</phase_requirements>

---

## Summary

Phase 2 is a targeted two-line removal. Both `AllocatedBuffer` and `FrameData` carry `Clone` derives that create a latent double-free hazard: cloning either struct copies raw handle values (integers/pointers) without creating new GPU resources, so a second `destroy` call on the clone would free the same memory twice. The fix is to remove the `Clone` derives.

A full audit confirms no `.clone()` call sites exist for either type anywhere in the codebase. The only `.clone()` calls in `.rs` files touch `egui_ctx` (an egui type), `allocator` and `device` (Arc-wrapped types in ui.rs), `textures_delta.free` (a Vec), and a path in `build.rs` — none of which are `AllocatedBuffer`, `FrameData`, or `GPUMeshBuffers`.

The cascade analysis is clean: `GPUMeshBuffers` (primitives.rs:27) already derives only `Debug` — no change needed there. `FrameData` contains `AllocatedBuffer` as a field, but since `FrameData`'s own `Clone` is being removed, the compiler will not attempt to derive or verify `AllocatedBuffer: Clone` for `FrameData`. There is no cascade problem.

**Primary recommendation:** Remove two derive attributes. No call-site changes. Build should be green immediately after.

---

## Standard Stack

Not applicable — this phase performs no new library integration. The existing stack is:

| Library | Version | Relevant Type |
|---------|---------|---------------|
| ash | 0.38 | `vk::Buffer`, `vk::CommandPool`, `vk::Semaphore`, `vk::Fence` — all `Copy + Clone` (u64 handles) |
| vk-mem | 0.5.0 | `vk_mem::Allocation` — `#[derive(Clone, Copy)]` (a thin `*mut VmaAllocation` pointer wrapper) |

---

## Architecture Patterns

### Why `Clone` on handle wrappers is dangerous

`vk_mem::Allocation` is `Clone + Copy` because it is defined as:

```rust
// Source: ~/.cargo/registry/src/index.crates.io-.../vk-mem-0.5.0/src/lib.rs:43-44
#[derive(Clone, Copy, Debug)]
pub struct Allocation(ffi::VmaAllocation);  // VmaAllocation = *mut opaque
```

It is a raw pointer wrapped in a newtype. Copying it gives you two Rust values pointing to the same VMA-internal allocation record. The vk-mem allocator tracks allocations by pointer identity. If two `AllocatedBuffer` values both hold the same `Allocation`, and both call `allocator.destroy_buffer(self.buffer, &mut self.allocation)`, VMA processes a double-free — undefined behavior in the allocator's internal state.

The same logic applies to `FrameData`: `vk::CommandPool`, `vk::Semaphore`, `vk::Fence` are all 64-bit opaque handles (`u64` newtypes in ash). Ash provides `Clone + Copy` on all handle types. Cloning `FrameData` produces two structs pointing at the same Vulkan objects. `destroy` on either copy calls `device.destroy_command_pool`, `destroy_semaphore`, `destroy_fence` on the same handles twice.

### Correct ownership model for GPU resource wrappers

GPU resource wrappers should model **unique ownership** — exactly one Rust value owns the right to destroy the resource:

```rust
// After fix: no Clone derive — the struct is the sole owner
pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    info: vk_mem::AllocationInfo,
}

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    acquire_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    descriptors: GrowableAllocator,
    scene_buffer: AllocatedBuffer,
}
```

This makes the type system enforce the invariant: you cannot accidentally create a second owner.

### Anti-Patterns to Avoid

- **Deriving `Clone` on any struct that owns a Vulkan handle or VMA allocation.** The Rust `Clone` derive is semantic — it states "I know how to make a second equal value." For GPU resource owners, there is no valid "second equal value."
- **Confusing handle-level Copy with wrapper-level Clone.** `vk::Buffer` being `Copy` is correct (it's just a number). `AllocatedBuffer` being `Clone` is wrong (it implies you have two owners of one GPU buffer allocation).

---

## Don't Hand-Roll

Not applicable to this phase — no new solutions being built.

---

## Common Pitfalls

### Pitfall 1: Cascade concern from GrowableAllocator

**What goes wrong:** Developer sees `GrowableAllocator` in `FrameData` and worries that `GrowableAllocator: Clone` (it does derive `Clone`, per descriptor.rs:103) means something breaks.

**Why it doesn't apply here:** `#[derive(Clone)]` on a struct auto-implements `Clone` by cloning each field. When you *remove* `#[derive(Clone)]` from `FrameData`, the compiler stops trying to implement `Clone` for `FrameData` entirely. It does not care whether fields are `Clone` or not. `GrowableAllocator` having `Clone` is irrelevant.

**How to avoid:** Read the removal direction — we are removing Clone from `FrameData`, not adding it. The field constraints only matter when adding a derive, not removing one.

### Pitfall 2: Assuming the compiler might miss a Clone use site

**What goes wrong:** Concern that a `.clone()` call somewhere might silently break or go undetected.

**Why it's not a concern:** If any code calls `.clone()` on an `AllocatedBuffer` or `FrameData` value after the derive is removed, the Rust compiler will emit `error[E0599]: no method named 'clone' found for struct 'AllocatedBuffer'`. This is a hard compile error. There is no silent path.

**Verified:** A grep of all `.clone()` calls across every `.rs` file in the workspace found zero calls on `AllocatedBuffer`, `FrameData`, or `GPUMeshBuffers`. The only `.clone()` calls in the codebase are on egui, Arc-wrapped types, and a Vec. The build is currently green.

**Warning signs:** If `cargo build` fails after the change, the error message will name the exact location.

### Pitfall 3: AllocatedImage is NOT in scope

**What goes wrong:** Developer notices `AllocatedImage` (renderer.rs:1349) also contains `vk_mem::Allocation` and wonders if it should be fixed too.

**Why it's out of scope:** CONTEXT.md explicitly scopes this phase to `AllocatedBuffer` and `FrameData` only. `AllocatedImage` does not derive `Clone` — it has no derive attributes at all. It is already safe.

### Pitfall 4: vk_mem::AllocationInfo clone behavior

**What goes wrong:** `AllocatedBuffer` also holds `vk_mem::AllocationInfo`. Is that safe to clone?

**Why it's a non-issue:** `AllocationInfo` is a struct of primitive fields (mapped_memory pointer, offsets, sizes, etc.). It is descriptive metadata, not an ownership handle. Cloning `AllocationInfo` is harmless — but the point is moot because we are removing Clone from `AllocatedBuffer` entirely.

---

## Exact Changes Required

### Change 1: Remove Clone from AllocatedBuffer

**File:** `H:/vulkan-rust/crates/engine/src/renderer.rs`
**Line:** 1407

```rust
// BEFORE
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AllocatedBuffer {

// AFTER
#[derive(Debug)]
#[allow(dead_code)]
pub struct AllocatedBuffer {
```

### Change 2: Remove Clone from FrameData

**File:** `H:/vulkan-rust/crates/engine/src/renderer.rs`
**Line:** 1315-1316 (comment + derive)

```rust
// BEFORE
// maybe no clone?
#[derive(Clone)]
pub struct FrameData {

// AFTER
#[derive(Debug)]
pub struct FrameData {
```

Note: The comment "// maybe no clone?" is now resolved — it can be removed. Adding `#[derive(Debug)]` to `FrameData` is optional but consistent with the rest of the file. The minimum change is just removing `#[derive(Clone)]`.

---

## Code Examples

### Verifying no .clone() call sites (reproducible command)

```bash
grep -rn "\.clone()" crates/engine/src/ --include="*.rs"
# Output: only ui.rs lines 40-41 (Arc types), line 86 (egui_ctx), line 164 (Vec)
# None touch AllocatedBuffer, FrameData, or GPUMeshBuffers
```

### Verifying the fix post-change

```bash
# Success criteria from CONTEXT.md
grep "derive.*Clone" crates/engine/src/renderer.rs
# AllocatedBuffer and FrameData lines must NOT appear in output
# QueueData, ImmediateSubmitData, ComputePushConstants, ComputeEffect lines WILL appear — that's correct

cargo build
# Must exit 0 with no new errors
```

---

## State of the Art

| Old Approach | Current Approach | Status |
|--------------|------------------|--------|
| `#[derive(Clone)]` on GPU handle wrappers | No `Clone` derive — explicit `fn clone()` or no Clone at all | Phase 2 implements this |
| "Clone is free / harmless" mental model | Clone on handle types = aliased ownership = double-free risk | Compile-time enforcement |

---

## Open Questions

None. All research questions answered with HIGH confidence from direct source inspection.

---

## Sources

### Primary (HIGH confidence)

- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/vk-mem-0.5.0/src/lib.rs:43-44` — `vk_mem::Allocation` is `#[derive(Clone, Copy, Debug)]` wrapping `ffi::VmaAllocation` (a raw pointer)
- `H:/vulkan-rust/crates/engine/src/renderer.rs:1315-1324` — `FrameData` definition with `#[derive(Clone)]`
- `H:/vulkan-rust/crates/engine/src/renderer.rs:1407-1413` — `AllocatedBuffer` definition with `#[derive(Debug, Clone)]`
- `H:/vulkan-rust/crates/engine/src/primitives.rs:25-31` — `GPUMeshBuffers` already `#[derive(Debug)]` only
- `H:/vulkan-rust/crates/engine/src/descriptor.rs:103-109` — `GrowableAllocator` derives `Clone` (present but irrelevant to FrameData removal)
- Grep of all `.rs` files for `.clone()` — zero hits on target types

### Secondary (MEDIUM confidence)

- ash 0.38 handle types (`vk::Buffer`, `vk::CommandPool`, etc.) — all defined as `#[derive(Clone, Copy)]` u64 newtypes; this is consistent with Vulkan spec's handle semantics

---

## Metadata

**Confidence breakdown:**
- Exact changes needed: HIGH — read directly from source
- No call-site updates needed: HIGH — confirmed by grep across entire workspace
- GrowableAllocator non-issue: HIGH — confirmed by reading descriptor.rs and understanding Rust derive semantics
- vk_mem::Allocation Clone behavior: HIGH — read directly from crate source in cargo registry

**Research date:** 2026-03-02
**Valid until:** Stable until vk-mem version changes or new code is added that calls `.clone()` on these types
