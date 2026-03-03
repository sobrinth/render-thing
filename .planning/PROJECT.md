# vulkan-rust

## What This Is

A Vulkan 1.3 renderer written in Rust using `ash`. Implements a rendering loop with compute background effects, a glTF mesh draw pass, and an egui debug UI. The project uses dynamic rendering (no render passes), synchronization2, buffer device address, and VMA for GPU memory. Critical correctness bugs (crash on mesh count, wrong index types, wrong UV channel, double-free-risk Clone derives) fixed in the cleanup milestone.

## Core Value

A correct, non-crashing foundation for experimenting with Vulkan rendering techniques.

## Requirements

### Validated

- ✓ Vulkan 1.3 context initialization with validation layers — existing
- ✓ Swapchain creation and recreation on window resize — existing
- ✓ Double-buffered frame-in-flight rendering (FRAME_OVERLAP = 2) — existing
- ✓ Compute background effects with per-effect push constants — existing
- ✓ glTF mesh loading and draw pass with dynamic rendering — existing
- ✓ egui debug UI integration with Vulkan dynamic rendering — existing
- ✓ Growable descriptor allocator — existing
- ✓ GPU memory allocation via vk_mem — existing
- ✓ Shader compilation from GLSL via build.rs — existing
- ✓ active_mesh defaults to 0 — prevents panic on models with <3 meshes — cleanup-2026-03
- ✓ glTF index reading handles U8/U16/U32 via into_u32() — no silent 0-triangle draws — cleanup-2026-03
- ✓ UV coordinates read from TEXCOORD_0 and written to Vertex fields — cleanup-2026-03
- ✓ glTF import errors logged before silent discard — cleanup-2026-03
- ✓ Clone removed from AllocatedBuffer, FrameData, GPUMeshBuffers — double-free risk eliminated at compile time — cleanup-2026-03

### Active

- [ ] Investigate and fix UI texture free timing — soundness concern flagged with `?` comments in ui.rs:193 (CRASH-06)
- [ ] Fix effect slider range — derive `0..=background_effects.len()-1` instead of hardcoded `0..=2` (CLEAN-01)
- [ ] Replace `mem::transmute` for push constants — use `bytemuck::bytes_of()` for safe, checked byte casting (CLEAN-03)
- [ ] Fix frame sleep integer division — remove sleep and rely on `PresentModeKHR::FIFO` for vsync pacing (CLEAN-04)
- [ ] Update dependencies — run `cargo update`, check for new major versions of ash, winit, vk-mem, egui, egui-ash-renderer (CLEAN-05)

### Out of Scope

- Global `#![allow(dead_code)]` audit — deferred; too many items to address cleanly alongside bug fixes
- Pipeline abstraction refactor — tech debt, not a crash risk
- `VkContext` / winit decoupling — architectural change, out of scope for this pass
- Scene graph or camera abstraction — feature work, not cleanup
- Texture / material support — feature work
- Precise image barrier stage masks — performance optimization, not a crash
- Pipeline cache — performance optimization, not a crash

## Context

Brownfield Rust/Vulkan codebase following the vkguide.dev pattern. Codebase map produced 2026-03-02.

Cleanup milestone (cleanup-2026-03) shipped 2026-03-03. 6 of 11 requirements completed; 5 low-risk items deferred to next pass.

Key files:
- `crates/engine/src/renderer.rs` — monolithic ~1450-line renderer
- `crates/engine/src/meshes.rs` — glTF loading (correctness fixes applied)
- `crates/engine/src/ui.rs` — egui integration; texture free timing still unresolved
- `crates/application/src/main.rs` — application entry point; frame sleep still present

## Constraints

- **Tech stack**: Rust nightly (pinned), Vulkan 1.3, ash — no changes to core rendering API
- **Scope**: Bug fixes and safe refactors only — no architectural changes this milestone
- **Safety**: All unsafe blocks structurally required by Vulkan FFI are acceptable; only reduce unsafe where a safe abstraction (bytemuck) exists

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Leave `#![allow(dead_code)]` | Too many items to audit cleanly during a bug-fix pass | — Deferred to next milestone |
| Use bytemuck for push constants | Provides compile-time Pod/Zeroable checks vs silent transmute | — Pending (CLEAN-03 deferred) |
| Default active_mesh to 0 | Safest default; UI slider still allows selecting any mesh | ✓ Good — shipped in cleanup-2026-03 |
| Remove Clone from handle wrappers | Eliminates double-free at compile time without runtime cost | ✓ Good — shipped in cleanup-2026-03 |

---
*Last updated: 2026-03-03 after cleanup-2026-03 milestone*
