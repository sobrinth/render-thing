# vulkan-rust

## What This Is

A Vulkan 1.3 renderer written in Rust using `ash`. Implements a rendering loop with compute background effects, a glTF mesh draw pass, and an egui debug UI. The project uses dynamic rendering (no render passes), synchronization2, buffer device address, and VMA for GPU memory. Currently in early development — core rendering works but has several correctness bugs and fragile areas discovered during codebase audit.

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

### Active

**Crash fixes:**
- [ ] Fix `active_mesh: 2` initialization — default to `0` to prevent out-of-bounds panic on models with <3 meshes
- [ ] Fix glTF index reading — handle U32 (and U8) indices, not just U16; currently silently draws 0 triangles for most meshes
- [ ] Fix glTF UV channel — read from channel 0, not channel 1; currently produces wrong UVs on virtually all assets
- [ ] Remove `Clone` from `FrameData` — duplicating Vulkan handles without duplicating GPU resources is a latent double-free
- [ ] Remove `Clone` from `AllocatedBuffer` — same double-free risk; used in GPUMeshBuffers and FrameData
- [ ] Investigate and fix UI texture free timing — soundness concern flagged with `?` comments in ui.rs:193

**Low-hanging fruit:**
- [ ] Fix effect slider range — derive `0..=background_effects.len()-1` instead of hardcoded `0..=2`
- [ ] Log glTF load errors — `gltf::import(...).ok()?` silently discards errors; log before discarding
- [ ] Replace `mem::transmute` for push constants — use `bytemuck::bytes_of()` for safe, checked byte casting
- [ ] Fix frame sleep integer division — `1000/60 = 16` (wrong); remove sleep and rely on present mode, or use a proper frame timer
- [ ] Update dependencies — run `cargo update`, check for new major versions of ash, winit, vk-mem, egui, egui-ash-renderer

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

Key files:
- `crates/engine/src/renderer.rs` — monolithic 1452-line renderer; most bugs live here
- `crates/engine/src/meshes.rs` — glTF loading; UV channel and index type bugs
- `crates/engine/src/ui.rs` — egui integration; texture free timing concern
- `crates/application/src/main.rs` — application entry point; frame sleep bug

## Constraints

- **Tech stack**: Rust nightly (pinned), Vulkan 1.3, ash — no changes to core rendering API
- **Scope**: Bug fixes and safe refactors only — no architectural changes this milestone
- **Safety**: All unsafe blocks that are structurally required by Vulkan FFI are acceptable; only reduce unsafe where a safe abstraction (bytemuck) exists

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Leave `#![allow(dead_code)]` | Too many items to audit cleanly during a bug-fix pass | — Pending |
| Use bytemuck for push constants | Provides compile-time Pod/Zeroable checks vs silent transmute | — Pending |
| Default active_mesh to 0 | Safest default; UI slider still allows selecting any mesh | — Pending |

---
*Last updated: 2026-03-02 after initialization*
