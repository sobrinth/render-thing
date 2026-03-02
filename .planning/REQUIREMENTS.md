# Requirements — vulkan-rust Cleanup Milestone

**Milestone goal:** Fix correctness bugs and low-hanging-fruit issues identified in the 2026-03-02 codebase audit. No architectural changes.

---

## v1 Requirements

### Crash Fixes

- [ ] **CRASH-01**: `active_mesh` is initialized to `2` unconditionally. If the loaded model has fewer than 3 meshes, `meshes[self.active_mesh]` panics. Default to `0` (or clamp to `meshes.len().saturating_sub(1)`) at initialization.
  - Files: `crates/engine/src/renderer.rs:154`

- [ ] **CRASH-02**: glTF index reader only handles `ReadIndices::U16`. Files with U32 indices (common for meshes >65535 vertices) or U8 indices are silently skipped, producing an empty index buffer and drawing 0 triangles. Handle all three index types: U8, U16, U32.
  - Files: `crates/engine/src/meshes.rs:56–62`

- [ ] **CRASH-03**: UV coordinates are read from `read_tex_coords(1)` (channel 1) instead of the standard `read_tex_coords(0)` (channel 0). Virtually all glTF files store primary UVs in channel 0. Reading channel 1 produces empty or wrong UVs for most assets.
  - Files: `crates/engine/src/meshes.rs:80`

- [ ] **CRASH-04**: `FrameData` derives `Clone`. Cloning duplicates `vk::CommandPool`, `vk::Semaphore`, and `vk::Fence` handles without duplicating the underlying GPU resources. The `destroy` method would then attempt to destroy the same handles twice. Remove the `Clone` derive.
  - Files: `crates/engine/src/renderer.rs:1316`

- [ ] **CRASH-05**: `AllocatedBuffer` derives `Clone`. Cloning duplicates the `vk::Buffer` handle and `vk_mem::Allocation`. Both `GPUMeshBuffers` and `FrameData` contain `AllocatedBuffer` and are also `Clone`. Remove `Clone` from `AllocatedBuffer` (and cascade: `GPUMeshBuffers`, `FrameData`).
  - Files: `crates/engine/src/renderer.rs:1407`, `crates/engine/src/primitives.rs`

- [ ] **CRASH-06**: UI texture free timing is marked unsound with `// ? soundness with multiple frames in flight` comments. With FRAME_OVERLAP=2, a texture freed in frame N may still be in use by GPU work from frame N-1. Investigate whether the current deferred-free strategy is actually safe, and fix or document with a clear explanation.
  - Files: `crates/engine/src/ui.rs:193–201`

### Low-Hanging Fruit

- [ ] **CLEAN-01**: Effect selector slider is hardcoded to `0..=2`. Adding or removing a compute effect requires manually updating this range. Change to `0..=(self.background_effects.len().saturating_sub(1)) as f32` (or equivalent).
  - Files: `crates/engine/src/ui.rs:107`

- [ ] **CLEAN-02**: `gltf::import(path).ok()?` silently discards load errors. If the asset is missing or corrupt, the renderer initializes with `meshes: None` and renders nothing with no log output. Log the error before discarding.
  - Files: `crates/engine/src/meshes.rs:36`

- [ ] **CLEAN-03**: Push constants are written using `mem::transmute` to convert typed structs to byte arrays. If struct size or alignment assumptions are wrong, the transmute silently sends incorrect bytes to the GPU. Replace with `bytemuck::bytes_of()`, which provides compile-time `Pod`/`Zeroable` checks.
  - Files: `crates/engine/src/renderer.rs:520`, `crates/engine/src/renderer.rs:628`

- [ ] **CLEAN-04**: The frame rate limiter uses `std::thread::sleep(Duration::from_millis(1000 / 60))`. Integer division gives 16ms (~62.5fps with uneven timing). The inline comment acknowledges this is wrong. Remove the sleep and rely on `PresentModeKHR::FIFO` for vsync pacing.
  - Files: `crates/application/src/main.rs:94`

- [ ] **CLEAN-05**: Dependencies may have newer versions available. Run `cargo update` to pick up patch-level updates within current version constraints. Check for new major versions of key dependencies (ash, winit, vk-mem, egui, egui-ash-renderer) and update where upgrades are non-breaking or worth the migration.
  - Files: `Cargo.toml`, `Cargo.lock`

---

## v2 Requirements (Deferred)

- Global `#![allow(dead_code)]` audit — remove crate-level suppression and clean up or document each flagged item
- Precise image barrier stage masks — use specific stages instead of `ALL_COMMANDS`
- Pipeline cache — persist `vk::PipelineCache` to disk between runs
- `VkContext` decoupling from winit (accept raw window handles)

---

## Out of Scope

- Pipeline abstraction refactor (`Pipeline` struct) — architectural, not a bug fix
- Scene graph / camera abstraction — feature work
- Texture / material support — feature work
- Test suite — low priority for a learning/development renderer
- Shader compiler toolchain improvements — separate concern

---

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| CRASH-01 | Phase 1 | Pending |
| CRASH-02 | Phase 1 | Pending |
| CRASH-03 | Phase 1 | Pending |
| CRASH-04 | Phase 2 | Pending |
| CRASH-05 | Phase 2 | Pending |
| CRASH-06 | Phase 3 | Pending |
| CLEAN-01 | Phase 3 | Pending |
| CLEAN-02 | Phase 1 | Pending |
| CLEAN-03 | Phase 4 | Pending |
| CLEAN-04 | Phase 3 | Pending |
| CLEAN-05 | Phase 5 | Pending |
