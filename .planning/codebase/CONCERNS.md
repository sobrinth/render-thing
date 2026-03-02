# Codebase Concerns

**Analysis Date:** 2026-03-02

---

## Tech Debt

**Monolithic renderer (explicitly flagged in todos.md):**
- Issue: `VulkanRenderer` in `crates/engine/src/renderer.rs` is 1452 lines and handles initialization, per-frame draw logic, pipeline creation, image management, buffer uploads, swapchain recreation, and immediate submit — all in a single struct/impl block.
- Files: `crates/engine/src/renderer.rs`
- Impact: Changes to any one subsystem require navigating the entire file. Adding a second pipeline type requires understanding the full monolith. Hard to test parts in isolation.
- Fix approach: Extract `pipeline` abstraction (noted in `todos.md`), extract `DescriptorSet` management into a dedicated type, split initialization helpers into a separate module.

**Missing `pipeline` abstraction:**
- Issue: Pipeline creation is implemented as two ad-hoc functions (`initialize_effect_pipelines`, `initialize_mesh_pipeline`) directly on `VulkanRenderer`. There is no reusable `Pipeline` struct wrapping both `vk::Pipeline` and `vk::PipelineLayout`.
- Files: `crates/engine/src/renderer.rs` (lines ~1004–1124), `crates/engine/src/pipeline.rs`
- Impact: Every new render pass requires duplicating the layout/pipeline creation pattern. Pipeline lifecycle (create + destroy) is managed inconsistently.
- Fix approach: Create a `Pipeline { pipeline: vk::Pipeline, layout: vk::PipelineLayout }` struct in `crates/engine/src/pipeline.rs` with proper `Drop` impl. `PipelineBuilder` already exists and is a good foundation.

**`VkContext` coupled to `winit::Window`:**
- Issue: `VkContext::initialize` takes a `&Window` and is responsible for surface creation, tightly coupling the graphics context to the windowing library.
- Files: `crates/engine/src/context.rs` line 30 (`// TODO: db: Probably move reference to winit out of VkContext`)
- Impact: Prevents reuse of `VkContext` in headless or off-screen rendering scenarios. Makes testing context initialization without a window impossible.
- Fix approach: Accept raw `RawDisplayHandle` / `RawWindowHandle` instead of `&Window`, or split context creation into instance/device setup (window-free) and surface creation (window-aware).

**`DescriptorSet` abstractions not yet covering draw-image layout:**
- Issue: `todos.md` explicitly notes "DescriptorSets and associated layouts (drawimage and more)" as pending refactor work. The draw-image descriptor set and its layout are managed as loose fields on `VulkanRenderer` (`draw_image_descriptors`, `draw_image_descriptor_layout`) rather than through a unified abstraction.
- Files: `crates/engine/src/renderer.rs` (struct fields lines 41–43, init at ~956–989)
- Impact: As more passes are added (shadow maps, GBuffer, post-processing), this pattern will produce more loose fields on the already large renderer struct.
- Fix approach: Encapsulate `(vk::DescriptorSetLayout, vk::DescriptorSet)` pairs into a typed struct per pass resource.

---

## `unsafe` Usage

**Pervasive but necessary `unsafe` for Vulkan FFI:**
- Issue: `unsafe` blocks appear ~80+ times across the engine crate. This is unavoidable with the `ash` Vulkan bindings, which expose the raw Vulkan C API. All direct Vulkan calls (`cmd_*`, `create_*`, `destroy_*`, queue operations, memory mapping) require `unsafe`.
- Files: All files in `crates/engine/src/`
- Risk level: Medium — the `unsafe` is structurally required, but the high density makes it easy to miss a correctness requirement (e.g., correct image layout at a barrier, correct fence state before reset).
- Current mitigations: Validation layers enabled in debug builds (`crates/engine/src/debug.rs`), `vk_mem` allocator handles most memory safety concerns.

**`mem::transmute` for push constant data:**
- Issue: Push constants are written to the command buffer using `mem::transmute` to convert typed structs directly to `[u8; N]` byte arrays.
  - `crates/engine/src/renderer.rs:520` — `transmute::<ComputePushConstants, [u8; 64]>(active_background.data)`
  - `crates/engine/src/renderer.rs:628` — `transmute::<GPUDrawPushConstants, [u8; size_of::<GPUDrawPushConstants>()]>(push_constants)`
- Impact: If the struct size or alignment assumptions are wrong (e.g., padding is added), the transmute silently sends incorrect bytes to the GPU. Both structs are `#[repr(C)]` which is correct, but this is fragile as fields are added.
- Fix approach: Use `bytemuck` crate with `Pod` + `Zeroable` derives and `bytemuck::bytes_of()`, which provides a safe checked alternative.

**Raw pointer copy for GPU scene data:**
- Issue: Scene data is written to the mapped GPU buffer via `std::ptr::copy_nonoverlapping` with manual pointer casting.
  - `crates/engine/src/renderer.rs:653–658`
- Impact: No bounds checking. If `GPUSceneData` grows larger than the buffer allocation, this is undefined behavior. The buffer is sized with `size_of::<GPUSceneData>()` at allocation time, so currently safe, but fragile.
- Fix approach: Use `bytemuck` or a typed mapped buffer wrapper.

**`unsafe extern "system"` debug callback:**
- Issue: `vulkan_debug_callback` in `crates/engine/src/debug.rs:8` is an `unsafe extern "system"` function called by the Vulkan runtime. It dereferences a raw pointer to `vk::DebugUtilsMessengerCallbackDataEXT`.
- Impact: Pointer dereference without null check. Standard for Vulkan callbacks — the driver guarantees non-null — but worth noting.

---

## Hardcoded Values

**Draw image resolution hardcoded to 2560×1440:**
- Issue: The draw and depth render targets are created at a fixed `(2560, 1440)` resolution regardless of window size or display resolution.
  - `crates/engine/src/renderer.rs:74` — `Self::create_image(&context, &gpu_alloc, (2560, 1440), ...)`
  - `crates/engine/src/renderer.rs:83` — depth image also `(2560, 1440)`
- Impact: On displays smaller than 2560×1440, GPU memory is wasted. On 4K displays, the render target is smaller than the swapchain, and the blit upscale produces a blurry image. The current render_scale slider compounds this by working against the already-fixed resolution ceiling.
- Fix approach: Size the draw image to `max(window_size, some_configured_max)` or a dynamic resolution target, and recreate it on window resize alongside the swapchain.

**Window size hardcoded in application:**
- Issue: `WIDTH: u32 = 1280` and `HEIGHT: u32 = 720` are constants in `crates/application/src/main.rs:9–10`. No configuration mechanism exists.
- Files: `crates/application/src/main.rs:9–10`
- Impact: Minor — trivial to change, but not user-configurable at runtime.

**Effect index slider hardcoded to `0..=2`:**
- Issue: The egui slider for background effect selection is hardcoded to `0..=2` in `crates/engine/src/ui.rs:107`.
- Files: `crates/engine/src/ui.rs:107`
- Impact: Adding or removing a compute effect requires manually updating this slider range. Should derive from `background_effects.len() - 1`.

**Magic number `active_mesh: 2` at initialization:**
- Issue: `VulkanRenderer` initializes `active_mesh: 2` in `crates/engine/src/renderer.rs:154`. This assumes the loaded glTF file has at least 3 meshes.
- Files: `crates/engine/src/renderer.rs:154`
- Impact: If the loaded model has fewer than 3 meshes, `meshes[self.active_mesh]` will panic at `crates/engine/src/renderer.rs:594`.

**Descriptor pool growth cap is off-by-one:**
- Issue: `GrowableAllocator::get_pool` caps `sets_per_pool` at `4092` (`crates/engine/src/descriptor.rs:188`). The Vulkan spec maximum for descriptor pool sets is typically `4096`. The value `4092` appears arbitrary rather than derived from a physical device query.
- Files: `crates/engine/src/descriptor.rs:188`

---

## Performance Concerns

**`ALL_COMMANDS` stage mask in image barriers:**
- Issue: `transition_image` in `crates/engine/src/renderer.rs:703–706` uses `PipelineStageFlags2::ALL_COMMANDS` for both `src_stage_mask` and `dst_stage_mask` on every image layout transition. The inline comment acknowledges this: `// Using ALL_COMMANDS is inefficient as it will stop gpu commands completely when it arrives at the barrier`.
- Files: `crates/engine/src/renderer.rs:703–706`
- Impact: Every image transition stalls the entire GPU pipeline, eliminating overlap between draw commands. With multiple transitions per frame, this serializes the GPU unnecessarily.
- Fix approach: Use precise stage masks per transition (e.g., `COLOR_ATTACHMENT_OUTPUT` → `TRANSFER`, `FRAGMENT_SHADER` → `COMPUTE_SHADER`).

**No pipeline cache:**
- Issue: Both `create_compute_pipelines` and `create_graphics_pipelines` pass `vk::PipelineCache::null()`.
  - `crates/engine/src/renderer.rs:1063`
  - `crates/engine/src/pipeline.rs:100`
- Impact: Pipeline compilation is fully repeated on every startup with no disk-backed cache. On drivers that do JIT compilation (common on Windows), this increases startup time. Especially relevant if more pipelines are added.
- Fix approach: Create and persist a `vk::PipelineCache` to disk between runs.

**`device_wait_idle` on every swapchain resize:**
- Issue: `resize_swapchain` calls `wait_gpu_idle()` (`crates/engine/src/renderer.rs:1239`), which blocks the CPU until the GPU is fully idle before recreating the swapchain.
- Files: `crates/engine/src/renderer.rs:1239`
- Impact: During rapid window resizes, this causes stuttering. Acceptable for a development renderer but worth noting.

**Frame-rate capped with `thread::sleep`:**
- Issue: The application sleeps `1000/60 ms` per frame in the `RedrawRequested` handler.
  - `crates/application/src/main.rs:94` — `std::thread::sleep(std::time::Duration::from_millis(1000 / 60)); // not really 60 fps`
- Impact: `1000 / 60 = 16` (integer division), not 16.67ms. This caps to ~62.5fps but with uneven frame timing. The comment acknowledges this is not a proper solution. No vsync coordination with the present mode.
- Fix approach: Remove the sleep; rely on `PresentModeKHR::FIFO` for vsync, or implement a proper frame timer.

---

## Missing Error Handling

**Pervasive `.unwrap()` on Vulkan operations:**
- Issue: Nearly every fallible Vulkan operation uses `.unwrap()` rather than returning a `Result`. Failures produce a panic with minimal context rather than a recoverable error or structured error message.
- Files: Throughout `crates/engine/src/renderer.rs`, `crates/engine/src/swapchain.rs`, `crates/engine/src/descriptor.rs`, `crates/engine/src/context.rs`
- Impact: Any Vulkan error (device lost, out of memory, surface lost) crashes the process immediately. For a learning/development renderer this is acceptable, but limits robustness.
- Notable panics: `renderer.rs:230` on `acquire_next_image` for non-`OUT_OF_DATE` errors, `renderer.rs:434` on non-`OUT_OF_DATE` present errors, `descriptor.rs:173` on unexpected descriptor allocation failures.

**glTF loading silently swallows errors:**
- Issue: `load_gltf_meshes` uses `gltf::import(path).ok()?` which silently discards import errors and returns `None`.
  - `crates/engine/src/meshes.rs:36`
- Impact: If the asset file is missing or corrupt, the renderer initializes with `meshes: None` and silently renders nothing (the draw call is behind `if let Some(meshes) = &self.meshes`). No log message is emitted on failure.
- Fix approach: Log the error before discarding it, at minimum.

**glTF index reading only handles U16:**
- Issue: The index reading code in `load_gltf_meshes` only handles `ReadIndices::U16(Iter::Standard(iter))`.
  - `crates/engine/src/meshes.rs:56–62`
- Impact: glTF files with `U32` indices (common for meshes with >65535 vertices) or `U8` indices are silently skipped, producing an empty index buffer. The renderer would then draw 0 primitives without any error.
- Fix approach: Handle all three index types (`U8`, `U16`, `U32`).

**glTF UV coordinates read from channel 1, not 0:**
- Issue: UV coordinates are read from `read_tex_coords(1)` instead of the conventional `read_tex_coords(0)`.
  - `crates/engine/src/meshes.rs:80`
- Impact: Most glTF files store primary UVs in channel 0. Reading channel 1 will produce empty UVs for most assets, causing incorrect or missing texture coordinates. The comment `// I hope so lol` on line 52 suggests uncertainty about correctness in the mesh loading code generally.

**Shader file loading panics on missing file:**
- Issue: `read_shader_from_file` uses `.unwrap()` on `std::fs::File::open`.
  - `crates/engine/src/renderer.rs:994`
- Impact: If a compiled shader `.spv` file is missing (e.g., shaders were not compiled before running), the process panics with a generic IO error. The build script compiles shaders automatically, but if `SKIP_SHADER_COMPILATION=true` is set and `.spv` files are absent, the panic message gives no guidance.

---

## Fragile Areas

**`active_mesh` index can go out of bounds:**
- Issue: The mesh index slider in the UI is bounded to `0..=meshes.len() - 1` (`crates/engine/src/ui.rs:116`), but `active_mesh` is initialized to `2` unconditionally. If `meshes` is `None` or contains fewer than 3 entries, the array access at `crates/engine/src/renderer.rs:594` panics.
- Files: `crates/engine/src/renderer.rs:594`, `crates/engine/src/renderer.rs:154`
- Safe modification: Clamp `active_mesh` to `meshes.len().saturating_sub(1)` at initialization, or default to `0`.

**Image queue sharing mode not implemented:**
- Issue: The swapchain create info comment `// TODO: db: Handle image sharing mode if graphics / present queue are different` at `crates/engine/src/swapchain.rs:56` means the code silently assumes the graphics and present queues are the same family index (i.e., `EXCLUSIVE` sharing mode is implicitly used).
- Files: `crates/engine/src/swapchain.rs:56`
- Impact: On hardware where graphics and present queues are different families (rare but valid), the swapchain images will not be correctly shared and Vulkan validation layers will produce errors. The device selection in `context.rs` does not enforce that both queues share a family.

**UI texture free timing is uncertain:**
- Issue: The comment in `crates/engine/src/ui.rs:193–194` reads `// ? soundness with multiple frames in flight` and `// ? move to after frame`. Textures marked for freeing are deferred to the next frame's `after_frame` call, but with `FRAME_OVERLAP = 2`, a texture freed in frame N may still be in use by GPU work from frame N-1.
- Files: `crates/engine/src/ui.rs:193–201`
- Impact: Potential use-after-free of GPU texture resources if the GPU is still reading the texture when it is destroyed. Validation layers may catch this in debug builds.

## Missing Abstractions

**Scene data allocated and written but never actually used in shaders:**
- Issue: `GPUSceneData` is written to a per-frame uniform buffer each draw call (`crates/engine/src/renderer.rs:651–673`), a descriptor set is allocated and written to point at it — but the descriptor set `desc_set` is never bound to the pipeline with `cmd_bind_descriptor_sets`. The mesh pipeline layout (`mesh_pipeline_layout`) was created with no set layouts (`crates/engine/src/renderer.rs:1085–1088`), so even if bound, the shader cannot access it.
- Files: `crates/engine/src/renderer.rs:651–677`
- Impact: The scene data (view, projection, lights) is computed and uploaded but has no effect on rendering. The mesh shader cannot access camera/lighting data.

**Model transform is hardcoded in draw_geometry:**
- Issue: The view, projection, and model matrices for the mesh are computed inline in `draw_geometry` with hardcoded values: camera at `z = -5`, model scale `2.0`, rotation `10 degrees`.
  - `crates/engine/src/renderer.rs:596–613`
- Files: `crates/engine/src/renderer.rs:596–613`
- Impact: No scene graph, no camera abstraction. Adding a second object or a movable camera requires modifying the renderer directly.

**No texture/material support:**
- Issue: UV coordinates are loaded from glTF (though from the wrong channel — see above) but are set to `0.0` in the assembled `Vertex` struct (`crates/engine/src/meshes.rs:112–113`). No samplers, no texture images, no material pipeline variants exist.
- Files: `crates/engine/src/meshes.rs:112–113`
- Impact: All meshes render with color-as-normals override (`OVERRIDE_COLORS: bool = true` at `crates/engine/src/meshes.rs:121`). Textures are not functional.

**`before_frame` UI function takes a large untyped tuple:**
- Issue: The `before_frame` function in `crates/engine/src/ui.rs:67–74` accepts a 6-element tuple of heterogeneous mutable references as its `active_data` parameter. This is the UI layer reaching directly into renderer internals.
- Files: `crates/engine/src/ui.rs:62–76`
- Impact: Tightly couples the UI code to the internal structure of `VulkanRenderer`. Adding a new controllable parameter requires changing the tuple signature at all call sites.
- Fix approach: Define a `UiControlData` struct and pass that instead.

---

## Build System Concerns

**Shader compilation depends on `glslangValidator` being in PATH:**
- Issue: `crates/assets/build.rs:47` invokes `glslangValidator` as an external process. If it is not installed or not in the system PATH, the build fails with a generic `Command` error.
- Files: `crates/assets/build.rs:47`
- Impact: New contributors get an unhelpful error message. No version check is performed, so old glslangValidator versions may produce incompatible SPIR-V silently.
- Fix approach: Document the required tool and version in README. Consider using the `shaderc` Rust crate to compile shaders without an external tool dependency.

**Build script `rerun-if-changed` lists a file that does not match compilation:**
- Issue: `build.rs` registers `shader.frag` and `shader.vert` (`crates/assets/build.rs:15–16`) as change-detection targets, but the `compile_shaders` function compiles all non-`.spv` files in the directory indiscriminately. These filenames suggest older tutorial-era shaders that may or may not still be present.
- Files: `crates/assets/build.rs:15–16`

---

## Test Coverage Gaps

**No tests exist anywhere in the codebase:**
- What's not tested: All engine logic, descriptor management, swapchain creation, mesh loading, pipeline building, buffer allocation.
- Files: All files in `crates/engine/src/`, `crates/application/src/`
- Risk: Regressions in any module go undetected. The Vulkan renderer is inherently hard to unit test (requires a GPU), but pure logic such as `choose_swapchain_surface_format`, `choose_swapchain_present_mode`, `choose_swapchain_extent` in `crates/engine/src/swapchain.rs`, the descriptor pool growth math in `crates/engine/src/descriptor.rs`, and glTF loading in `crates/engine/src/meshes.rs` are all testable without GPU hardware.
- Priority: Low (learning/development project), but `meshes.rs` logic in particular has correctness bugs (wrong UV channel, missing U32 index support) that tests would have caught.

---

*Concerns audit: 2026-03-02*
