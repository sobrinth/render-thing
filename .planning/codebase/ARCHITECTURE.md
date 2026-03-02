# Architecture

**Analysis Date:** 2026-03-02

## Pattern Overview

**Overall:** Layered Vulkan renderer with thin application shell

**Key Characteristics:**
- Single-threaded render loop driven by `winit` event loop in Poll mode
- Double-buffered frame-in-flight design (FRAME_OVERLAP = 2)
- Render-to-intermediate-image approach: geometry is drawn into an HDR offscreen image, then blit-copied to the swapchain image before UI is composited on top
- No ECS, no scene graph — all scene data and draw calls are hardcoded in `VulkanRenderer`
- Vulkan 1.3 features used: dynamic rendering (no render passes), synchronization2, buffer device address, descriptor indexing

## Layers

**Application Layer:**
- Purpose: Window lifecycle management and event dispatch
- Location: `crates/application/src/main.rs`
- Contains: `Application` struct implementing `winit::ApplicationHandler`
- Depends on: `engine` crate, `winit`
- Used by: OS / Cargo (binary entry point)

**Engine Public API:**
- Purpose: Thin facade that hides all Vulkan details from the application
- Location: `crates/engine/src/lib.rs`
- Contains: `Engine` struct with `initialize`, `draw`, `resize`, `on_window_event`, `stop`
- Depends on: `VulkanRenderer` (internal)
- Used by: `application` crate

**Renderer Core:**
- Purpose: All Vulkan state, pipelines, frame management, and draw logic
- Location: `crates/engine/src/renderer.rs`
- Contains: `VulkanRenderer`, `FrameData`, `AllocatedImage`, `AllocatedBuffer`, `ComputeEffect`, `ImmediateSubmitData`, `QueueData`
- Depends on: `context`, `swapchain`, `descriptor`, `pipeline`, `meshes`, `primitives`, `ui`
- Used by: `Engine` only

**Vulkan Context:**
- Purpose: Instance, physical device, logical device, and surface lifecycle
- Location: `crates/engine/src/context.rs`
- Contains: `VkContext` — wraps `ash::Entry`, `Instance`, `Device`, `SurfaceKHR`, `PhysicalDevice`, debug messenger
- Depends on: `debug`, `swapchain` (for device suitability checks)
- Used by: `VulkanRenderer`

**Swapchain:**
- Purpose: Swapchain creation, image/view management, format/present-mode selection, resize
- Location: `crates/engine/src/swapchain.rs`
- Contains: `Swapchain`, `SwapchainProperties`, `SwapchainSupportDetails`
- Depends on: `VkContext`
- Used by: `VulkanRenderer`

**Descriptor System:**
- Purpose: Descriptor pool management, layout building, and set writes
- Location: `crates/engine/src/descriptor.rs`
- Contains: `LayoutBuilder`, `Allocator` (fixed pool), `GrowableAllocator` (per-frame, auto-grows), `DescriptorWriter`, `PoolSizeRatio`
- Depends on: `ash`
- Used by: `VulkanRenderer`, `FrameData`

**Pipeline Builder:**
- Purpose: Ergonomic builder for `vk::Pipeline` (graphics pipelines only; compute pipelines built inline in renderer)
- Location: `crates/engine/src/pipeline.rs`
- Contains: `PipelineBuilder` — configures shaders, topology, rasterization, depth, blending, dynamic state, dynamic rendering
- Depends on: `ash`
- Used by: `VulkanRenderer::initialize_mesh_pipeline`

**GPU Primitives / Data Types:**
- Purpose: `#[repr(C)]` structs that are uploaded to the GPU verbatim
- Location: `crates/engine/src/primitives.rs`
- Contains: `Vertex`, `GPUMeshBuffers`, `GPUDrawPushConstants`, `GPUSceneData`
- Depends on: `AllocatedBuffer` (from renderer)
- Used by: `VulkanRenderer`, `meshes`

**Mesh Loading:**
- Purpose: glTF mesh import and GPU upload
- Location: `crates/engine/src/meshes.rs`
- Contains: `MeshAsset`, `GeoSurface`, `load_gltf_meshes`
- Depends on: `gltf` crate, `VulkanRenderer::upload_mesh`, `primitives`
- Used by: `VulkanRenderer` (at initialization)

**UI System:**
- Purpose: egui integration — per-frame begin/render/end lifecycle
- Location: `crates/engine/src/ui.rs`
- Contains: `UiContext`, free functions `before_frame`, `render`, `after_frame`
- Depends on: `egui`, `egui-winit`, `egui-ash-renderer`
- Used by: `VulkanRenderer`

**Debug Layer:**
- Purpose: Vulkan validation layer setup and debug callback
- Location: `crates/engine/src/debug.rs`
- Contains: `vulkan_debug_callback`, `setup_debug_messenger`, `check_validation_layer_support`, `get_layer_names_and_pointers`
- Active only in debug builds (`cfg!(debug_assertions)`)
- Used by: `VkContext`

**Assets Build Script:**
- Purpose: Compile GLSL shader sources to SPIR-V at Cargo build time using `glslangValidator`
- Location: `crates/assets/build.rs`
- Note: `assets` lib crate itself is empty (`src/assets.rs` is blank); it exists solely for the build script
- Env var `SKIP_SHADER_COMPILATION=true` bypasses compilation

## Data Flow

**Per-Frame Render Loop:**

1. `Application::window_event` receives `WindowEvent::RedrawRequested` from winit
2. `Engine::draw` delegates to `VulkanRenderer::draw`
3. If resize is pending: swapchain is destroyed and recreated, frame is skipped
4. Fence wait on `FrameData::render_fence` — blocks until the GPU has finished the previous frame using this slot
5. `swapchain.acquire_next_image` → gets next swapchain image index
6. `ui::before_frame` — runs egui layout pass, produces `Vec<ClippedPrimitive>`
7. Command buffer reset and begin (ONE_TIME_SUBMIT)
8. **Background pass** (compute):
   - Transition draw image: `UNDEFINED → GENERAL`
   - Dispatch active compute effect (`gradient`, `gradient_color`, or `sky`) via `draw_background`
9. **Geometry pass** (graphics):
   - Transition draw image: `GENERAL → COLOR_ATTACHMENT_OPTIMAL`
   - Transition depth image: `UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL`
   - `cmd_begin_rendering` with draw image + depth image (dynamic rendering, no render pass object)
   - Bind mesh pipeline, set dynamic viewport + scissor
   - Push `GPUDrawPushConstants` (world matrix + vertex buffer device address) per mesh
   - `cmd_draw_indexed` for active mesh surface
   - Copy `GPUSceneData` to frame's `scene_buffer` (host-mapped uniform buffer)
   - Allocate per-frame descriptor set from `GrowableAllocator`, write scene buffer, `update_set`
   - `cmd_end_rendering`
10. **Blit to swapchain**:
    - Transition draw image: `COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL`
    - Transition swapchain image: `UNDEFINED → TRANSFER_DST_OPTIMAL`
    - `cmd_blit_image2` (linear filter) from draw image to swapchain image
11. **UI pass** (graphics, directly on swapchain image):
    - Transition swapchain image: `TRANSFER_DST_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL`
    - `cmd_begin_rendering` on swapchain image view
    - `ui::render` — egui-ash-renderer draws clipped primitives
    - `cmd_end_rendering`
12. Transition swapchain image: `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR`
13. `end_command_buffer`
14. `queue_submit2` — waits on `acquire_semaphore`, signals `swapchain.semaphores[image_index]`, signals `render_fence`
15. `queue_present` — waits on swapchain semaphore
16. `ui::after_frame` — frees textures from previous frame
17. `frame_number` incremented; `(frame_number % FRAME_OVERLAP)` selects the next frame slot

**Mesh Upload Flow (startup, immediate submit):**

1. `load_gltf_meshes` called from `VulkanRenderer::initialize` with `assets/models/basicmesh.glb`
2. `gltf` crate parses the file; positions, normals, UVs, colors extracted into `Vec<Vertex>`
3. `VulkanRenderer::upload_mesh` → `upload_mesh_internal`:
   - Creates GPU-local vertex buffer (STORAGE_BUFFER | SHADER_DEVICE_ADDRESS) and index buffer (INDEX_BUFFER)
   - Creates a host-visible staging buffer
   - Copies vertex + index data into staging via `ptr::copy_nonoverlapping`
   - Uses `immediate_submit` (dedicated command pool + fence) to `cmd_copy_buffer` staging → GPU buffers synchronously
   - Destroys staging buffer
4. Vertex buffer device address stored in `GPUMeshBuffers::vertex_buffer_address` and pushed as a push constant each draw call (bindless vertex fetch in the shader)

**State Management:**
- No ECS or component system. All render state (`meshes`, `active_mesh`, `background_effects`, `active_background_effect`, `render_scale`) lives directly on `VulkanRenderer`
- Per-frame transient state (descriptor sets, scene uniform) allocated fresh each frame from `FrameData::descriptors` (a `GrowableAllocator` that is reset at the start of each frame)
- egui mutable state (shader data, mesh selection) is passed as raw mutable references through the `ui::before_frame` tuple argument

## Key Abstractions

**`AllocatedBuffer`:**
- Purpose: GPU buffer + VMA allocation bundled together with the `AllocationInfo`
- Location: `crates/engine/src/renderer.rs` (bottom section)
- Pattern: Created via `AllocatedBuffer::create`, destroyed via `AllocatedBuffer::destroy`; wraps `vk_mem::Allocator`

**`AllocatedImage`:**
- Purpose: GPU image + view + VMA allocation bundled together
- Location: `crates/engine/src/renderer.rs`
- Pattern: Created via `VulkanRenderer::create_image`, destroyed via `AllocatedImage::destroy`

**`FrameData`:**
- Purpose: All per-frame-in-flight GPU resources: command pool/buffer, acquire semaphore, render fence, per-frame descriptor allocator, scene uniform buffer
- Location: `crates/engine/src/renderer.rs`
- Pattern: `Vec<FrameData>` with length `FRAME_OVERLAP` (2); indexed by `frame_number % FRAME_OVERLAP`

**`GrowableAllocator`:**
- Purpose: Per-frame descriptor pool that automatically provisions new pools when the current one is full
- Location: `crates/engine/src/descriptor.rs`
- Pattern: `clear_pools` at frame start (moves full pools back to ready), `allocate` during frame; grows pool count by 1.5x when exhausted, capped at 4092 sets

**`ComputeEffect`:**
- Purpose: A named compute pipeline with `ComputePushConstants` (4x vec4 of floats) for background rendering
- Location: `crates/engine/src/renderer.rs`
- Three effects: `color_grid` (gradient.comp), `gradient` (gradient_color.comp), `sky` (sky.comp)
- Switched via egui slider at runtime

**`PipelineBuilder`:**
- Purpose: Builder pattern for graphics pipeline creation; eliminates repetitive Vulkan struct wiring
- Location: `crates/engine/src/pipeline.rs`
- Pattern: `PipelineBuilder::init()` → configure via setters → `build(device)` → destroy shader modules manually

## Entry Points

**Application Binary:**
- Location: `crates/application/src/main.rs`
- Triggers: `cargo run` (default member is `render-thing`)
- Responsibilities: Creates winit event loop, instantiates `Application`, runs event loop until close

**Engine Initialization:**
- Location: `crates/engine/src/lib.rs` → `Engine::initialize`
- Triggers: Called from `Application::resumed` (winit lifecycle)
- Responsibilities: Creates `VulkanRenderer`, which creates full Vulkan context, swapchain, pipelines, loads meshes

**Build-time Shader Compilation:**
- Location: `crates/assets/build.rs`
- Triggers: `cargo build` when any `.glsl` source in `assets/shaders/` changes
- Responsibilities: Calls `glslangValidator -V` for each non-`.spv` file in the shaders directory

## Error Handling

**Strategy:** Panic-on-failure throughout. No `Result` propagation into application code.

**Patterns:**
- All Vulkan calls use `.unwrap()` or `.expect("message")` — recoverable errors are not modeled
- Swapchain OUT_OF_DATE is the one handled case: sets `resize_requested = true` and skips the frame
- Window minimize (size 0,0) during resize: returns `true` (minimized) to skip recreation
- Asset loading (`load_gltf_meshes`) returns `Option<Vec<MeshAsset>>` — `None` stored as `meshes: None`, draw silently skipped

## Cross-Cutting Concerns

**Logging:** `log` crate with `env_logger`. `log::trace!` for Vulkan object lifecycle (drop), `log::debug!` for device selection and asset loading, `log::warn!`/`log::error!` from Vulkan debug callback.

**Validation:** Vulkan validation layer (`VK_LAYER_KHRONOS_validation`) enabled only in debug builds. Debug messenger forwards messages to the `log` facade.

**Memory Management:** All GPU allocations go through a shared `Arc<vk_mem::Allocator>` (VMA). The `Arc` is cloned into egui renderer and each component that needs allocation. RAII `Drop` impls on `VulkanRenderer` destroy everything in correct dependency order (wait idle → destroy resources → destroy swapchain → context drops last).

**Frame Timing:** `std::thread::sleep(Duration::from_millis(1000 / 60))` in the application loop provides a crude ~60fps cap. Not accurate; acknowledged in a comment.

---

*Architecture analysis: 2026-03-02*
