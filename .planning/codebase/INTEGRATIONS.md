# External Integrations

**Analysis Date:** 2026-03-02

## Graphics API

**Vulkan:**
- **Version target:** Vulkan 1.3 (set explicitly via `vk::make_api_version(0, 1, 3, 0)` in `crates/engine/src/context.rs`)
- **Bindings:** `ash` 0.38 - raw unsafe Rust bindings generated from the Vulkan headers (header version 1.3.281 per Cargo.lock)
- **Surface integration:** `ash-window` 0.13 bridges `winit`'s raw window handle types (`HasDisplayHandle`, `HasWindowHandle`) to Vulkan surface creation via `ash_window::create_surface()`

**Required Vulkan Extensions (Instance):**
- Platform surface extensions (enumerated dynamically from `ash_window::enumerate_required_extensions`)
- `VK_EXT_debug_utils` (debug builds only, added in `crates/engine/src/context.rs`)

**Required Vulkan Extensions (Device):**
- `VK_KHR_swapchain` - declared in `get_required_device_extensions()` in `crates/engine/src/context.rs`

**Required Vulkan Device Features:**
- Vulkan 1.0: `samplerAnisotropy`
- Vulkan 1.2 (`VkPhysicalDeviceVulkan12Features`): `bufferDeviceAddress`, `descriptorIndexing`
- Vulkan 1.3 (`VkPhysicalDeviceVulkan13Features`): `dynamicRendering`, `synchronization2`

The use of `dynamicRendering` (Vulkan 1.3 core) means no render pass objects are created anywhere in the codebase. All rendering targets draw images and swapchain images directly.

## Windowing

**winit 0.30 (resolved: 0.52.6):**
- Provides `EventLoop`, `Window`, `ApplicationHandler` trait
- The application binary (`crates/application/src/main.rs`) uses the new-style `ApplicationHandler` impl pattern introduced in winit 0.30
- Event loop runs in `Poll` mode (spin loop, not event-driven sleep)
- Window is created at 1280x720 with title "Render Thing"
- Frame pacing is a hard-coded `sleep(1000/60ms)` in the `RedrawRequested` handler

## GPU Memory Management

**vk-mem 0.5.0 (resolved: 0.9.5) — Vulkan Memory Allocator (VMA):**
- C library bindings for AMD's VMA
- Allocator created in `crates/engine/src/context.rs` with:
  - `vulkan_api_version = VK_API_VERSION_1_3`
  - `BUFFER_DEVICE_ADDRESS` allocator flag (required for GPU-side vertex buffer pointers)
- Allocator is wrapped in `Arc<vk_mem::Allocator>` and shared across the renderer and UI context
- Used for all `AllocatedBuffer` and `AllocatedImage` GPU memory in `crates/engine/src/renderer.rs`

## Immediate-Mode GUI

**egui 0.33.2 + egui-winit 0.33.2 + egui-ash-renderer 0.11.0:**
- Full egui integration implemented in `crates/engine/src/ui.rs`
- `egui-winit::State` translates winit `WindowEvent`s into egui input
- `egui-ash-renderer::Renderer` renders tessellated egui primitives to the swapchain via Vulkan dynamic rendering
- Renderer is created with `Renderer::with_vk_mem_allocator()` so egui textures use VMA-managed memory
- `in_flight_frames` set to `FRAME_OVERLAP` (2) to match the double-buffering frame overlap
- The GUI panel ("Shader control" window) exposes: render scale slider, background effect selector, push constant data editors, mesh selector

## 3D Asset Loading

**gltf 1.4.1 (with `utils` feature):**
- Used in `crates/engine/src/meshes.rs` via `gltf::import(path)`
- Loads `.glb` binary glTF files
- Reads: vertex positions, normals, UV coordinates (set 1), RGBA colors (set 0), U16 triangle indices
- Uploaded to GPU as `GPUMeshBuffers` (index + vertex buffer pair with device address)
- Assets at `assets/models/`: `basicmesh.glb`, `ashford_abbey.glb`, `happy_cloud.glb`, `chalet.obj` (obj not loaded in current code)

## Mathematics

**nalgebra-glm 0.20.0 (aliased as `glm`):**
- Used as `extern crate nalgebra_glm as glm` in `crates/engine/src/lib.rs`
- Provides GLSL-compatible types: `glm::Mat4` used in `GPUDrawPushConstants.world_matrix` and `GPUSceneData` view/proj matrices
- The `GPUSceneData` and `GPUDrawPushConstants` structs are `#[repr(C)]` for direct GPU upload

## Shader Compilation

**glslangValidator (external CLI tool, not a Rust crate):**
- Invoked by the Cargo build script at `crates/assets/build.rs` during `cargo build`
- Compiles all non-`.spv` files in `assets/shaders/` to SPIR-V `.spv` files in-place
- Command: `glslangValidator -V <input> -o <input>.spv`
- Must be installed separately (part of the Vulkan SDK)
- Controlled by env var `SKIP_SHADER_COMPILATION=true` to skip recompilation

**Shader inventory in `assets/shaders/`:**
| Source File | Type | GLSL Version | Notes |
|---|---|---|---|
| `shader.vert` / `shader.frag` | Graphics | 450 | Basic textured triangle |
| `colored_triangle.vert` / `.frag` | Graphics | 450 | Hardcoded vertex positions |
| `colored_triangle_mesh.vert` | Graphics | 450 | Buffer reference push constants; uses `GL_EXT_buffer_reference` |
| `gradient.comp` | Compute | 460 | Simple UV gradient, 16x16 workgroup |
| `gradient_color.comp` | Compute | 460 | Parameterized color gradient |
| `sky.comp` | Compute | 460 | Sky effect |

The mesh vertex shader (`colored_triangle_mesh.vert`) uses `GL_EXT_buffer_reference` to read the vertex buffer directly via a GPU device address passed in push constants — this is the Vulkan 1.2 `bufferDeviceAddress` feature in action.

## Validation / Debug Layer

**VK_LAYER_KHRONOS_validation:**
- Enabled automatically in debug builds (when `cfg!(debug_assertions)`)
- Controlled in `crates/engine/src/debug.rs`
- Debug messenger routes Vulkan messages to the `log` facade at appropriate levels (verbose→debug, info→info, warning→warn, error→error)
- Disabled in release builds with zero overhead

## Logging

**log 0.4 + env_logger 0.11:**
- `log::trace!`, `log::debug!`, `log::info!`, `log::warn!`, `log::error!` used throughout engine code
- Initialized in `main()` via `env_logger::init()`
- Log level controlled by `RUST_LOG` environment variable (e.g. `RUST_LOG=vulkan_rust=debug cargo run`)

## Dependency Auditing

**cargo-deny:**
- Config at `deny.toml`
- Enforces crates.io-only sources (no git dependencies allowed)
- License allowlist enforced
- Run with `cargo deny check`

## Platform Notes

**Windows (primary development platform):**
- Uses MSVC ABI (`x86_64-pc-windows-msvc`)
- Vulkan loaded dynamically via `Entry::load()` (links against `vulkan-1.dll`)
- `imgui.ini` present in repo root (egui layout persistence file)

**macOS (aarch64-apple-darwin):**
- Vulkan on macOS requires MoltenVK (translates Vulkan calls to Metal)
- `ash_window` handles the platform-specific surface type (CAMetalLayer via MoltenVK)

**Linux (x86_64-unknown-linux-gnu):**
- Standard Vulkan via libvulkan.so
- Display server abstraction handled by winit + ash-window (X11 or Wayland)

## File Storage

- No cloud or remote file storage
- All assets are local files committed to the repository under `assets/`
- No networking, no remote services, no authentication

## CI/CD

- No CI pipeline configuration detected (no `.github/`, `.gitlab-ci.yml`, etc.)
- `cargo-deny` is configured but not wired to any automated pipeline

---

*Integration audit: 2026-03-02*
