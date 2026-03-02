# Codebase Structure

**Analysis Date:** 2026-03-02

## Directory Layout

```
vulkan-rust/
├── Cargo.toml              # Workspace root — defines members and shared dependencies
├── Cargo.lock              # Locked dependency tree
├── rust-toolchain.toml     # Pins nightly Rust; multi-target (Windows/Linux/macOS)
├── rustfmt.toml            # Formatting config
├── deny.toml               # cargo-deny license/advisory policy
├── todos.md                # Informal developer notes and refactor TODOs
├── imgui.ini               # egui window layout persisted state
├── assets/                 # Runtime assets (shaders, models, images)
│   ├── shaders/            # GLSL sources + compiled .spv files
│   ├── models/             # glTF and OBJ mesh files
│   │   └── downloaded/     # Third-party downloaded models
│   └── images/             # Texture images (jpg)
├── crates/                 # Workspace member crates
│   ├── application/        # Binary entry point (render-thing)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   ├── engine/             # Core Vulkan rendering engine (library)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Public API: Engine struct
│   │       ├── renderer.rs     # VulkanRenderer + all GPU data structs
│   │       ├── context.rs      # VkContext: instance/device/surface
│   │       ├── swapchain.rs    # Swapchain creation and management
│   │       ├── descriptor.rs   # Descriptor pools, layouts, writers
│   │       ├── pipeline.rs     # PipelineBuilder for graphics pipelines
│   │       ├── primitives.rs   # GPU-facing data structs (Vertex, push consts)
│   │       ├── meshes.rs       # glTF loading + GPU mesh upload
│   │       ├── ui.rs           # egui integration (before/render/after frame)
│   │       └── debug.rs        # Vulkan validation layer setup
│   └── assets/             # Build-script-only crate (compiles shaders)
│       ├── Cargo.toml
│       ├── build.rs            # Calls glslangValidator at build time
│       └── src/
│           └── assets.rs       # Empty — crate exists only for build.rs
└── .planning/
    └── codebase/           # GSD planning documents
```

## Directory Purposes

**`crates/application/`:**
- Purpose: The runnable binary. Thin layer between winit and the engine.
- Contains: One file — `main.rs` with `Application` (winit `ApplicationHandler`) and `main()`
- Key files: `crates/application/src/main.rs`

**`crates/engine/`:**
- Purpose: All Vulkan rendering logic. Compiled as a `lib` crate; re-exported only through `Engine`.
- Contains: 9 Rust modules (all `pub(crate)` internal except `Engine` itself)
- Key files: `crates/engine/src/lib.rs`, `crates/engine/src/renderer.rs`

**`crates/assets/`:**
- Purpose: Shader compilation at build time only. The `lib` target (`src/assets.rs`) is empty.
- Contains: `build.rs` — scans `assets/shaders/`, calls `glslangValidator -V` on every non-`.spv` file
- Key files: `crates/assets/build.rs`
- Note: Must be listed as a workspace member for `build.rs` to run; controlled by `SKIP_SHADER_COMPILATION` env var

**`assets/shaders/`:**
- Purpose: GLSL shader sources and their compiled SPIR-V output
- Generated `.spv` files are checked into the repo alongside sources
- Key files:
  - `gradient.comp` / `gradient_color.comp` / `sky.comp` — background compute effects
  - `colored_triangle_mesh.vert` — mesh vertex shader (reads vertices via buffer device address)
  - `colored_triangle.frag` — mesh fragment shader
  - `colored_triangle.vert`, `shader.vert`, `shader.frag` — older/unused shaders still present

**`assets/models/`:**
- Purpose: 3D mesh assets loaded at runtime
- `basicmesh.glb` — primary mesh (loaded at startup in `VulkanRenderer::initialize`)
- `downloaded/` — additional glTF models (not currently wired into the loader)
- `chalet.obj` — older OBJ format model (not loaded)

**`assets/images/`:**
- Purpose: Texture images (not currently wired into any sampler/material system)
- `chalet.jpg`, `statue.jpg`

## Key File Locations

**Entry Points:**
- `crates/application/src/main.rs`: `fn main()` — binary entry, winit event loop
- `crates/engine/src/lib.rs`: `Engine::initialize` — engine startup, called from `Application::resumed`

**Vulkan Initialization:**
- `crates/engine/src/context.rs`: `VkContext::initialize` — instance, device, surface, VMA allocator
- `crates/engine/src/swapchain.rs`: `Swapchain::create` — swapchain and image views

**Frame Render Loop:**
- `crates/engine/src/renderer.rs`: `VulkanRenderer::draw` (line 172) — the entire per-frame sequence

**Pipeline Creation:**
- `crates/engine/src/renderer.rs`: `VulkanRenderer::initialize_mesh_pipeline` (line 1068) — graphics pipeline
- `crates/engine/src/renderer.rs`: `VulkanRenderer::initialize_effect_pipelines` (line 1004) — compute pipelines
- `crates/engine/src/pipeline.rs`: `PipelineBuilder` — reusable graphics pipeline builder

**GPU Data Structures:**
- `crates/engine/src/primitives.rs`: `Vertex`, `GPUDrawPushConstants`, `GPUSceneData`, `GPUMeshBuffers`
- `crates/engine/src/renderer.rs` (bottom): `FrameData`, `AllocatedBuffer`, `AllocatedImage`, `ComputeEffect`, `ComputePushConstants`, `QueueData`, `ImmediateSubmitData`

**Descriptor Management:**
- `crates/engine/src/descriptor.rs`: `LayoutBuilder`, `Allocator`, `GrowableAllocator`, `DescriptorWriter`

**Mesh Loading:**
- `crates/engine/src/meshes.rs`: `load_gltf_meshes` — parses glTF, builds `MeshAsset`
- `crates/engine/src/renderer.rs`: `upload_mesh_internal` (line 1137) — staging buffer GPU upload

**UI:**
- `crates/engine/src/ui.rs`: `UiContext`, `before_frame`, `render`, `after_frame`

**Build / Shaders:**
- `crates/assets/build.rs`: shader compilation via `glslangValidator`
- `assets/shaders/`: all GLSL sources and compiled `.spv` files

## Naming Conventions

**Files:**
- Snake_case module files: `renderer.rs`, `context.rs`, `debug.rs`
- Module name matches the primary concept it contains

**Structs:**
- PascalCase: `VulkanRenderer`, `VkContext`, `FrameData`, `AllocatedBuffer`
- GPU-facing structs prefixed with `GPU`: `GPUMeshBuffers`, `GPUDrawPushConstants`, `GPUSceneData`
- Vulkan wrapper structs prefixed with `Vk`: `VkContext`

**Functions:**
- Snake_case: `initialize`, `draw`, `upload_mesh`, `create_image`, `init_descriptors`
- Constructor-style functions named `initialize`, `create`, `init`, or `new` depending on context
- Free functions in `ui.rs` named by lifecycle position: `before_frame`, `render`, `after_frame`

**Visibility:**
- All engine internals are `pub(crate)` — nothing leaks through `engine` except `Engine` itself
- `#![allow(dead_code)]` at the crate root suppresses warnings for in-progress code

## Where to Add New Code

**New render feature / draw pass:**
- Primary code: `crates/engine/src/renderer.rs` — add to `VulkanRenderer` struct and `draw` method
- New pipeline: use `PipelineBuilder` from `crates/engine/src/pipeline.rs`
- New GPU data types: `crates/engine/src/primitives.rs`

**New shader:**
- Source: `assets/shaders/<name>.<stage>` (e.g. `assets/shaders/shadow.vert`)
- The `assets` build script auto-discovers and compiles all non-`.spv` files in `assets/shaders/`
- Update `crates/assets/build.rs` `rerun-if-changed` lines to add the new file

**New mesh / model:**
- Drop `.glb` file in `assets/models/`
- Wire into `VulkanRenderer::initialize` by calling `load_gltf_meshes` with the new path

**New UI control:**
- `crates/engine/src/ui.rs`: `before_frame` function — add egui widgets
- Pass new mutable references through the `active_data` tuple argument (currently a workaround; refactoring to a proper struct is noted in `todos.md`)

**New descriptor set layout:**
- `crates/engine/src/descriptor.rs`: use `LayoutBuilder` to define bindings, then `Allocator` or `GrowableAllocator` to allocate

**New abstraction / module:**
- Add `<name>.rs` to `crates/engine/src/`
- Declare as `mod <name>;` in `crates/engine/src/lib.rs`

## Special Directories

**`assets/shaders/` (`.spv` files):**
- Purpose: Compiled SPIR-V binaries consumed at runtime by `VulkanRenderer`
- Generated: Yes (by `crates/assets/build.rs` via `glslangValidator`)
- Committed: Yes — `.spv` files are committed alongside sources so builds without `glslangValidator` installed still work (controlled by `SKIP_SHADER_COMPILATION`)

**`target/`:**
- Purpose: Cargo build artifacts
- Generated: Yes
- Committed: No (in `.gitignore`)

**`.planning/codebase/`:**
- Purpose: GSD architecture and convention documents for AI-assisted development
- Generated: By GSD map-codebase commands
- Committed: Yes

**`.cargo/`:**
- Purpose: Workspace-level Cargo configuration
- Committed: Yes (contains registry or patch config if present)

---

*Structure analysis: 2026-03-02*
