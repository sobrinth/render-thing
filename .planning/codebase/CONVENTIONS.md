# Coding Conventions

**Analysis Date:** 2026-03-02

## Rust Edition & Toolchain

- **Edition:** Rust 2024 (set in `rustfmt.toml` and `Cargo.toml`)
- **Toolchain:** Nightly (pinned in `rust-toolchain.toml`)
- **Components:** `rustfmt`, `clippy` (both required)
- **Targets:** `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`

## Formatting Rules

Config file: `rustfmt.toml`

- `edition = "2024"` — use Rust 2024 edition formatting rules
- `newline_style = "Unix"` — LF line endings, not CRLF (important on Windows dev machines)

Run formatting: `cargo fmt`
Check formatting: `cargo fmt -- --check`

## Naming Patterns

**Types (structs, enums):**
- `PascalCase` throughout
- GPU-bound types prefixed with `GPU`: `GPUMeshBuffers`, `GPUDrawPushConstants`, `GPUSceneData`
- Vulkan wrapper types prefixed with `Vk`: `VkContext`
- Allocated GPU resource types prefixed with `Allocated`: `AllocatedImage`, `AllocatedBuffer`
- Builder types suffixed with `Builder`: `LayoutBuilder`, `PipelineBuilder`
- Context/state types suffixed with `Context`: `VkContext`, `UiContext`
- Support detail types suffixed with `Details`: `SwapchainSupportDetails`
- Properties types suffixed with `Properties`: `SwapchainProperties`

**Functions:**
- `snake_case` throughout
- Constructors use `new()`, `initialize()`, `init()`, `create()`, or `init_pool()` — no single canonical name, but `initialize` is used for main entry points (`VkContext::initialize`, `Engine::initialize`)
- Builder entry point is `init()` for builders: `PipelineBuilder::init()`
- Destruction methods named `destroy()` or `drop` via `Drop` trait
- Boolean-returning predicates named `is_*` or `check_*`: `is_device_suitable`, `check_device_extension_support`
- Private helpers follow descriptive verb-noun style: `find_queue_families`, `get_ideal_swapchain_properties`, `choose_swapchain_surface_format`

**Variables and fields:**
- `snake_case` throughout
- GPU/Vulkan suffixes respected: `swapchain_fn`, `swapchain_image_views`, `graphics_queue`
- Underscore prefix `_` for intentionally unused variables: `_vulkan_fn`, `_app`, `_position`

**Constants:**
- `SCREAMING_SNAKE_CASE`: `FRAME_OVERLAP`, `REQUIRED_LAYERS`, `OVERRIDE_COLORS`

**Modules:**
- `snake_case` filenames: `context.rs`, `debug.rs`, `descriptor.rs`, `meshes.rs`, `pipeline.rs`, `primitives.rs`, `renderer.rs`, `swapchain.rs`, `ui.rs`

**Crate names:**
- `snake_case`: `engine`, `assets`
- Package (binary) name can differ from crate: binary is `render-thing`, crate is `application`

## Visibility Pattern

The engine crate uses `pub(crate)` extensively for internal APIs — types and functions visible to the whole crate but not exported. Only the public API surface of the `engine` crate (the `Engine` struct and its methods in `lib.rs`) uses bare `pub`.

```rust
// Internal use only - not part of public API
pub(crate) struct VulkanRenderer { ... }
pub(crate) fn initialize(window: &Window) -> Self { ... }

// Public API (in lib.rs)
pub struct Engine { ... }
pub fn initialize(window: &Window) -> Self { ... }
```

Some types in `descriptor.rs` and `primitives.rs` mix `pub` and `pub(crate)` inconsistently — this appears to be in-progress cleanup.

## Import Organization

Imports are grouped and ordered as follows (observed consistently across files):

1. Internal crate imports (`use crate::...`)
2. External crate imports (alphabetically ordered)
3. Standard library imports (`use std::...`)

Example from `crates/engine/src/renderer.rs`:
```rust
use crate::context::VkContext;
use crate::descriptor::{DescriptorWriter, GrowableAllocator, LayoutBuilder, PoolSizeRatio};
// ... more crate imports ...
use ash::{Device, vk};
use glm::Mat4;
use itertools::Itertools;
// ... more external imports ...
use std::mem;
use std::path::Path;
use std::sync::Arc;
```

`extern crate` is used only for aliasing: `extern crate nalgebra_glm as glm;` in `lib.rs`.

## Error Handling

This is a graphics/rendering application in early development. The error handling strategy is **panic-on-failure** for GPU initialization and Vulkan API calls:

- `.unwrap()` is used liberally for Vulkan API results — the assumption is that if Vulkan fails to initialize or execute, there is no reasonable recovery path
- `.expect("message")` is used when a failure needs a human-readable cause: `.expect("Failed to create ash entrypoint")`, `.expect("No suitable physical device found.")`
- `Result` is used at the boundary where it makes sense: `VkContext::create_instance` returns `Result<Instance, Box<dyn Error>>`
- `Option` is used for optional data: `meshes: Option<Vec<MeshAsset>>`, `load_gltf_meshes` returns `Option<Vec<MeshAsset>>`
- Pattern matching on `vk::Result` error variants is done in `GrowableAllocator::allocate` to handle pool exhaustion gracefully before falling back to `panic!`

```rust
// Typical pattern: unwrap on Vulkan initialization
let allocator = unsafe { vk_mem::Allocator::new(alloc_info) }.unwrap();

// Expect with message for critical failures
.expect("No suitable physical device found.")

// Graceful handling before panic fallback
Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
    // handle and retry
}
Err(e) => panic!("Failed to allocate descriptor set: {:?}", e),
```

There is **no custom error type** defined. `Box<dyn Error>` is used in the one function that returns `Result`.

## Logging

The `log` crate is used throughout the engine crate. The backend is `env_logger`, initialized in `main.rs`.

Log levels by usage:
- `log::trace!` — lifecycle events (start/end of Drop, resource creation)
- `log::debug!` — operational details (device selection, swapchain creation, shader loading, mesh loading)
- `log::info!` — informational (Vulkan debug INFO messages routed here)
- `log::warn!` — warnings (Vulkan debug WARNING messages)
- `log::error!` — errors (Vulkan debug ERROR messages)

Pattern for lifecycle logging:
```rust
log::trace!("Start: Dropping context");
// ... cleanup ...
log::trace!("End: Dropping context");
```

The application crate does not call log macros directly.

## Code Organization Patterns

**Module structure:** The `engine` crate is a flat set of modules under `src/`, with no nested submodules. Each module owns a distinct concern:
- `context.rs` — Vulkan instance, physical/logical device, surface
- `swapchain.rs` — swapchain creation, image views, semaphores
- `descriptor.rs` — descriptor set layouts, allocators, writers
- `pipeline.rs` — graphics pipeline builder
- `renderer.rs` — main renderer, frame loop, GPU resource management
- `primitives.rs` — GPU-side data structures (`Vertex`, push constants, scene data)
- `meshes.rs` — glTF loading, mesh asset type
- `debug.rs` — Vulkan validation layer setup
- `ui.rs` — egui integration (before_frame/render/after_frame functions)

**Initialization pattern:** Types use named constructor functions returning `Self`. No `Default` derives on GPU types (only on data-only application structs like `Application` in `main.rs`).

**Builder pattern:** `PipelineBuilder` and `LayoutBuilder` use mutable `&mut self` setters then a `build()` call — similar to Vulkan's own builder pattern from `ash`.

**Drop / cleanup:** Types that own GPU resources implement `Drop` (`VkContext`, `VulkanRenderer`) or provide an explicit `destroy()` method called manually before the resource goes out of scope. The `Drop` trait is used where RAII cleanup order is guaranteed (context, renderer). Explicit `destroy()` is used for types where the calling code controls cleanup order.

**`unsafe` usage:** `unsafe` blocks are used only for direct Vulkan FFI calls through `ash`. They are not wrapped in safe abstractions — callers are expected to uphold Vulkan's synchronization and lifetime requirements.

## Attributes

Common attributes observed:

```rust
#[allow(dead_code)]          // suppresses warnings for work-in-progress fields/methods
#[repr(C)]                   // on GPU-bound structs passed to shaders via push constants/UBOs
#[derive(Debug, Copy, Clone)]  // on plain data types
#[derive(Debug, Clone)]        // on heap-owning types where Copy isn't applicable
#[derive(Clone, Copy)]         // on small value types (indices, queue data)
#[derive(Default)]             // only on the application shell struct in main.rs
```

The file-level `#![allow(dead_code)]` in `lib.rs` suppresses dead code warnings globally for the engine crate while it is in active development.

## Comments

- Inline comments explain non-obvious Vulkan choices: `// Vulkan spec does not allow passing an array containing duplicated family indices.`
- Doc comments (`///`) are used selectively on public-ish functions, explaining behavior and panics:

```rust
/// Find a queue family with at least one graphics queue and one with
/// at least one presentation queue from `device`.
///
/// #Returns
///
/// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
fn find_queue_families(...) -> (Option<u32>, Option<u32>)
```

- Block comments are used in `pipeline.rs` to document the C struct being modeled, as a reference for Vulkan struct fields.
- `TODO` comments use format `// TODO: db: <description>` where `db` is the author's initials.
- Commented-out code is present in `main.rs` for input handling that is not yet implemented — this is acceptable during active development.

## Struct Layout Notes

GPU-facing structs use `#[repr(C)]` to match GLSL layout:
```rust
#[repr(C)]
pub struct GPUDrawPushConstants {
    pub world_matrix: [[f32; 4]; 4],
    pub vertex_buffer: vk::DeviceAddress,
}
```

Non-GPU structs do not use `#[repr(C)]`.

---

*Convention analysis: 2026-03-02*
