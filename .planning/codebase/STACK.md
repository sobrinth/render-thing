# Technology Stack

**Analysis Date:** 2026-03-02

## Languages

**Primary:**
- Rust (nightly channel) - All application and engine code
- GLSL - GPU shader programs in `assets/shaders/`

**Secondary:**
- None

## Runtime

**Environment:**
- Rust nightly toolchain (pinned via `rust-toolchain.toml`)
- No async runtime - single-threaded event loop via winit

**Package Manager:**
- Cargo (workspace layout, resolver v3)
- Lockfile: present (`Cargo.lock`)

## Workspace Structure

The project is a Cargo workspace with resolver v3 (Rust 2024 edition features).

```
Cargo.toml          # Workspace root, all dependency versions pinned here
crates/
  application/      # Binary crate: windowing, event loop, entry point (pkg: render-thing)
  engine/           # Library crate: all Vulkan/rendering logic
  assets/           # Library crate: empty stub; build.rs compiles GLSL shaders
```

All crates share `edition = "2024"` and `version = "0.1.0"` from `[workspace.package]`.

## Frameworks

**Core:**
- `ash` 0.38.0 (resolved: 0.38.0+1.3.281) - Raw Vulkan bindings for Rust; unsafe FFI to libvulkan
- `winit` 0.30 (resolved: 0.52.6) - Cross-platform window creation and event loop
- `egui` 0.33.2 - Immediate-mode GUI framework
- `vk-mem` 0.5.0 (resolved: 0.9.5) - Vulkan Memory Allocator (VMA) bindings for GPU memory management

**Windowing/Surface:**
- `ash-window` 0.13 (resolved: 0.38.0+1.3.281) - Bridges winit raw window handles to Vulkan surfaces

**UI Rendering:**
- `egui-ash-renderer` 0.11.0 - Renders egui draw primitives using Vulkan dynamic rendering + vk-mem
- `egui-winit` 0.33.2 - Translates winit window events into egui input

**Math:**
- `nalgebra-glm` 0.20.0 (resolved: 0.34.1 nalgebra, 0.34.1 nalgebra-glm aliased as `glm`) - GLSL-compatible linear algebra (Mat4, Vec3, etc.)

**Asset Loading:**
- `gltf` 1.4.1 with `utils` feature - glTF 2.0 mesh/scene import

**Utilities:**
- `log` 0.4 (resolved: 0.4.14) - Logging facade
- `env_logger` 0.11 (resolved: 0.1.3) - Env-var-driven log backend; init via `env_logger::init()`
- `itertools` 0.14.0 (resolved: 1.70.1) - Iterator adapters (`collect_vec`, `Itertools` trait)

## Key Feature Flags

`egui-ash-renderer` is configured with non-default features:
```toml
egui-ash-renderer = { version = "0.11.0", default-features = false, features = [
    "dynamic-rendering",   # uses VK_KHR_dynamic_rendering instead of render passes
    "vk-mem"               # use VMA allocator for egui texture buffers
] }
```

This means the renderer avoids legacy Vulkan render pass objects entirely.

## Toolchain Configuration

From `rust-toolchain.toml`:
- **Channel:** nightly (required - uses Rust 2024 edition features)
- **Components:** rustfmt, clippy
- **Profile:** minimal
- **Target triples:**
  - `aarch64-apple-darwin` (Apple Silicon macOS)
  - `x86_64-unknown-linux-gnu` (Linux x86_64)
  - `x86_64-pc-windows-msvc` (Windows x86_64, MSVC ABI)

## Code Style / Formatting

From `rustfmt.toml`:
- `edition = "2024"`
- `newline_style = "Unix"` (LF line endings enforced)

## Cargo Environment

From `.cargo/config.toml`:
- `CARGO_WORKSPACE_DIR` is set as a relative env var - used in `assets/build.rs` to locate the shader source directory at build time (`env!("CARGO_WORKSPACE_DIR")`)

## Dependency Auditing

`cargo-deny` is configured via `deny.toml`:
- Allowed licenses: MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, Unicode-3.0, ISC
- Sources restricted to crates.io only (no git deps)
- Multiple crate versions allowed (not denied)
- Duplicate detection configured for the three target triples

## Build Requirements (External Tools)

The `assets` build script (`crates/assets/build.rs`) invokes `glslangValidator` as an external command-line tool at build time. This must be installed and on `PATH`:
- **`glslangValidator`** - Compiles `.vert`, `.frag`, `.comp` GLSL source files to SPIR-V `.spv` binaries
- Skip compilation with env var: `SKIP_SHADER_COMPILATION=true`

Shaders are tracked for incremental rebuild via `cargo::rerun-if-changed` directives.

## Platform Requirements

**Development:**
- Rust nightly toolchain
- Vulkan SDK (provides `glslangValidator` and runtime Vulkan loader)
- A GPU with Vulkan 1.3 driver support

**Production:**
- Vulkan 1.3-capable GPU driver required
- Targets: macOS (Metal via MoltenVK implied by `aarch64-apple-darwin`), Linux, Windows MSVC

---

*Stack analysis: 2026-03-02*
