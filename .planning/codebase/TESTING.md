# Testing Patterns

**Analysis Date:** 2026-03-02

## Testing Strategy Overview

This codebase has **no automated tests of any kind.** A search for `#[test]`, `#[cfg(test)]`, and `mod tests` across all `.rs` files returns zero results. There are no test modules, no test files, and no testing dependencies in any `Cargo.toml`.

This is consistent with the nature of the project: a Vulkan rendering engine whose primary output is visual — a rendered window displaying 3D geometry and compute shader effects. Unit testing GPU rendering code requires significant infrastructure (headless Vulkan contexts, image comparison, etc.) that has not been set up here.

## Test Framework

**Runner:** None configured.

**Assertion Library:** None.

**Dev Dependencies:** None in any crate (`engine`, `assets`, `application`).

**Run Commands:**
```bash
cargo test    # runs, but no tests exist — output: "running 0 tests"
```

## Test File Organization

No test files exist. Patterns to follow if tests are added:

- Co-locate unit tests in the same `.rs` file using `#[cfg(test)] mod tests { ... }` at the bottom
- Place integration tests in `crates/<crate>/tests/` directories
- Name test files by the module they test: `tests/descriptor_tests.rs`, `tests/swapchain_tests.rs`

## CI/CD

No CI pipeline exists. There is no `.github/` directory, no `Makefile`, no `justfile`, and no other automation configuration. All building and testing is done manually by the developer.

## Manual Testing Approach

Testing is visual and interactive. The developer runs the application binary and observes the rendered output:

```bash
cargo run                    # build and run with dev profile (includes validation layers)
cargo run --release          # build and run with optimizations
```

**What gets exercised on each run:**
- Vulkan device selection and initialization (`crates/engine/src/context.rs`)
- Swapchain creation and presentation (`crates/engine/src/swapchain.rs`)
- Shader compilation (via `crates/assets/build.rs` build script using `glslangValidator`)
- glTF mesh loading (`crates/engine/src/meshes.rs`)
- Compute and graphics pipeline creation (`crates/engine/src/pipeline.rs`, `renderer.rs`)
- Frame rendering loop (draw background, draw geometry, draw UI)
- egui debug overlay with live controls

**Vulkan Validation Layers:**

The engine enables Vulkan validation layers automatically in debug builds (`cfg!(debug_assertions)`). Validation messages are routed through the `log` crate:

```rust
// In debug.rs — validation layer output goes to log
Flag::VERBOSE => log::debug!("{:?} - {:?}", typ, message),
Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
_ => log::error!("{:?} - {:?}", typ, message),
```

To see validation output, set the `RUST_LOG` environment variable:
```bash
RUST_LOG=debug cargo run        # see validation layer debug output
RUST_LOG=warn cargo run         # see only warnings and errors
```

Validation layers check for:
- Incorrect Vulkan API usage
- Missing synchronization
- Resource lifetime violations
- Performance warnings

**Shader Compilation (Build-Time Testing):**

The `crates/assets/build.rs` build script compiles all GLSL shaders to SPIR-V using `glslangValidator` at build time. A shader compilation failure causes `panic!` and fails the build, providing a form of compile-time shader validation:

```bash
cargo build    # will fail with shader errors if GLSL is invalid
```

Shaders that are compiled:
- `assets/shaders/shader.vert` / `shader.frag`
- `assets/shaders/colored_triangle.vert` / `colored_triangle.frag`
- `assets/shaders/colored_triangle_mesh.vert`
- `assets/shaders/gradient.comp`
- `assets/shaders/gradient_color.comp`
- `assets/shaders/sky.comp`

To skip shader recompilation during iteration:
```bash
SKIP_SHADER_COMPILATION=true cargo build
```

## Testing Limitations for Graphics Code

The main barriers to automated testing in this codebase:

1. **Vulkan requires a GPU** — `VkContext::initialize` calls `Entry::load()` and will fail without a Vulkan-capable GPU and driver. Headless testing requires a software Vulkan implementation (e.g., `lavapipe`) or GPU mocking.

2. **Visual correctness is hard to assert** — the correct output is "does the frame look right," which requires image comparison with reference screenshots or human review.

3. **GPU memory lifetimes** — `AllocatedBuffer`, `AllocatedImage` and other GPU resources must be cleaned up before the Vulkan device is destroyed. Testing resource lifecycle correctness is complex without a full rendering context.

4. **Frame-dependent state** — many bugs only manifest after several frames (e.g., descriptor pool exhaustion, swapchain resizing, semaphore signaling).

## What Could Be Unit Tested

If tests are added in future, these components are most amenable:

- `crates/engine/src/swapchain.rs` — `choose_swapchain_surface_format`, `choose_swapchain_present_mode`, `choose_swapchain_extent` are pure functions with no Vulkan dependency
- `crates/engine/src/primitives.rs` — `Vertex::default()` construction
- `crates/assets/build.rs` logic — path construction and shader enumeration (would need refactoring to be testable)
- Future pure math/geometry utilities if added to `primitives.rs`

Example of a function that is already unit-testable without GPU:
```rust
// swapchain.rs — pure logic, no Vulkan FFI
fn choose_swapchain_extent(
    capabilities: vk::SurfaceCapabilitiesKHR,
    preferred_dimensions: [u32; 2],
) -> vk::Extent2D
```

---

*Testing analysis: 2026-03-02*
