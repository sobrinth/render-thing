# Rust Vulkan Renderer - Project Overview

## Workspace Structure

Monorepo with 3 crates under `crates/`:

1. **engine** (main renderer library)
   - Core Vulkan renderer with frame management, pipeline, descriptor allocation
   - Handles window integration, synchronization, resource upload
   - Exposes public `Engine` API for initialization and frame drawing

2. **application** (binary executable)
   - Winit event loop integration
   - Input event bridging (mouse, keyboard, resize)
   - egui UI state management
   - Entry point: calls `Engine::initialize()` and `draw()` per frame

3. **assets** (placeholder for shader/model resources)
   - Currently: glTF mesh files, compiled SPIR-V shaders

## Build & Run

```bash
cargo build --release
cargo run --bin render-thing
```

Workspace uses Rust edition 2024, Vulkan API 1.3.

## Key Architectural Decisions

**Double/Triple Buffering:** `FRAME_OVERLAP = 2` constant enables GPU/CPU pipelining; frame N executes while frame N-1 renders. Each frame has its own command buffer, fence, semaphore, and per-frame descriptor allocator.

**Typestate Command Buffer Pattern:** `CommandBuffer<S>` uses phantom types (Initial, Recording, Executable, Submitted) to enforce state machine at compile time. Prevents resubmitting or resetting in-flight buffers.

**Manual Vulkan Resource Lifecycle:** `ManuallyDrop` wrappers for `VkContext` and `RendererResources` ensure custom drop order: resources dropped before device/instance.

**Immediate Submit Abstraction:** `ImmediateSubmitData` encapsulates one-off GPU work (mesh uploads, image transitions) without explicit synchronization in calling code.

**Dynamic Rendering:** Uses Vulkan's dynamic rendering instead of renderpasses; descriptor sets sized per-frame.

**Growable Descriptor Allocator:** `GrowableAllocator` expands pool count on exhaustion, avoiding pre-allocation guessing.

## Vulkan Initialization & Frame Loop

### Initialization (Single)

1. Create Vulkan instance (with debug layer in debug mode)
2. Create surface from window handle
3. Select physical device (requirements: graphics queue, swapchain support, sampler anisotropy)
4. Create logical device with Vulkan 1.3 features: dynamic rendering, synchronization2, buffer device address
5. Create vk-mem allocator with BUFFER_DEVICE_ADDRESS enabled
6. Create swapchain (format: B8G8R8A8_UNORM/SRGB, present mode: FIFO)
7. Initialize 2 frame data structures with pools, fences, semaphores
8. Create compute effect pipelines (3 background shaders: gradient, gradient_color, sky)
9. Create mesh pipeline (vertex + fragment shaders with depth test)
10. Load glTF mesh (basicmesh.glb) with immediate submit
11. Create default 1x1 textures (white, grey, black, checkerboard) for fallback sampling

### Frame Loop (Per-frame)

1. **Resize check:** If window resized, recreate swapchain and return early
2. **GPU sync:** Wait on frame's render fence (1s timeout) to ensure last frame done
3. **Acquire next image:** Request next swapchain image with acquire semaphore
4. **Reset command buffer:** Reset primary command buffer from pool
5. **Draw background:** Dispatch compute shader to fill draw image (16x16 workgroups)
6. **Draw geometry:** Render mesh with material texture, depth test enabled
7. **Copy to swapchain:** Blit draw image → swapchain image
8. **Render UI:** egui primitives onto swapchain with dynamic rendering
9. **Submit:** Queue submit with wait/signal semaphore pair
10. **Present:** vkQueuePresentKHR with present semaphore
11. **UI cleanup:** Free textures marked for deletion from previous frame

**Image layout transitions:** UNDEFINED → GENERAL (compute) → COLOR_ATTACHMENT → TRANSFER_SRC → TRANSFER_DST → PRESENT
**Synchronization:** All-stages barriers (inefficient but safe) used; comments note room for optimization.

## Platform Compatibility (Windows vs Unix)

- **ash-window:** Abstracts surface creation via `HasDisplayHandle`/`HasWindowHandle` traits → works on Windows, Linux, macOS
- **Shader files:** Assume UNIX-style paths (`assets/shaders/*.comp.spv`). On Windows, forward slashes still work in Rust `File::open()`
- **Input layer:** winit abstracts OS input events; engine layer receives normalized key/mouse events
- **No Windows-specific code:** Renderer is platform-agnostic; example application uses winit which is cross-platform

## Hazards & Unsafe Blocks

All unsafe blocks are necessary for Vulkan FFI; they are scoped tightly:

- Command buffer wrapping (frame.rs line 95): requires waiting on fence before reset
- Pointer dereference for push constants (frame.rs line 317): static lifetime math structs
- Descriptor set updates (descriptor.rs line 304): builders keep lifetime; write calls happen immediately
- Image memory mapping (resources.rs line 204): paired with unmap; short-lived pointers
- Device address queries (resources.rs line 298): raw pointer cast for device address storage

No null pointer dereferences, no use-after-free; Rust's borrow checker prevents most errors except Vulkan state invariants (enforced by fences/semaphores).
