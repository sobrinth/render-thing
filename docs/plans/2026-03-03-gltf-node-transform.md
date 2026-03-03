# glTF Node-Aware Mesh Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `load_gltf_meshes` to iterate over glTF scene nodes instead of raw mesh definitions, applying each node's local transform to vertex positions and normals before GPU upload.

**Architecture:** Switch the outer loop from `document.meshes()` to `document.default_scene().nodes()` (falling back to `document.nodes()`). Extract a `node_to_mat4` helper for the local→world transform. Bake position and normal transforms into the vertex buffer at load time using `nalgebra_glm`.

**Tech Stack:** Rust, `gltf` crate (already in use), `nalgebra_glm 0.20` aliased as `glm` (already in use via `extern crate nalgebra_glm as glm` in `lib.rs`)

**Design doc:** `docs/plans/2026-03-03-gltf-node-transform-design.md`

---

### Task 1: Add `node_to_mat4` helper

**Files:**
- Modify: `crates/engine/src/meshes.rs`

**Background:**

The glTF spec stores 4×4 matrices in **column-major** order. The gltf crate exposes them as `[[f32; 4]; 4]` where `m[column][row]`. `glm::Mat4::from_column_slice` expects a flat `&[f32]` in the same column-major order: `[col0_row0, col0_row1, col0_row2, col0_row3, col1_row0, ...]`. So passing `m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], ...` is correct.

The `glm` alias is already declared in `lib.rs` as `extern crate nalgebra_glm as glm`. In a submodule like `meshes.rs`, access it as `crate::glm` or just add `use nalgebra_glm as glm;` locally. The simplest approach is a local `use`:

**Step 1: Add import at top of meshes.rs**

Add after the existing `use` block at the top of `crates/engine/src/meshes.rs`:

```rust
use nalgebra_glm as glm;
```

The existing imports at the top look like:
```rust
use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::renderer::VulkanRenderer;
use std::path::Path;
use std::sync::Arc;
use vk_mem::Allocator;
```

Add the `nalgebra_glm` line after the `vk_mem` import.

**Step 2: Add `node_to_mat4` helper function**

Add this private function after the `MeshAsset` impl block (before `load_gltf_meshes`):

```rust
fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
    let m = node.transform().matrix(); // [[f32; 4]; 4], column-major (m[col][row])
    glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}
```

**Step 3: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Compiles without errors or warnings about unused imports.

**Step 4: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): add node_to_mat4 helper for glTF node transforms"
```

---

### Task 2: Switch outer loop to node-based iteration

**Files:**
- Modify: `crates/engine/src/meshes.rs:42–111`

**Background:**

Currently the outer loop is:
```rust
for mesh in document.meshes() {
    let name = String::from(mesh.name()?);
    // ...
}
```

The new loop iterates nodes instead. `document.default_scene()` returns `Option<gltf::Scene>`. The scene's nodes are the scene roots — for flat hierarchies (the current target) these are the mesh nodes themselves. If no default scene exists, fall back to all document nodes.

`scene.nodes()` and `document.nodes()` return different concrete iterator types, so collect both into a `Vec<gltf::Node<'_>>` before the loop to avoid boxing.

**Step 1: Replace the outer `for mesh in document.meshes()` loop header**

Find this code (starting at line ~42):
```rust
    for mesh in document.meshes() {
        let name = String::from(mesh.name()?);
```

Replace with:
```rust
    let nodes: Vec<gltf::Node> = document
        .default_scene()
        .map(|s| s.nodes().collect())
        .unwrap_or_else(|| document.nodes().collect());

    for node in nodes {
        let Some(mesh) = node.mesh() else { continue };

        let name = node
            .name()
            .or_else(|| mesh.name())
            .map(String::from)
            .unwrap_or_else(|| format!("node_{}", node.index()));
```

**Step 2: Fix the closing brace count**

The `for mesh in document.meshes()` block ends at the bottom of the function just before the `if meshes.is_empty()` check. The `for node in nodes` block replaces it exactly — no brace count changes are needed since we just replaced the loop header and the `let name` line.

**Step 3: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Compiles without errors. There should be no warnings about the old `mesh.name()?` pattern since that's been replaced.

**Step 4: Run the renderer and visually verify**

Start the renderer. Meshes with identity transforms should look identical to before. The key change is that unnamed meshes no longer cause the whole mesh to be silently skipped.

**Step 5: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): iterate glTF scene nodes instead of raw mesh definitions"
```

---

### Task 3: Apply node transform to vertex positions and normals

**Files:**
- Modify: `crates/engine/src/meshes.rs` (inside the `for primitive in mesh.primitives()` loop)

**Background:**

After Task 2, the `transform` is available as `node_to_mat4(&node)` and must be applied to each vertex. Two attributes need different transform math:

- **Position** is an affine point: multiply with `w=1.0` so translation applies.
- **Normal** is a direction, not a point: multiply with the *inverse-transpose* of the transform using `w=0.0` so translation does not apply, and non-uniform scaling is handled correctly. The result must be re-normalized.

`glm::inverse` + `glm::transpose` are the standard functions. `glm::normalize` normalizes the xyz components. Access xyz from a `glm::Vec4` via `.xyz()`.

**Step 1: Compute transform matrices before the primitive loop**

Find this line inside `for node in nodes`:
```rust
        for primitive in mesh.primitives() {
```

Add two lines directly before it (after the `let name = ...` block):
```rust
        let transform = node_to_mat4(&node);
        let normal_matrix = glm::transpose(&glm::inverse(&transform));
```

**Step 2: Apply transforms inside the vertex construction closure**

Find the `positions.iter().enumerate().for_each(|(i, v)| {` closure. Currently it builds `vtx` like this:

```rust
                let vtx = Vertex {
                    position: *v,
                    normal,
                    color,
                    uv_x,
                    uv_y,
                };
```

Replace the lines from `let normal = if normals.is_empty()` through the `vertices.push(vtx)` with:

```rust
                let normal = if normals.is_empty() {
                    [1.0_f32, 0.0, 0.0]
                } else {
                    normals[i]
                };
                let color = if colors.is_empty() {
                    [1.0_f32, 1.0, 1.0, 1.0]
                } else {
                    colors[i]
                };
                let (uv_x, uv_y) = uvs
                    .get(i)
                    .copied()
                    .map(|v| (v[0], v[1]))
                    .unwrap_or((0.0, 0.0));

                let world_pos = transform * glm::vec4(v[0], v[1], v[2], 1.0);
                let world_normal_raw = normal_matrix * glm::vec4(normal[0], normal[1], normal[2], 0.0);
                let world_normal = glm::normalize(&world_normal_raw.xyz());

                let vtx = Vertex {
                    position: [world_pos.x, world_pos.y, world_pos.z],
                    normal: [world_normal.x, world_normal.y, world_normal.z],
                    color,
                    uv_x,
                    uv_y,
                };
                vertices.push(vtx);
```

**Step 3: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Clean compile, no errors.

**Step 4: Visual verification**

Run the renderer with a glTF that has non-identity node transforms. Verify:
- Meshes appear at their correct world-space positions
- The normal color debug visualization (`OVERRIDE_COLORS = true`) shows correct world-space normals — smooth transitions between orientations, not abrupt jumps

If testing with a model that only has identity transforms: behavior should be identical to before (the identity matrix transforms positions and normals to themselves).

**Step 5: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): bake glTF node transforms into vertex positions and normals"
```

---

### Task 4: Upgrade path smoke test (optional, documents future work)

This task is informational — no code change. Add a comment above `node_to_mat4` to document the upgrade path.

**Step 1: Add upgrade comment**

Above `fn node_to_mat4`:

```rust
// Returns the node's LOCAL transform as a glm::Mat4.
//
// Upgrade path to full hierarchy: replace this with a lookup into a
// `HashMap<usize, glm::Mat4>` (node_index -> world_transform) built by
// walking from scene roots and composing `parent_world * node_local`.
// The call site in `load_gltf_meshes` does not change.
fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
```

**Step 2: Compile and commit**

```bash
cargo build -p engine
git add crates/engine/src/meshes.rs
git commit -m "docs(meshes): document upgrade path to full node hierarchy transform"
```
