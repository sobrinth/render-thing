# glTF Full Node-Hierarchy Traversal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `load_gltf_meshes` to visit all nodes at any depth in the glTF node tree, composing world transforms correctly so deeply-nested meshes appear at the right position.

**Architecture:** Two private helpers are added to `meshes.rs`. `scene_roots` selects the correct traversal entry points (default scene → first scene → orphan roots). `collect_mesh_nodes` does a recursive DFS from those roots, composing `parent_world * node_local` at each level and collecting only mesh-bearing nodes into a flat `Vec<(gltf::Node, glm::Mat4)>`. The main processing loop iterates that flat list — everything inside the loop (name resolution, vertex baking) is unchanged.

**Tech Stack:** Rust, `gltf` crate, `nalgebra_glm` aliased as `glm` (already imported in `meshes.rs`)

**Design doc:** `docs/plans/2026-03-03-gltf-full-hierarchy-design.md`

---

### Task 1: Add `scene_roots` helper

**Files:**
- Modify: `crates/engine/src/meshes.rs`

**Background:**

The current call site uses:
```rust
let nodes: Vec<gltf::Node> = document
    .default_scene()
    .map(|s| s.nodes().collect())
    .unwrap_or_else(|| document.nodes().collect());
```

`document.nodes()` returns ALL nodes flat — including non-root ones — so it would double-count children during recursive traversal. `scene_roots` fixes this with a three-level fallback.

**Step 1: Add `scene_roots` after `node_to_mat4`**

In `crates/engine/src/meshes.rs`, add this function directly after `node_to_mat4` (before `load_gltf_meshes`):

```rust
fn scene_roots(document: &gltf::Document) -> Vec<gltf::Node<'_>> {
    if let Some(scene) = document.default_scene() {
        return scene.nodes().collect();
    }
    if let Some(scene) = document.scenes().next() {
        return scene.nodes().collect();
    }
    // No scenes at all — find nodes that are not referenced as any child
    let child_indices: std::collections::HashSet<usize> = document
        .nodes()
        .flat_map(|n| n.children().map(|c| c.index()))
        .collect();
    document
        .nodes()
        .filter(|n| !child_indices.contains(&n.index()))
        .collect()
}
```

**Step 2: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Compiles without errors. `scene_roots` will show an unused-function warning until Task 3 — that's fine.

**Step 3: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): add scene_roots helper for glTF traversal entry points"
```

---

### Task 2: Add `collect_mesh_nodes` helper

**Files:**
- Modify: `crates/engine/src/meshes.rs`

**Background:**

This function performs a DFS from a set of root nodes. At each node it:
1. Computes `world = parent_world * node_local`
2. If the node has a mesh, pushes `(node.clone(), world)` to `out`
3. Recurses into the node's children with `world` as the new parent

`node.clone()` is a cheap index-only clone (the `gltf::Node` type is a thin wrapper around an index into the document).

**Step 1: Add `collect_mesh_nodes` after `scene_roots`**

```rust
fn collect_mesh_nodes<'a>(
    nodes: impl Iterator<Item = gltf::Node<'a>>,
    parent_world: &glm::Mat4,
    out: &mut Vec<(gltf::Node<'a>, glm::Mat4)>,
) {
    for node in nodes {
        let world = parent_world * node_to_mat4(&node);
        if node.mesh().is_some() {
            out.push((node.clone(), world));
        }
        collect_mesh_nodes(node.children(), &world, out);
    }
}
```

**Step 2: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Compiles without errors or warnings (aside from the `scene_roots` unused warning from Task 1, which disappears once Task 3 lands).

**Step 3: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): add collect_mesh_nodes DFS helper for full hierarchy traversal"
```

---

### Task 3: Update `load_gltf_meshes` call site

**Files:**
- Modify: `crates/engine/src/meshes.rs`

**Background:**

The current call site (starting at the `let nodes:` line) is:

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
        let mut surfaces = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let transform = node_to_mat4(&node);
        let normal_matrix = glm::transpose(&glm::inverse(&transform));
```

Replace from `let nodes:` through `let normal_matrix =` with the new call site. Everything after `let normal_matrix =` (the primitive loop, vertex baking, upload) stays exactly as-is.

**Step 1: Replace the call site**

Find and replace this block:

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
        let mut surfaces = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let transform = node_to_mat4(&node);
        let normal_matrix = glm::transpose(&glm::inverse(&transform));
```

With:

```rust
    let mut mesh_nodes: Vec<(gltf::Node, glm::Mat4)> = Vec::new();
    collect_mesh_nodes(
        scene_roots(&document).into_iter(),
        &glm::Mat4::identity(),
        &mut mesh_nodes,
    );

    for (node, transform) in mesh_nodes {
        let mesh = node.mesh().unwrap(); // safe: collect_mesh_nodes only pushes mesh-bearing nodes

        let name = node
            .name()
            .or_else(|| mesh.name())
            .map(String::from)
            .unwrap_or_else(|| format!("node_{}", node.index()));
        let mut surfaces = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let normal_matrix = glm::transpose(&glm::inverse(&transform));
```

**Step 2: Verify it compiles**

```bash
cargo build -p engine
```

Expected: Clean compile, no warnings.

**Step 3: Visual verification**

Run the renderer with `abbey.glb` (or any glTF with nested nodes). Verify:
- All meshes appear at correct world-space positions
- Normal colour debug (`OVERRIDE_COLORS = true`) shows smooth world-space normals

**Step 4: Commit**

```bash
git add crates/engine/src/meshes.rs
git commit -m "feat(meshes): traverse full glTF node hierarchy for mesh loading"
```
