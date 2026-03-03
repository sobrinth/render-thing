# Design: glTF Node-Aware Mesh Loading

**Date:** 2026-03-03
**File:** `crates/engine/src/meshes.rs`
**Status:** Approved

## Problem

`load_gltf_meshes` iterates over `document.meshes()` directly, ignoring the node hierarchy. All vertex positions are loaded in the mesh's local coordinate system. Meshes placed at non-identity positions/rotations/scales in a scene appear at the wrong location.

The TODO at line 41 acknowledged this: `// TODO: Should probably load meshes relative to nodes... due to transform`

## Approach

Option A: iterate scene nodes, apply per-node local transform, bake into vertices. A `node_to_mat4` helper isolates the transform logic for future upgrade to full hierarchy traversal.

## Design

### Iteration Strategy

Replace `for mesh in document.meshes()` with iteration over nodes:

1. Prefer `document.default_scene().nodes()` — uses the scene the author designated as default
2. Fall back to `document.nodes()` if no default scene exists
3. Skip nodes where `node.mesh()` is `None`

One `MeshAsset` is created **per node** (not per mesh definition). If two nodes reference the same mesh definition with different transforms, they become two separate `MeshAsset`s with different baked vertex data. This is correct for the flat/baked approach.

### Transform Helper

```rust
fn node_to_mat4(node: &gltf::Node) -> glm::Mat4 {
    let m = node.transform().matrix();
    glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ])
}
```

**Upgrade path to full hierarchy:** Replace `node_to_mat4` with a function that receives a `&HashMap<usize, glm::Mat4>` (node index → world transform) built by a recursive pre-pass. The call site in `load_gltf_meshes` does not change.

### Vertex Transform Baking

Two vertex attributes require transformation:

**Position** (affine point transform):
```rust
let world_pos = transform * glm::vec4(x, y, z, 1.0);
// use world_pos.x, world_pos.y, world_pos.z
```

**Normal** (inverse-transpose, direction not point):
```rust
let normal_matrix = glm::transpose(&glm::inverse(&transform));
let world_normal = normal_matrix * glm::vec4(nx, ny, nz, 0.0);
let normal = glm::normalize(&world_normal.xyz());
```

Using `w=0` for normals and `w=1` for positions handles the affine/direction distinction without extracting the upper-3×3 submatrix.

The `OVERRIDE_COLORS` debug block (colors ← normals) applies **after** baking, so it shows world-space normals.

### MeshAsset Naming

Priority:
1. `node.name()` — preferred; nodes are usually named in DCC tools
2. `mesh.name()` — fallback if node has no name
3. `format!("node_{}", node.index())` — last resort

Removes the current silent skip (`mesh.name()?`) on unnamed meshes.

## Upgrade Path to Full Hierarchy

When upgrading from flat to recursive:

1. Add a `fn build_world_transforms(document: &gltf::Document) -> HashMap<usize, glm::Mat4>` that walks from root nodes, composing `parent_world * node_local`
2. Replace `node_to_mat4(&node)` call with a lookup in the pre-built map
3. No changes required to the vertex baking, naming, or `MeshAsset` structures

## Files Changed

- `crates/engine/src/meshes.rs` — primary change (iteration, helper, baking)
- No changes to `renderer.rs`, `primitives.rs`, or shaders
- No new dependencies
