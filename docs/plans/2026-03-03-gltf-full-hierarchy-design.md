# Design: glTF Full Node-Hierarchy Traversal

**Date:** 2026-03-03
**File:** `crates/engine/src/meshes.rs`
**Status:** Approved

## Problem

The current `load_gltf_meshes` only visits scene root nodes. Meshes attached to child nodes (at any depth) are silently ignored. A full recursive traversal is needed to compose world transforms correctly for arbitrarily deep node hierarchies.

## Approach

DFS traversal collecting `(node, world_transform)` pairs. Traversal logic is isolated in two helpers; the vertex-baking loop is unchanged.

## Design

### Scene Root Selection

```rust
fn scene_roots(document: &gltf::Document) -> Vec<gltf::Node<'_>> {
    // 1. Default scene
    if let Some(scene) = document.default_scene() {
        return scene.nodes().collect();
    }
    // 2. First available scene
    if let Some(scene) = document.scenes().next() {
        return scene.nodes().collect();
    }
    // 3. No scenes — compute orphan roots (nodes unreferenced as any child)
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

Priority: default scene → first scene → orphan roots. The orphan-root fallback handles documents with no scenes at all (rare but valid glTF).

### `collect_mesh_nodes` Helper

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

`node_to_mat4` is unchanged — it extracts the node's local transform. World transform is composed as `parent_world * node_local` at each level.

A node with both a mesh and children emits its own entry and then recurses — both are processed, which matches glTF spec behaviour.

### Call Site in `load_gltf_meshes`

```rust
let mut mesh_nodes: Vec<(gltf::Node, glm::Mat4)> = Vec::new();
collect_mesh_nodes(
    scene_roots(&document).into_iter(),
    &glm::Mat4::identity(),
    &mut mesh_nodes,
);

for (node, transform) in mesh_nodes {
    let mesh = node.mesh().unwrap(); // safe: only mesh-bearing nodes collected
    let name = node
        .name()
        .or_else(|| mesh.name())
        .map(String::from)
        .unwrap_or_else(|| format!("node_{}", node.index()));
    // ... rest of loop unchanged
}
```

The `let nodes: Vec<gltf::Node>` block and the `for node in nodes` loop header are replaced by the above. Everything inside the loop (name resolution, primitive iteration, vertex baking with `transform` and `normal_matrix`) is untouched.

## Files Changed

- `crates/engine/src/meshes.rs` — add `scene_roots`, add `collect_mesh_nodes`, update call site in `load_gltf_meshes`
- No changes to other files, no new dependencies
