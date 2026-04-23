# Module: meshes.rs

## Purpose
glTF mesh loading and GPU upload. Parses glTF files, transforms vertices to world space, collects indices and surfaces, uploads via renderer.upload_mesh(). Handles missing normals/UVs/colors gracefully.

## Key Types
- **MeshAsset**: Holds mesh name, surfaces (index ranges), GPU mesh buffers (vertex, index, device address).
- **GeoSurface**: Index range (start, count) for a sub-mesh / primitive.

## Notable Patterns
- **Scene graph traversal**: collect_mesh_nodes() recursively walks glTF node hierarchy, accumulating world transforms
- **Deferred GPU upload**: Mesh loaded into CPU-side vectors, then submitted via immediate submit once
- **Fallback defaults**: Missing normals → [1,0,0], missing UVs → (0,0), missing colors → white [1,1,1,1]
- **Override colors**: OVERRIDE_COLORS flag (line 166) sets vertex color to normal (useful for debug visualization)
- **Transform handling**: World matrices applied; normal matrix (transpose of inverse) for normal transformation

## Unsafe Blocks
Line 81: node.mesh().unwrap() - safe; collect_mesh_nodes guarantees node.mesh().is_some()

## Dependencies
- primitives (Vertex, GPUMeshBuffers)
- renderer (VulkanRenderer::upload_mesh reference)
- gltf (mesh parsing)
- nalgebra_glm (matrix math for transforms)
