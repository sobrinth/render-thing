# Phase 1 Context — Mesh Loading Correctness

**Phase goal:** glTF meshes load and render correctly — correct index types, correct UVs, safe mesh selection, and visible errors on bad assets
**Requirements:** CRASH-01, CRASH-02, CRASH-03, CLEAN-02

---

## Decisions

### CRASH-01: active_mesh initialization

**Decision:** Initialize `active_mesh` to `0` unconditionally.

- File: `crates/engine/src/renderer.rs:154`
- Change `active_mesh: 2` → `active_mesh: 0`
- No dynamic clamping needed; the UI slider handles selection after load

---

### CRASH-02: glTF index type handling

**Decision:** Handle all three index types (U8, U16, U32) by casting to `u32`. Skip non-indexed primitives with a `log::warn!`.

- File: `crates/engine/src/meshes.rs:56–62`
- Replace the single `if let U16` with a `match` on `reader.read_indices()`:
  - `ReadIndices::U8(iter)` → cast each to `u32`
  - `ReadIndices::U16(iter)` → cast each to `u32`
  - `ReadIndices::U32(iter)` → use directly
  - `None` (non-indexed primitive) → `log::warn!("Skipping non-indexed primitive in mesh '{}'", name)` and `continue` to next primitive
- Do not attempt to generate sequential indices for non-indexed primitives (out of scope)

---

### CRASH-03: UV channel and vertex assignment

**Decision:** Read from channel 0. Write UV data into the vertex struct. Silently fall back to `(0.0, 0.0)` when channel 0 is absent.

- File: `crates/engine/src/meshes.rs:80` and `meshes.rs:108–114`
- Change `reader.read_tex_coords(1)` → `reader.read_tex_coords(0)`
- In the vertex-building loop, replace `uv_x: 0.0, uv_y: 0.0` with values from `uvs`:
  ```rust
  let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
  ```
- No log needed when channel 0 is absent (normal for untextured meshes)

---

### CLEAN-02: glTF load error logging

**Decision:** Use `log::error!` with both the path and the error detail before discarding.

- File: `crates/engine/src/meshes.rs:36`
- Replace `gltf::import(path).ok()?` with explicit error capture:
  ```rust
  let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
      log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
  }).ok()?;
  ```
- Rationale: a missing/corrupt asset is a user-visible error, not a warning. Including the path and error detail is consistent with the existing `log::debug!` call above it.

---

## Out-of-Scope (noted, not acted on)

- **Multi-primitive mesh bug:** `vertices.clear()` and `indices.clear()` run inside the primitive loop, so only the last primitive's geometry is uploaded per mesh. This is a correctness issue but is not in Phase 1's requirements. Defer to a future phase.
- **Non-indexed primitive support:** Generating sequential indices for non-indexed gltf primitives is deferred. Phase 1 logs a warning and skips.

---

## Code Context

| File | Lines | Relevance |
|------|-------|-----------|
| `crates/engine/src/renderer.rs` | 154 | `active_mesh: 2` initialization |
| `crates/engine/src/meshes.rs` | 36 | `gltf::import` silent error discard |
| `crates/engine/src/meshes.rs` | 56–62 | U16-only index reader |
| `crates/engine/src/meshes.rs` | 80 | UV read from wrong channel (1 instead of 0) |
| `crates/engine/src/meshes.rs` | 108–114 | Vertex UV fields hardcoded to 0.0 (uvs vec never used) |
