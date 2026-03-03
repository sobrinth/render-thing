---
phase: 01-mesh-loading-correctness
plan: 01
subsystem: mesh-loading
tags: [bug-fix, gltf, correctness, crash]
dependency_graph:
  requires: []
  provides: [correct-gltf-loading, safe-renderer-init]
  affects: [renderer, mesh-pipeline]
tech_stack:
  added: []
  patterns: [into_u32-for-index-normalization, map_err-error-logging, into_f32-uv-normalization]
key_files:
  modified:
    - crates/engine/src/meshes.rs
    - crates/engine/src/renderer.rs
decisions:
  - Default active_mesh to 0 — safe default; UI slider bounds selection after load (CRASH-01)
  - Retain gltf::accessor::Iter import — still used by ReadColors::RgbaF32 block at line 89
metrics:
  duration: ~15 minutes
  completed: 2026-03-02
---

# Phase 1 Plan 01: Mesh Loading Correctness Summary

Four correctness bugs fixed in the glTF mesh loading pipeline: safe active_mesh init (0 not 2), all index types via into_u32, TEXCOORD_0 UVs written to Vertex fields, and logged glTF import errors.

---

## Changes Made

### Fix 1 — CRASH-01: active_mesh default (renderer.rs line 154)

**File:** `crates/engine/src/renderer.rs`
**Commit:** e817bea

```rust
// BEFORE
active_mesh: 2,

// AFTER
active_mesh: 0,
```

**Why:** `active_mesh: 2` indexes `meshes[2]` unconditionally on first render. If the loaded glTF contains fewer than 3 meshes the program panics. Defaulting to 0 is safe; the egui slider bounds selection after load.

---

### Fix 2 — CLEAN-02: glTF import error logging (meshes.rs line 36)

**File:** `crates/engine/src/meshes.rs`
**Commit:** 8ecfd6a

```rust
// BEFORE
let (document, buffers, _images) = gltf::import(path).ok()?;

// AFTER
let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
    log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
}).ok()?;
```

**Why:** The bare `.ok()?` silently dropped I/O and parse errors. `map_err` logs the error before discarding it. `&path` (not `path`) is passed so the closure can still call `path.as_ref().display()` without a move conflict.

---

### Fix 3 — CRASH-02: Index type handling (meshes.rs lines 56-62)

**File:** `crates/engine/src/meshes.rs`
**Commit:** 846913e

```rust
// BEFORE
if let Some(gltf::mesh::util::ReadIndices::U16(Iter::Standard(iter))) =
    reader.read_indices()
{
    for v in iter {
        indices.push(v as u32);
    }
}

// AFTER
match reader.read_indices() {
    Some(read_indices) => {
        indices = read_indices.into_u32().collect();
    }
    None => {
        log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
        continue;
    }
}
```

**Why:** The original code only matched U16 Standard (non-sparse) indices. U8 and U32 index buffers would silently produce an empty index list, yielding zero-triangle draws. `into_u32()` handles U8, U16, and U32 in both Standard and Sparse accessor layouts. The `None` arm replaces the silent skip with a logged warning.

---

### Fix 4 — CRASH-03: UV channel and vertex UV assignment (meshes.rs lines 79-86, 108-114)

**File:** `crates/engine/src/meshes.rs`
**Commit:** 3009d08

**Part A — UV collection:**
```rust
// BEFORE
let mut uvs = Vec::new();
if let Some(gltf::mesh::util::ReadTexCoords::F32(Iter::Standard(iter))) =
    reader.read_tex_coords(1)
{
    for v in iter {
        uvs.push(v);
    }
}

// AFTER
let uvs: Vec<[f32; 2]> = reader
    .read_tex_coords(0)
    .map(|tc| tc.into_f32().collect())
    .unwrap_or_default();
```

**Why:** Channel 1 is the lightmap UV set; most assets only export TEXCOORD_0 (channel 0). `into_f32()` normalises U8/U16 storage formats. `unwrap_or_default()` is correct for untextured meshes.

**Part B — Vertex construction:**
```rust
// BEFORE
let vtx = Vertex {
    position: *v,
    normal,
    color,
    uv_x: 0.0,
    uv_y: 0.0,
};

// AFTER
let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
let vtx = Vertex {
    position: *v,
    normal,
    color,
    uv_x,
    uv_y,
};
```

**Why:** The hardcoded 0.0 meant texture coordinates were never passed to the GPU, so UV-mapped surfaces would always sample from UV (0,0). `uvs.get(i)` is bounds-safe if the UV array is shorter than the position array.

---

## Deviations from Plan

**Deviation 1: `use gltf::accessor::Iter` import not removed**

The plan instructed removing the `use gltf::accessor::Iter;` import after Task 3, conditional on confirming it was not referenced elsewhere. The import IS still used at line 89:

```rust
if let Some(gltf::mesh::util::ReadColors::RgbaF32(Iter::Standard(iter))) =
    reader.read_colors(0)
```

The colors block was explicitly called out as out of scope for Phase 1 (PLAN.md Task 5 notes). Removing the `Iter` import without fixing the colors block would cause a compile error. The import was therefore retained. This is the correct outcome per plan instructions ("confirm the `Iter` type is not referenced elsewhere in meshes.rs before removing").

No other deviations. All four tasks executed exactly as specified.

---

## Final cargo build Output (last 3 lines)

```
   Compiling engine v0.0.0 (H:\vulkan-rust\crates\engine)
   Compiling render-thing v0.1.0 (H:\vulkan-rust\crates\application)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 13.03s
```

Build exit code: 0.

---

## Requirement Status

| REQ-ID   | Status | Evidence |
|----------|--------|---------|
| CRASH-01 | PASS   | `active_mesh: 2` not found in renderer.rs; `active_mesh: 0` at line 154 |
| CRASH-02 | PASS   | `into_u32().collect()` present; `ReadIndices::U16(Iter::Standard` removed |
| CRASH-03 | PASS   | `read_tex_coords(0)` present; `read_tex_coords(1)` absent; `uvs.get(i)` in vertex loop |
| CLEAN-02 | PASS   | `map_err` with `log::error!` present; bare `gltf::import(path).ok()` absent |

---

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 — CRASH-01 | e817bea | fix(01-01): default active_mesh to 0 instead of 2 |
| Task 2 — CLEAN-02 | 8ecfd6a | fix(01-02): log error on glTF import failure instead of silent discard |
| Task 3 — CRASH-02 | 846913e | fix(01-03): handle all index types via into_u32(), warn on non-indexed |
| Task 4 — CRASH-03 | 3009d08 | fix(01-04): read TEXCOORD_0 and write UVs into Vertex fields |

## Self-Check: PASSED
