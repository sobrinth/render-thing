# Phase 1: Mesh Loading Correctness - Research

**Researched:** 2026-03-02
**Domain:** `gltf` crate 1.4.1 API — index reading, UV reading, error handling
**Confidence:** HIGH (all findings verified against docs.rs 1.4.1 and GitHub source)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**CRASH-01: active_mesh initialization**
Initialize `active_mesh` to `0` unconditionally.
- File: `crates/engine/src/renderer.rs:154`
- Change `active_mesh: 2` → `active_mesh: 0`
- No dynamic clamping needed; the UI slider handles selection after load

**CRASH-02: glTF index type handling**
Handle all three index types (U8, U16, U32) by casting to `u32`. Skip non-indexed primitives with a `log::warn!`.
- File: `crates/engine/src/meshes.rs:56–62`
- Replace the single `if let U16` with a `match` on `reader.read_indices()`:
  - `ReadIndices::U8(iter)` → cast each to `u32`
  - `ReadIndices::U16(iter)` → cast each to `u32`
  - `ReadIndices::U32(iter)` → use directly
  - `None` → `log::warn!("Skipping non-indexed primitive in mesh '{}'", name)` and `continue`
- Do not attempt to generate sequential indices for non-indexed primitives (out of scope)

**CRASH-03: UV channel and vertex assignment**
Read from channel 0. Write UV data into the vertex struct. Silently fall back to `(0.0, 0.0)` when channel 0 is absent.
- File: `crates/engine/src/meshes.rs:80` and `meshes.rs:108–114`
- Change `reader.read_tex_coords(1)` → `reader.read_tex_coords(0)`
- In the vertex-building loop, replace `uv_x: 0.0, uv_y: 0.0` with values from `uvs`:
  ```rust
  let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
  ```
- No log needed when channel 0 is absent (normal for untextured meshes)

**CLEAN-02: glTF load error logging**
Use `log::error!` with both the path and the error detail before discarding.
- File: `crates/engine/src/meshes.rs:36`
- Replace `gltf::import(path).ok()?` with explicit error capture:
  ```rust
  let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
      log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
  }).ok()?;
  ```

### Deferred Ideas (OUT OF SCOPE)
- **Multi-primitive mesh bug:** `vertices.clear()` and `indices.clear()` run inside the primitive loop, so only the last primitive's geometry is uploaded per mesh. Deferred to a future phase.
- **Non-indexed primitive support:** Generating sequential indices for non-indexed glTF primitives is deferred. Phase 1 logs a warning and skips.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CRASH-01 | `active_mesh: 2` panics when fewer than 3 meshes load | Simple initialization fix; no API research needed |
| CRASH-02 | U16-only index match silently produces zero indices for U8/U32 assets | `ReadIndices` enum verified: U8/U16/U32 variants exist; `into_u32()` confirmed as idiomatic API |
| CRASH-03 | UVs read from channel 1 (wrong) and never written into vertex struct | `read_tex_coords(set: u32)` confirmed; channel 0 is the standard base UV set |
| CLEAN-02 | `gltf::import(path).ok()?` silently discards the error string | `gltf::Error` implements `Display`; `{e}` in format string yields useful message |
</phase_requirements>

---

## Summary

The phase fixes four bugs in `crates/engine/src/meshes.rs` (and one line in `renderer.rs`). All bugs are straightforward API misuse issues with well-defined correct solutions in the `gltf` 1.4.1 API.

The most consequential bug is the U16-only index match (CRASH-02). The `gltf` crate provides `ReadIndices::into_u32()` — a single method that transparently casts U8, U16, or U32 indices to `u32`. Using `into_u32().collect()` is simpler and more correct than a manual `match`, and the CONTEXT.md decision to use a `match` is also valid (both approaches work). The `into_u32()` approach is more idiomatic.

The UV bug (CRASH-03) has two parts: the wrong channel number AND the fact that the collected `uvs` Vec is never actually read back into the vertex struct. Both must be fixed together.

**Primary recommendation:** Use `reader.read_indices().map(|ri| ri.into_u32().collect::<Vec<_>>())` for index reading — it handles all three types in one line and is the canonical approach in the gltf crate docs.

---

## Standard Stack

### Core (already in use)
| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `gltf` | 1.4.1 | glTF 2.0 asset loading | Already in Cargo.toml with `features = ["utils"]` |
| `log` | 0.4 | Structured logging | Already in use |

No new dependencies required for this phase.

---

## Architecture Patterns

### CRASH-02: Reading All Index Types

**Current (broken) code — lines 56–62 of meshes.rs:**
```rust
// Only handles U16; silently produces zero indices for U8 and U32 assets
if let Some(gltf::mesh::util::ReadIndices::U16(Iter::Standard(iter))) =
    reader.read_indices()
{
    for v in iter {
        indices.push(v as u32);
    }
}
```

**Why it is broken in two ways:**
1. Only matches the `U16` variant — U8 and U32 assets produce an empty `indices` Vec.
2. Only matches `Iter::Standard` — if the accessor uses sparse encoding (`Iter::Sparse`), the arm does not match even for U16 data. This is a latent bug.

**Option A — `into_u32()` (idiomatic, recommended):**
```rust
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html
if let Some(read_indices) = reader.read_indices() {
    indices = read_indices.into_u32().collect();
} else {
    log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
    continue;
}
```

`CastingIter<U32>` implements `ExactSizeIterator` and `Iterator<Item = u32>`. It dispatches internally over U8/U16/U32 variants and handles both `Standard` and `Sparse` accessors transparently. `collect()` into a `Vec<u32>` works directly.

**Option B — explicit `match` (per CONTEXT.md decision):**
```rust
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html
match reader.read_indices() {
    Some(ReadIndices::U8(iter))  => for v in iter { indices.push(v as u32); },
    Some(ReadIndices::U16(iter)) => for v in iter { indices.push(v as u32); },
    Some(ReadIndices::U32(iter)) => for v in iter { indices.push(v); },
    None => {
        log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
        continue;
    }
}
```

This requires: `use gltf::mesh::util::ReadIndices;` at the top of the match (or qualify inline). The `iter` in each arm is `gltf::accessor::Iter<'_, T>`, which implements `Iterator` regardless of whether the underlying accessor is Standard or Sparse — no need to match on `Iter::Standard` in the arm itself.

**CRITICAL NOTE on the existing `Iter::Standard` pattern:**
The current code destructures `Iter::Standard(iter)` inside the arm. This is wrong practice. `gltf::accessor::Iter<T>` already implements `Iterator<Item = T>` directly — you do NOT need to destructure the `Standard`/`Sparse` variant. The correct arm is simply `ReadIndices::U16(iter)` where `iter: gltf::accessor::Iter<u16>`.

**Recommendation:** Option A (`into_u32()`) is shorter and eliminates the Sparse accessor latent bug. Option B is also correct as long as arms do NOT destructure `Iter::Standard`. The CONTEXT.md specifies Option B — it is valid.

---

### CRASH-03: UV Channel and Vertex Assignment

**What `set: u32` means in `read_tex_coords(set)`:**
glTF 2.0 supports multiple UV sets per primitive via `TEXCOORD_0`, `TEXCOORD_1`, etc. attributes. The integer argument selects which attribute set to read. Set `0` is `TEXCOORD_0` (the base/primary UV map used by PBR materials). Set `1` is `TEXCOORD_1` (used for lightmaps in some workflows). Standard meshes only have set 0.

**Current code (line 80) reads set 1:**
```rust
// BUG: Should be 0, not 1
if let Some(gltf::mesh::util::ReadTexCoords::F32(Iter::Standard(iter))) =
    reader.read_tex_coords(1)
```

**Fix — read from set 0, use `into_f32()` (handles U8/U16/F32 variants):**
```rust
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadTexCoords.html
let uvs: Vec<[f32; 2]> = reader
    .read_tex_coords(0)
    .map(|tc| tc.into_f32().collect())
    .unwrap_or_default();
```

`ReadTexCoords::into_f32()` normalizes U8 (`[0, 255]` → `[0.0, 1.0]`) and U16 (`[0, 65535]` → `[0.0, 1.0]`) per the glTF spec. Most exporters write F32 directly, so this is the common case.

**Fix — write UVs into vertex struct (lines 108–114):**
```rust
// Current: uv_x: 0.0, uv_y: 0.0  ← never uses the uvs vec
// Fixed:
let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
let vtx = Vertex {
    position: *v,
    normal,
    color,
    uv_x,
    uv_y,
};
```

Both parts (channel fix AND vertex assignment) must be applied together. Fixing the channel alone leaves `uv_x/uv_y` still zeroed.

---

### CLEAN-02: glTF Import Error Capture

**Current code (line 36):**
```rust
let (document, buffers, _images) = gltf::import(path).ok()?;
```

**What `gltf::Error` looks like:**

`gltf::import` returns `gltf::Result<(Document, Vec<Data>, Vec<Data>)>` where `gltf::Result<T>` is a type alias for `std::result::Result<T, gltf::Error>`.

`gltf::Error` is an enum with 12 variants including `Io`, `Deserialize`, `Validation`, `Binary`, `BufferLength`, and others. It implements `std::error::Error` and `std::fmt::Display`. The `{}` format specifier produces a human-readable string like `"IO error: No such file or directory (os error 2)"` for an `Io` variant.

**Fix:**
```rust
// Source: docs.rs/gltf/1.4.1/gltf/enum.Error.html
let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
    log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
}).ok()?;
```

Note: `&path` (borrow) is passed to `import` so that `path.as_ref().display()` is still accessible in the closure. If `path` were moved into `import()`, borrowing it in the closure would not compile.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Casting U8/U16/U32 indices to u32 | Manual if-let chain per type | `ReadIndices::into_u32()` | Handles all three types + sparse accessors transparently |
| Normalizing U8/U16 UVs to f32 | Manual bit-manipulation | `ReadTexCoords::into_f32()` | Follows glTF spec normalization precisely |

---

## Common Pitfalls

### Pitfall 1: Destructuring `Iter::Standard` inside `ReadIndices` arms
**What goes wrong:** `if let Some(ReadIndices::U16(Iter::Standard(iter))) = ...` — the `Iter::Standard` destructuring silently fails if the accessor is sparse-encoded. The entire `if let` arm does not match, indices remain empty, and the mesh renders with no geometry.
**Why it happens:** The developer assumes all accessors are Standard (contiguous memory layout). glTF permits sparse accessors, and the `gltf` crate's `Iter<T>` enum wraps both cases.
**How to avoid:** Never destructure `Iter::Standard` when you just want to iterate. Use `iter` directly — `gltf::accessor::Iter<T>` implements `Iterator<Item = T>` for both variants. Or use `into_u32()` / `into_f32()` which handle all cases internally.

### Pitfall 2: Reading UVs from channel 1 instead of 0
**What goes wrong:** `read_tex_coords(1)` returns `None` for the vast majority of glTF files that only export `TEXCOORD_0`. The `uvs` Vec is empty; every vertex gets `(0.0, 0.0)`.
**Why it happens:** Off-by-one; the attribute name is `TEXCOORD_0` but `0` might seem like "first in zero-indexed" rather than "the set parameter".
**How to avoid:** Standard glTF PBR UVs are always channel 0. Only use channel 1 for lightmaps (a specific use case).

### Pitfall 3: Collecting UVs but never reading them back
**What goes wrong:** The `uvs` Vec is populated but `Vertex { uv_x: 0.0, uv_y: 0.0 }` is hardcoded — UVs are thrown away after collection.
**Why it happens:** Two separate code sites must be changed together (collection + usage). Fixing only one leaves the bug.
**How to avoid:** When changing the UV channel fix, audit the vertex struct construction immediately below.

### Pitfall 4: Moving `path` into `gltf::import` then borrowing it for the error message
**What goes wrong:** `gltf::import(path).map_err(|e| { ... path.as_ref().display() ... })` — `path` is moved into `import()`, borrow-checker error in the closure.
**How to avoid:** Pass `&path` to `import()` instead of `path`.

### Pitfall 5: `mesh.name()` returns `Option<&str>` — using `?` propagates None
**What goes wrong:** Line 41 uses `mesh.name()?` inside a loop body, which silently skips meshes without names.
**Status:** This is pre-existing behavior and is NOT being changed in Phase 1. Noted for awareness.

---

## Code Examples

### Complete index reading replacement (Option A — idiomatic):
```rust
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html
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

### Complete index reading replacement (Option B — explicit match per CONTEXT.md):
```rust
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html
use gltf::mesh::util::ReadIndices; // at top of file or inline

match reader.read_indices() {
    Some(ReadIndices::U8(iter))  => {
        for v in iter { indices.push(v as u32); }
    }
    Some(ReadIndices::U16(iter)) => {
        for v in iter { indices.push(v as u32); }
    }
    Some(ReadIndices::U32(iter)) => {
        for v in iter { indices.push(v); }
    }
    None => {
        log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
        continue;
    }
}
```

### Complete UV fix:
```rust
// Line ~80: collection (fix channel 0, use into_f32)
// Source: docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadTexCoords.html
let uvs: Vec<[f32; 2]> = reader
    .read_tex_coords(0)
    .map(|tc| tc.into_f32().collect())
    .unwrap_or_default();

// Lines ~108-114: usage in vertex building loop
let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
let vtx = Vertex {
    position: *v,
    normal,
    color,
    uv_x,
    uv_y,
};
```

### Complete error logging fix:
```rust
// Line 36
// Source: docs.rs/gltf/1.4.1/gltf/enum.Error.html
let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
    log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
}).ok()?;
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Match specific `Iter::Standard` variant | Match `ReadIndices` variant directly; `Iter<T>` is already an `Iterator` | Handles sparse accessors; eliminates silent failure |
| Per-variant manual cast | `into_u32()` / `into_f32()` convenience methods | Single line handles all storage types |

---

## Open Questions

1. **ReadColors sparse accessor behavior**
   - What we know: Line 89 uses the same `Iter::Standard` anti-pattern for `ReadColors::RgbaF32`.
   - What's unclear: Whether any real assets use sparse color accessors (uncommon).
   - Status: Out of scope for Phase 1, but the fix is the same — remove the `Iter::Standard` destructuring.

2. **`primitive.indices()?.count()` for surface count (line 52)**
   - What we know: `primitive.indices()` returns `Option<gltf::Accessor>` and `?.count()` propagates None with the `?` operator, silently skipping the primitive if non-indexed.
   - What's unclear: With the CRASH-02 fix adding `continue` for non-indexed primitives, this `?` on line 52 becomes redundant but harmless since `continue` runs first.
   - Recommendation: Leave as-is. The `continue` in the index-reading block exits before line 52 is re-reached for non-indexed primitives.

---

## Sources

### Primary (HIGH confidence)
- [docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html](https://docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadIndices.html) — `ReadIndices` enum variants, `into_u32()` signature
- [docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadTexCoords.html](https://docs.rs/gltf/1.4.1/gltf/mesh/util/enum.ReadTexCoords.html) — `ReadTexCoords` variants, `into_f32()` signature
- [docs.rs/gltf/1.4.1/gltf/mesh/struct.Reader.html](https://docs.rs/gltf/1.4.1/gltf/mesh/struct.Reader.html) — `read_indices()`, `read_tex_coords(set: u32)`, `read_positions()` signatures
- [docs.rs/gltf/1.4.1/gltf/enum.Error.html](https://docs.rs/gltf/1.4.1/gltf/enum.Error.html) — `gltf::Error` Display impl, 12 variants
- [github.com/gltf-rs/gltf — src/mesh/util/mod.rs](https://github.com/gltf-rs/gltf/blob/master/src/mesh/util/mod.rs) — `ReadIndices` enum source + `into_u32()` impl
- [github.com/gltf-rs/gltf — src/accessor/util.rs](https://github.com/gltf-rs/gltf/blob/master/src/accessor/util.rs) — `Iter<T>` enum: `Standard(ItemIter<T>)` and `Sparse(SparseIter<T>)` variants confirmed
- [docs.rs/gltf/latest/src/gltf/mesh/util/indices.rs.html](https://docs.rs/gltf/latest/src/gltf/mesh/util/indices.rs.html) — `CastingIter` source, `Cast` trait, `U32` impl

### Secondary (MEDIUM confidence)
- WebSearch results confirming `into_u32().collect()` pattern used in community gltf loader examples

---

## Metadata

**Confidence breakdown:**
- `ReadIndices` API (`into_u32`, all three variants): HIGH — verified directly against docs.rs 1.4.1 and GitHub source
- `gltf::accessor::Iter` variants (Standard/Sparse): HIGH — verified from GitHub source
- `read_tex_coords(set)` parameter meaning: HIGH — verified from docs.rs Reader page and glTF 2.0 spec (`TEXCOORD_0` = set 0)
- `ReadTexCoords::into_f32()` normalization: MEDIUM — function confirmed, normalization behavior described in docs but not verified against the glTF 2.0 spec normalization formula directly
- `gltf::Error` Display output format: MEDIUM — type implements Display, exact string format depends on variant (not tested at runtime)

**Research date:** 2026-03-02
**Valid until:** 2026-09-01 (gltf 1.x is stable; no breaking changes expected in the near term)
