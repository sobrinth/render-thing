---
phase: 01-mesh-loading-correctness
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/engine/src/meshes.rs
  - crates/engine/src/renderer.rs
autonomous: true
requirements: [CRASH-01, CRASH-02, CRASH-03, CLEAN-02]

must_haves:
  truths:
    - "Renderer starts without panic when the loaded glTF contains fewer than 3 meshes"
    - "Meshes with U8, U16, and U32 index buffers render visible triangles"
    - "UV-mapped surfaces display correct texture coordinates from glTF channel 0"
    - "Loading a missing or corrupt glTF file produces a logged error, not silent failure"
  artifacts:
    - path: "crates/engine/src/meshes.rs"
      provides: "glTF loading: correct index types, correct UVs, error logging"
      contains: "into_u32(), read_tex_coords(0), log::error!"
    - path: "crates/engine/src/renderer.rs"
      provides: "Safe active_mesh initialization"
      contains: "active_mesh: 0"
  key_links:
    - from: "crates/engine/src/meshes.rs:36"
      to: "log::error! macro"
      via: "map_err closure"
      pattern: "map_err.*log::error!"
    - from: "crates/engine/src/meshes.rs:56-62"
      to: "reader.read_indices()"
      via: "into_u32().collect()"
      pattern: "into_u32\\(\\)\\.collect"
    - from: "crates/engine/src/meshes.rs:80"
      to: "uvs Vec"
      via: "read_tex_coords(0).map(|tc| tc.into_f32().collect()).unwrap_or_default()"
      pattern: "read_tex_coords\\(0\\)"
    - from: "crates/engine/src/meshes.rs:108-114"
      to: "Vertex { uv_x, uv_y }"
      via: "uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0))"
      pattern: "uvs\\.get\\(i\\)"
---

<objective>
Fix four correctness bugs in the glTF mesh loading pipeline so that meshes render correctly and the renderer starts safely.

Purpose: These are the foundational correctness fixes for the milestone. Every other phase depends on a renderer that loads correctly without crashing.

Output:
- `crates/engine/src/meshes.rs` — corrected index reading, UV reading, and error logging
- `crates/engine/src/renderer.rs` — safe active_mesh initialization
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/1/CONTEXT.md
@.planning/phases/1/RESEARCH.md

<interfaces>
<!-- Key types in scope. Extracted from crates/engine/src/meshes.rs. -->

```rust
// crates/engine/src/meshes.rs — current imports (top of file)
use crate::primitives::{GPUMeshBuffers, Vertex};
use crate::renderer::VulkanRenderer;
use gltf::accessor::Iter;        // <-- NOTE: Iter is imported but must NOT be used
                                  //     in the fixed code (see RESEARCH.md pitfall 1)
use std::path::Path;
use std::sync::Arc;
use vk_mem::Allocator;

// Vertex struct fields relevant to UV fix (from crates/engine/src/primitives.rs):
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_x: f32,
    pub normal: [f32; 3],
    pub uv_y: f32,
    pub color: [f32; 4],
}
```

<!-- The `use gltf::accessor::Iter;` import becomes dead after the fix.    -->
<!-- Remove it or suppress with #[allow(unused_imports)] only if needed.   -->
<!-- Preferred: remove it since it is no longer referenced.                -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix active_mesh panic (CRASH-01)</name>
  <files>crates/engine/src/renderer.rs</files>
  <action>
At line 154, change the struct field initializer:

  BEFORE:
      active_mesh: 2,

  AFTER:
      active_mesh: 0,

This is a single token change in the renderer struct initialization block (lines ~140–157). The surrounding context is:

  ```rust
  meshes: None,
  active_mesh: 0,    // was: 2
  scene_data_layout,
  ```

No other changes in this file for this task.

Rationale: `active_mesh: 2` unconditionally indexes `meshes[2]` on first render. If the loaded
glTF has fewer than 3 meshes, this panics. Defaulting to 0 is safe because the UI slider bounds
the selection after load (per CONTEXT.md decision).
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>`cargo check -p engine` exits 0. The line `active_mesh: 2` no longer appears in renderer.rs.</done>
</task>

<task type="auto">
  <name>Task 2: Fix glTF import error logging (CLEAN-02)</name>
  <files>crates/engine/src/meshes.rs</files>
  <action>
At line 36, replace the silent discard with an explicit error log.

  BEFORE (line 36):
      let (document, buffers, _images) = gltf::import(path).ok()?;

  AFTER (lines 36–39):
      let (document, buffers, _images) = gltf::import(&path).map_err(|e| {
          log::error!("Failed to load glTF '{}': {}", path.as_ref().display(), e);
      }).ok()?;

Key detail: pass `&path` (not `path`) to `gltf::import` so that `path` remains accessible
inside the closure for the `path.as_ref().display()` call. Moving `path` into `import()` would
cause a borrow-checker error in the closure (see RESEARCH.md pitfall 4).

No other changes in this file for this task.
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>`cargo check -p engine` exits 0. Line 36 of meshes.rs contains `map_err` with `log::error!`. The string `gltf::import(path).ok()` no longer appears.</done>
</task>

<task type="auto">
  <name>Task 3: Fix index type handling (CRASH-02)</name>
  <files>crates/engine/src/meshes.rs</files>
  <action>
Replace lines 56–62 (the U16-only `if let` block) with a `match` that handles all three index types.

  BEFORE (lines 56–62):
      if let Some(gltf::mesh::util::ReadIndices::U16(Iter::Standard(iter))) =
          reader.read_indices()
      {
          for v in iter {
              indices.push(v as u32);
          }
      }

  AFTER:
      match reader.read_indices() {
          Some(read_indices) => {
              indices = read_indices.into_u32().collect();
          }
          None => {
              log::warn!("Skipping non-indexed primitive in mesh '{}'", name);
              continue;
          }
      }

Notes:
- `into_u32()` returns a `CastingIter<U32>` that implements `Iterator<Item = u32>`. It handles
  U8, U16, and U32 source types AND both Standard and Sparse accessors internally. No manual
  per-variant casting is needed.
- The `None` arm replaces the previous silent skip (empty indices produced 0-triangle draws).
- Do NOT destructure `Iter::Standard(iter)` inside any arm. Use the outer `read_indices` binding
  directly. This is the critical fix for the sparse accessor latent bug (RESEARCH.md pitfall 1).
- After this change, `use gltf::accessor::Iter;` at the top of the file is no longer used.
  Remove that import line (line 3) to keep the file warning-free. Confirm the `Iter` type is
  not referenced elsewhere in meshes.rs before removing.
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>`cargo check -p engine` exits 0. The pattern `ReadIndices::U16(Iter::Standard` no longer appears. `into_u32().collect()` is present. `use gltf::accessor::Iter` is removed (or produces no unused-import warning).</done>
</task>

<task type="auto">
  <name>Task 4: Fix UV channel and vertex UV assignment (CRASH-03)</name>
  <files>crates/engine/src/meshes.rs</files>
  <action>
This fix has two parts that must be applied together. Fixing only one leaves the bug.

--- PART A: Fix the UV collection block (lines 79–86) ---

  BEFORE (lines 79–86):
      let mut uvs = Vec::new();
      if let Some(gltf::mesh::util::ReadTexCoords::F32(Iter::Standard(iter))) =
          reader.read_tex_coords(1)
      {
          for v in iter {
              uvs.push(v);
          }
      }

  AFTER:
      let uvs: Vec<[f32; 2]> = reader
          .read_tex_coords(0)
          .map(|tc| tc.into_f32().collect())
          .unwrap_or_default();

Changes:
- `read_tex_coords(1)` → `read_tex_coords(0)`: reads from TEXCOORD_0 (the standard primary UV
  set used by glTF PBR materials). Channel 1 is for lightmaps and is absent in most assets.
- `into_f32()`: normalizes U8/U16 storage formats to [0.0, 1.0] range per glTF spec. Most
  exporters write F32 directly, so this is a no-op for common assets.
- `unwrap_or_default()`: silently falls back to an empty Vec when channel 0 is absent, which
  is correct for untextured meshes (no log needed per CONTEXT.md decision).
- Remove the `mut` qualifier since the Vec is now constructed in one expression, not mutated.

--- PART B: Fix the vertex construction loop (lines 108–114) ---

  BEFORE (lines 108–114, inside `positions.iter().enumerate().for_each(|(i, v)| { ... })`):
      let vtx = Vertex {
          position: *v,
          normal,
          color,
          uv_x: 0.0,
          uv_y: 0.0,
      };

  AFTER:
      let (uv_x, uv_y) = uvs.get(i).copied().map(|v| (v[0], v[1])).unwrap_or((0.0, 0.0));
      let vtx = Vertex {
          position: *v,
          normal,
          color,
          uv_x,
          uv_y,
      };

The new `let (uv_x, uv_y) = ...` line is inserted immediately before the `let vtx = Vertex {`
line. `uvs.get(i)` is safe even when `uvs` is shorter than `positions` (falls back to 0.0).

Do NOT change the `OVERRIDE_COLORS` block below the loop (lines 121–126). It is an intentional
debug aid and is explicitly out of scope.
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>`cargo check -p engine` exits 0. `read_tex_coords(0)` is present. `read_tex_coords(1)` does not appear. `uvs.get(i)` is present in the vertex loop. `uv_x: 0.0` and `uv_y: 0.0` as hardcoded literals no longer appear in the Vertex construction.</done>
</task>

<task type="auto">
  <name>Task 5: Final build verification</name>
  <files></files>
  <action>
Run a full workspace build to confirm all four fixes compile together cleanly with no errors
and no new warnings introduced by this phase.

  cargo build 2>&1

If the build fails, diagnose from the error output. Common causes:
- `Iter` import removal in Task 3 was too aggressive (check if `Iter` is still referenced in
  the colors block at line ~89 — if so, keep the import and suppress with `#[allow(unused_imports)]`
  only as a last resort; prefer removing the `Iter::Standard` destructure from the colors block too,
  replacing with just `ReadColors::RgbaF32(iter)`, though that change is out of scope for Phase 1).
- `path` borrow conflict in Task 2 (confirm `&path` is passed, not `path`).

Do not mark this task done until `cargo build` exits 0.
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && cargo build 2>&1 | tail -5</automated>
  </verify>
  <done>`cargo build` exits 0 with output ending in `Finished` or `Compiling ... Finished`. No error lines present.</done>
</task>

</tasks>

<verification>
After all tasks complete, verify each requirement is addressed:

| REQ-ID | Check |
|--------|-------|
| CRASH-01 | `grep "active_mesh: 2" crates/engine/src/renderer.rs` returns no matches |
| CRASH-02 | `grep "into_u32" crates/engine/src/meshes.rs` returns a match; `grep "Iter::Standard" crates/engine/src/meshes.rs` returns no matches in the index block |
| CRASH-03 | `grep "read_tex_coords(0)" crates/engine/src/meshes.rs` returns a match; `grep "read_tex_coords(1)" crates/engine/src/meshes.rs` returns no matches; `grep "uvs.get(i)" crates/engine/src/meshes.rs` returns a match |
| CLEAN-02 | `grep "map_err" crates/engine/src/meshes.rs` returns a match; `grep 'gltf::import(path).ok()' crates/engine/src/meshes.rs` returns no matches |

Run all checks as one command:
```
cd H:/vulkan-rust && \
  grep -c "active_mesh: 2" crates/engine/src/renderer.rs && echo "FAIL: CRASH-01 not fixed" || \
  grep -q "into_u32" crates/engine/src/meshes.rs && echo "PASS: CRASH-02 index fix present" && \
  grep -q "read_tex_coords(0)" crates/engine/src/meshes.rs && echo "PASS: CRASH-03 channel fix present" && \
  grep -q "uvs.get(i)" crates/engine/src/meshes.rs && echo "PASS: CRASH-03 vertex assignment present" && \
  grep -q "map_err" crates/engine/src/meshes.rs && echo "PASS: CLEAN-02 error log present"
```
</verification>

<success_criteria>
Phase 1 is complete when ALL of the following are true:

1. `cargo build` exits 0 with no new errors.
2. `crates/engine/src/renderer.rs` line ~154 reads `active_mesh: 0` (not `2`).
3. `crates/engine/src/meshes.rs` uses `into_u32().collect()` for index reading with no `Iter::Standard` destructuring in that block.
4. `crates/engine/src/meshes.rs` reads `read_tex_coords(0)` (not `1`) and writes the result into `Vertex { uv_x, uv_y }` via `uvs.get(i)`.
5. `crates/engine/src/meshes.rs` uses `gltf::import(&path).map_err(|e| { log::error!(...) }).ok()?` (not the bare `.ok()?` form).
</success_criteria>

<output>
After completion, create `.planning/phases/1/01-SUMMARY.md` with:
- What was changed (file + line + before/after for each fix)
- Any deviations from the plan (e.g., if the `Iter` import removal affected the colors block)
- Final `cargo build` output (last 3 lines)
- Status of each requirement: CRASH-01, CRASH-02, CRASH-03, CLEAN-02
</output>
