# Milestones

## cleanup-2026-03 Mesh Loading Correctness Cleanup (Shipped: 2026-03-03)

**Phases completed:** 2 phases, 2 plans, 0 tasks

**Requirements completed:** CRASH-01, CRASH-02, CRASH-03, CRASH-04, CRASH-05, CLEAN-02 (6/11)

**Key accomplishments:**
- Fixed active_mesh default from 2→0 — prevents panic on models with <3 meshes
- glTF index reading handles U8/U16/U32 via into_u32() — eliminates 0-triangle draws
- UV coordinates read from TEXCOORD_0 and written to Vertex.uv_x/uv_y
- glTF import errors now logged before silent discard
- Clone removed from AllocatedBuffer, FrameData, GPUMeshBuffers — double-free risk eliminated at compile time

**Known gaps (deferred):**
- CRASH-06: UI texture free timing soundness (ui.rs:193)
- CLEAN-01: Effect slider range hardcoded to 0..=2
- CLEAN-03: mem::transmute for push constants (renderer.rs:520, 628)
- CLEAN-04: Frame sleep integer division (main.rs)
- CLEAN-05: Dependency updates not run

---

