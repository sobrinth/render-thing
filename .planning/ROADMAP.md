# ROADMAP — vulkan-rust Cleanup Milestone

**Milestone goal:** Fix correctness bugs and low-hanging-fruit issues identified in the 2026-03-02 codebase audit. Deliver a correct, non-crashing renderer foundation.

**Depth:** Quick (3–5 phases)
**Coverage:** 11/11 v1 requirements mapped

---

## Phases

- [ ] **Phase 1: Mesh Loading Correctness** — Fix glTF loading so meshes actually render with correct geometry and UVs
- [ ] **Phase 2: Handle Safety** — Remove Clone from Vulkan handle wrappers to eliminate double-free risk
- [ ] **Phase 3: UI and Timing** — Fix UI texture safety, dynamic slider range, and frame sleep
- [ ] **Phase 4: Safe Push Constants** — Replace unsafe transmute with bytemuck for push constant byte casting
- [ ] **Phase 5: Dependency Updates** — Run cargo update and check for non-breaking major version upgrades

---

## Phase Details

### Phase 1: Mesh Loading Correctness
**Goal**: glTF meshes load and render correctly — correct index types, correct UVs, safe mesh selection, and visible errors on bad assets
**Depends on**: Nothing (foundational correctness fix)
**Requirements**: CRASH-01, CRASH-02, CRASH-03, CLEAN-02
**Success Criteria** (what must be TRUE):
  1. Renderer starts without panic regardless of how many meshes the loaded glTF file contains
  2. Meshes with U8, U16, and U32 index buffers all render visible triangles (not 0-triangle draws)
  3. UV-mapped surfaces display correct texture coordinates from glTF channel 0
  4. Loading a missing or corrupt glTF file produces a logged error message rather than silent failure
**Plans**: 1 plan

Plans:
- [ ] 01-PLAN.md — Fix active_mesh init, index types (U8/U16/U32), UV channel + vertex assignment, glTF error logging

### Phase 2: Handle Safety
**Goal**: Vulkan handle wrappers cannot be accidentally cloned, eliminating the double-free hazard at compile time
**Depends on**: Phase 1
**Requirements**: CRASH-04, CRASH-05
**Success Criteria** (what must be TRUE):
  1. `AllocatedBuffer` does not implement `Clone` — any accidental clone attempt is a compile error
  2. `FrameData` does not implement `Clone` — any accidental clone attempt is a compile error
  3. `GPUMeshBuffers` does not implement `Clone` — cascade removal is complete
  4. Project compiles clean with no uses of the removed Clone impls
**Plans**: TBD

### Phase 3: UI and Timing
**Goal**: UI is sound with multiple frames in flight, the effect slider reflects actual effect count, and frame pacing relies on the present mode rather than broken integer math
**Depends on**: Phase 1
**Requirements**: CRASH-06, CLEAN-01, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. Adding or removing a compute background effect automatically updates the UI slider range without a code change
  2. Frame pacing is driven by `PresentModeKHR::FIFO` with no sleep call in the render loop
  3. The UI texture free strategy is either confirmed safe with a clear code comment or corrected so textures are not freed while in-flight GPU work may reference them
**Plans**: TBD

### Phase 4: Safe Push Constants
**Goal**: Push constant byte casting uses bytemuck with compile-time Pod/Zeroable checks instead of unsafe transmute
**Depends on**: Phase 1
**Requirements**: CLEAN-03
**Success Criteria** (what must be TRUE):
  1. `mem::transmute` is no longer used for push constant conversion in renderer.rs
  2. Push constant structs derive or implement `bytemuck::Pod` and `bytemuck::Zeroable`
  3. Project compiles and renders identically to before the change
**Plans**: TBD

### Phase 5: Dependency Updates
**Goal**: Dependencies are at current patch levels; any available non-breaking major upgrades are either applied or explicitly deferred with a note
**Depends on**: Phase 4 (all code fixes complete before touching deps)
**Requirements**: CLEAN-05
**Success Criteria** (what must be TRUE):
  1. `cargo update` has been run and `Cargo.lock` reflects current patch versions
  2. Key dependencies (ash, winit, vk-mem, egui, egui-ash-renderer) have been checked for new major versions
  3. Any applied major upgrades do not break compilation or rendering
  4. Any deferred major upgrades are noted in PROJECT.md Key Decisions
**Plans**: TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Mesh Loading Correctness | 0/1 | Planned | — |
| 2. Handle Safety | 0/? | Not started | — |
| 3. UI and Timing | 0/? | Not started | — |
| 4. Safe Push Constants | 0/? | Not started | — |
| 5. Dependency Updates | 0/? | Not started | — |
