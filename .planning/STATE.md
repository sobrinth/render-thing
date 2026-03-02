# STATE — vulkan-rust Cleanup Milestone

*Project memory. Updated at each phase/plan boundary.*

---

## Project Reference

**Core value:** A correct, non-crashing foundation for experimenting with Vulkan rendering techniques.
**Milestone goal:** Fix correctness bugs and low-hanging-fruit issues from the 2026-03-02 codebase audit.
**Scope constraint:** Bug fixes and safe refactors only — no architectural changes.

---

## Current Position

**Current phase:** Phase 2 — Handle Safety
**Current plan:** (next plan TBD)
**Status:** Phase 1 complete; ready to begin Phase 2

**Progress:**
```
[COMPLETE  ] Phase 1: Mesh Loading Correctness
[          ] Phase 2: Handle Safety
[          ] Phase 3: UI and Timing
[          ] Phase 4: Safe Push Constants
[          ] Phase 5: Dependency Updates
```

Overall: 1/5 phases complete

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases defined | 5 |
| Requirements mapped | 11/11 |
| Plans created | 1 |
| Plans complete | 1 |

---

## Accumulated Context

### Key Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| Default active_mesh to 0 | Safest default; UI slider still allows any valid mesh | Accepted |
| Retain gltf::accessor::Iter import | Colors block (line 89) still uses Iter::Standard; out of scope for Phase 1 | Accepted |
| Use bytemuck for push constants | Compile-time Pod/Zeroable checks vs silent transmute | Pending |
| Leave `#![allow(dead_code)]` | Too many items to audit cleanly during a bug-fix pass | Accepted |
| Remove sleep, rely on FIFO present mode | Integer division gives wrong frame time; FIFO is correct | Pending |

### Known Constraints

- Rust nightly (pinned) — do not change toolchain
- Core rendering API stays on ash + Vulkan 1.3
- No architectural changes (no VkContext decoupling, no pipeline abstraction)
- All Vulkan FFI unsafe blocks are acceptable; only reduce unsafe where bytemuck provides a safe alternative

### Key Files

| File | Relevance |
|------|-----------|
| `crates/engine/src/renderer.rs` | Monolithic 1452-line renderer; CRASH-01 (fixed), CRASH-04, CRASH-05, CLEAN-03 |
| `crates/engine/src/meshes.rs` | glTF loading; CRASH-02 (fixed), CRASH-03 (fixed), CLEAN-02 (fixed) |
| `crates/engine/src/ui.rs` | egui integration; CRASH-06, CLEAN-01 |
| `crates/application/src/main.rs` | Application entry point; CLEAN-04 |
| `crates/engine/src/primitives.rs` | AllocatedBuffer definition; CRASH-05 cascade |
| `Cargo.toml` / `Cargo.lock` | Dependency manifest; CLEAN-05 |

### Blockers

None currently.

### Open Questions

- CRASH-06: Is the current deferred-free strategy for UI textures actually safe with FRAME_OVERLAP=2? Needs investigation before deciding whether to fix or document.

---

## Session Continuity

**Last updated:** 2026-03-02 (Phase 1 complete)
**Stopped at:** Completed 01-PLAN.md (all 4 tasks, cargo build exit 0)
**Next action:** Plan and execute Phase 2: Handle Safety
