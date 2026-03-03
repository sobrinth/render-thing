# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: cleanup-2026-03 — Mesh Loading Correctness Cleanup

**Shipped:** 2026-03-03
**Phases:** 2 | **Plans:** 2 | **Sessions:** ~2

### What Was Built
- Fixed active_mesh default from 2→0, preventing panic on models with <3 meshes (CRASH-01)
- glTF index reading now handles U8/U16/U32 via into_u32() — no more 0-triangle draws (CRASH-02)
- UV coordinates read from TEXCOORD_0 and written to Vertex.uv_x/uv_y fields (CRASH-03)
- glTF import errors now logged via map_err before silent discard (CLEAN-02)
- Clone removed from AllocatedBuffer, FrameData, GPUMeshBuffers — double-free risk eliminated at compile time (CRASH-04, CRASH-05)

### What Worked
- GSD Phase 1 execution was clean: 4 targeted bug fixes, each in its own atomic commit
- Codebase map from `/gsd:map-codebase` gave precise file:line locations for every bug — no hunting
- Using `into_u32()` to normalize all index types was an elegant single-arm match vs 3 separate cases

### What Was Inefficient
- Phases 3–5 were not executed via GSD; deferred work required manual tracking rather than structured plans
- Phase 2 (Clone removal) was executed manually outside GSD — no SUMMARY.md, no formal verification step
- Milestone closed with 5/11 requirements deferred; a tighter initial scope would have given a cleaner completion

### Patterns Established
- `into_u32()` / `into_f32()` normalization pattern for glTF accessor data avoids multi-arm matches
- `map_err(|e| log::error!(...)).ok()?` idiom for converting fallible operations into logged Options

### Key Lessons
1. GSD codebase map pays for itself immediately — specific file:line refs in requirements made Phase 1 near-mechanical
2. Mixing GSD-tracked and manual execution in the same milestone creates inconsistent state — commit to one or the other per phase
3. Scoping a cleanup milestone to exactly what you'll do (not the full audit) avoids the "proceeding with gaps" ceremony

### Cost Observations
- Model mix: ~100% sonnet
- Sessions: ~2 (codebase map + Phase 1 execution)
- Notable: Phase 1 took ~15 minutes per SUMMARY.md; the planning overhead was proportionally higher than execution

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| cleanup-2026-03 | ~2 | 2 | First GSD milestone — baseline established |

### Cumulative Quality

| Milestone | Crash Fixes | Soundness Fixes | Zero-Dep Additions |
|-----------|-------------|-----------------|-------------------|
| cleanup-2026-03 | 5 | 1 (Clone removal) | 0 |

### Top Lessons (Verified Across Milestones)

1. Scope the milestone to exactly what will ship — don't plan 5 phases if only 2 will be executed
