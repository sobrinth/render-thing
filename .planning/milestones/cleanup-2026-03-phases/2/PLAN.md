---
phase: 02-handle-safety
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/engine/src/renderer.rs
autonomous: true
requirements:
  - CRASH-04
  - CRASH-05

must_haves:
  truths:
    - "AllocatedBuffer does not implement Clone — any .clone() call on it is a compile error"
    - "FrameData does not implement Clone — any .clone() call on it is a compile error"
    - "The stale comment '// maybe no clone?' is removed — the question is answered by removing the derive"
    - "cargo build exits 0 with no new errors introduced"
    - "No other Clone derives in renderer.rs are disturbed (QueueData, ImmediateSubmitData, ComputePushConstants, ComputeEffect keep their derives)"
  artifacts:
    - path: crates/engine/src/renderer.rs
      provides: "AllocatedBuffer struct definition without Clone derive"
      contains: "#[derive(Debug)]\n#[allow(dead_code)]\npub struct AllocatedBuffer"
    - path: crates/engine/src/renderer.rs
      provides: "FrameData struct definition without Clone derive and without the stale comment"
      contains: "pub struct FrameData {"
  key_links:
    - from: "crates/engine/src/renderer.rs:1315"
      to: "removed"
      via: "delete line containing '// maybe no clone?'"
      pattern: "maybe no clone"
    - from: "crates/engine/src/renderer.rs:1316"
      to: "crates/engine/src/renderer.rs:1315 (renumbered)"
      via: "replace '#[derive(Clone)]' with nothing — FrameData gets no derive"
      pattern: "derive.*Clone.*FrameData"
    - from: "crates/engine/src/renderer.rs:1407"
      to: "crates/engine/src/renderer.rs:1407"
      via: "replace '#[derive(Debug, Clone)]' with '#[derive(Debug)]'"
      pattern: "derive.*Debug.*Clone.*AllocatedBuffer"
---

<objective>
Remove the `Clone` derive from `AllocatedBuffer` (CRASH-05) and `FrameData` (CRASH-04) in renderer.rs.

Purpose: Both types wrap Vulkan GPU resource handles. Deriving `Clone` on them produces a second Rust value pointing at the same underlying GPU resources (VMA allocation, command pool, semaphore, fence). If a clone is ever destroyed, it calls `destroy` on handles that the original value still owns — a double-free that corrupts the VMA allocator or triggers Vulkan validation errors. Making Clone unavailable turns this latent runtime hazard into a hard compile error.

Output: renderer.rs with two fewer derive attributes and one fewer stale comment. The build is green and no call sites require updating (confirmed by grep — zero `.clone()` calls on these types exist in the workspace).
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md

<!-- Key source location confirmed by pre-plan read:

crates/engine/src/renderer.rs:1315-1316 (FrameData):
  // maybe no clone?        ← line 1315, delete entirely
  #[derive(Clone)]          ← line 1316, delete entirely (no derive replaces it)
  pub struct FrameData {

crates/engine/src/renderer.rs:1407 (AllocatedBuffer):
  #[derive(Debug, Clone)]   ← change to #[derive(Debug)]
  #[allow(dead_code)]
  pub struct AllocatedBuffer {

Unaffected derives nearby (do NOT touch):
  renderer.rs:1343  #[derive(Copy, Clone)]     — QueueData (Copy types only)
  renderer.rs:1366  #[derive(Debug, Clone, Copy)]  — ImmediateSubmitData (out of scope)
  renderer.rs:1383  #[derive(Debug, Clone, Copy)]  — ComputePushConstants (Copy types only)
  renderer.rs:1392  #[derive(Debug, Clone, Copy)]  — ComputeEffect (out of scope)
-->
</context>

<tasks>

<task type="auto">
  <name>Task 1: Remove Clone from AllocatedBuffer (CRASH-05)</name>
  <files>crates/engine/src/renderer.rs</files>
  <action>
    On line 1407 of `crates/engine/src/renderer.rs`, change:

        #[derive(Debug, Clone)]

    to:

        #[derive(Debug)]

    The line immediately below it (`#[allow(dead_code)]`) and the struct body are unchanged. This is a single-token removal from the derive list.

    Do NOT touch any other derive on any other struct. Specifically, leave untouched:
    - Line 1343: `#[derive(Copy, Clone)]` on `QueueData`
    - Line 1366: `#[derive(Debug, Clone, Copy)]` on `ImmediateSubmitData`
    - Line 1383: `#[derive(Debug, Clone, Copy)]` on `ComputePushConstants`
    - Line 1392: `#[derive(Debug, Clone, Copy)]` on `ComputeEffect`
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && rtk cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>
    `grep -n "derive.*Clone" crates/engine/src/renderer.rs` does not produce a line containing `AllocatedBuffer` in its context (the line 1407 entry is gone). `cargo check -p engine` emits zero error lines.
  </done>
</task>

<task type="auto">
  <name>Task 2: Remove Clone from FrameData and delete stale comment (CRASH-04)</name>
  <files>crates/engine/src/renderer.rs</files>
  <action>
    In `crates/engine/src/renderer.rs`, remove the two lines that precede `pub struct FrameData {`:

        // maybe no clone?    ← delete this line entirely
        #[derive(Clone)]      ← delete this line entirely

    After deletion the struct definition should read:

        pub struct FrameData {
            pub command_pool: vk::CommandPool,
            ...

    There is no replacement derive. FrameData requires no auto-derived traits — it has a manual `destroy` method and is not Debug-printed anywhere. Do not add `#[derive(Debug)]` unless it already exists on the struct (it does not — verify before adding anything).

    The blank line between the closing `}` of the `Drop` impl above (line 1313) and `pub struct FrameData` may be preserved as-is for readability.
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && rtk cargo check -p engine 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>
    `grep -n "maybe no clone" crates/engine/src/renderer.rs` returns no matches. `grep -n "derive.*Clone" crates/engine/src/renderer.rs` does not produce a line whose nearby context is the FrameData struct. `cargo check -p engine` emits zero error lines.
  </done>
</task>

<task type="auto">
  <name>Task 3: Full build verification</name>
  <files></files>
  <action>
    Run `cargo build` from the workspace root `H:/vulkan-rust` to confirm the entire workspace compiles cleanly after both derive removals. This catches any cross-crate effects that `cargo check -p engine` might not surface (e.g., the `application` crate).
  </action>
  <verify>
    <automated>cd H:/vulkan-rust && rtk cargo build 2>&1 | grep -E "^error" | head -20</automated>
  </verify>
  <done>
    `cargo build` exits 0. The grep for `^error` returns no lines. The binary artifacts are produced without new warnings about the changed structs.
  </done>
</task>

</tasks>

<verification>
Run these checks after all tasks complete:

```bash
# 1. Neither target struct retains a Clone derive
cd H:/vulkan-rust
grep -n "derive.*Clone" crates/engine/src/renderer.rs
# Expected output: lines for QueueData, ImmediateSubmitData, ComputePushConstants, ComputeEffect
# Must NOT appear: any line whose next non-blank line is "pub struct AllocatedBuffer"
# Must NOT appear: any line whose next non-blank line is "pub struct FrameData"

# 2. Stale comment is gone
grep -n "maybe no clone" crates/engine/src/renderer.rs
# Expected: no output (zero matches)

# 3. No .clone() calls on the affected types exist
grep -rn "\.clone()" crates/engine/src/ --include="*.rs"
# Expected: only lines in ui.rs touching egui_ctx, Arc-wrapped allocator/device, Vec
# Must NOT appear: any line containing AllocatedBuffer or FrameData

# 4. Full build is clean
rtk cargo build
# Expected: exits 0, zero error lines
```
</verification>

<success_criteria>
1. `AllocatedBuffer` in renderer.rs has `#[derive(Debug)]` only — the `Clone` token is absent from its derive list
2. `FrameData` in renderer.rs has no derive attribute at all — neither `Clone` nor any replacement
3. The comment `// maybe no clone?` does not appear anywhere in renderer.rs
4. `cargo build` exits 0 with no new errors
5. All other Clone derives in renderer.rs are undisturbed
</success_criteria>

<output>
After completion, create `.planning/phases/02-handle-safety/02-01-SUMMARY.md` with:
- What changed (the two exact line edits)
- Verification results (grep outputs confirming absence of Clone derives on target types)
- Build status (exit code, any warnings)
- Requirements closed: CRASH-04, CRASH-05
</output>
