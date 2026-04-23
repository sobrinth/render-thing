# render-thing

Rust Vulkan renderer using ash + egui.

## Context Pack

Before any planning or multi-file work, read:
- `.claude/context/overview.md` — architecture, frame loop, build commands, unsafe inventory
- `.claude/context/modules/<name>.md` — per-module purpose, key types, invariants

## Scope Discipline

When asked to review or refactor code, propose only the minimal set of changes that satisfy the literal request. Do not broaden scope unless asked.

## Cross-Platform Compatibility

Hooks and shell commands must work on both Windows and Unix. Prefer `cargo` commands over shell-specific tools.
