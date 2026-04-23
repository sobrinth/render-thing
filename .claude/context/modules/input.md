# Module: input.rs

## Purpose
Type-safe input event abstraction. Bridges winit events to engine with custom enum hierarchy (ElementState, MouseButton, Key variants). Exported publicly for type safety.

## Key Types
- **ElementState**: Pressed | Released (mirror of winit but owned by engine)
- **MouseButton**: Left | Right | Middle | Other
- **NamedKey**: Space | Shift | Other (subset of named keys we handle)
- **Key**: Character(String) | Named(NamedKey) | Other

## Notable Patterns
- **String keys for characters**: Supports arbitrary character key names (not just ASCII)
- **Named key subset**: Only Space, Shift handled; others map to Other variant

## Unsafe Blocks
None; pure type definitions.

## Dependencies
None (no internal or external dependencies).
