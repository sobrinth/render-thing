# Module: debug.rs

## Purpose
Vulkan debug messenger setup and validation layer management. Logs Vulkan validation messages to env_logger with severity levels (trace, debug, warn, error).

## Key Types
- Debug messenger callback function (vulkan_debug_callback) with CStr message parsing

## Notable Patterns
- **Gated by cfg!(debug_assertions)**: Debug layer only enabled in debug builds
- **Severity mapping**: VERBOSE→trace, INFO→debug, WARNING→warn, ERROR→error
- **CString lifetime management**: get_layer_names_and_pointers() returns owned CStrings to keep them alive

## Unsafe Blocks
- Line 16: CStr::from_ptr(p_message) - safe; callback contract guarantees valid pointer
- Line 9-24: Extern callback - standard Vulkan FFI pattern

## Dependencies
- ash debug_utils extension
- itertools (collect_vec)
- std::ffi (CString, CStr)
