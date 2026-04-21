---
name: new-shader
description: Create a new GLSL shader file and register it in build.rs
---

When creating a new shader:
1. Create the file at `assets/shaders/<name>.<ext>` (ext: .vert, .frag, .comp)
2. Add `println!("cargo::rerun-if-changed=../../assets/shaders/<name>.<ext>");` to `crates/engine/build.rs`
3. Confirm the shader type matches its usage (vertex/fragment/compute)
