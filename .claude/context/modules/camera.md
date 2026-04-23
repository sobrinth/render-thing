# Module: camera.rs

## Purpose
Freecam camera controller. Tracks position, rotation (pitch/yaw), velocity. Responds to WASD, Space, Shift, and mouse drag for view/projection matrices.

## Key Types
- **Camera**: Holds position, velocity (Vec3), pitch/yaw angles, mouse_pressed flag. Supports key/mouse events.

## Notable Patterns
- **Velocity-based movement**: handle_key_event() sets velocity.x/y/z based on WASD/Space/Shift; update() applies rotated velocity each frame
- **Mouse drag rotation**: Yaw/pitch updated only when left-mouse button pressed; delta_x/y divided by 200 for smoothing
- **Matrix composition**: get_view_matrix() = inverse(translation * rotation); get_rotation_matrix() = yaw rotation * pitch rotation (quaternion-based)
- **Inverse view transform**: Standard camera transform inverted to get view matrix

## Unsafe Blocks
None; uses nalgebra_glm for all math.

## Dependencies
- input (ElementState, Key, NamedKey, MouseButton enums)
- nalgebra_glm (Mat4, Vec3, quaternion ops)
