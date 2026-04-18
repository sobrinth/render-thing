# Rendering Conventions

This document describes the geometric and mathematical conventions used throughout the renderer. Understanding these is important when adding new geometry, shaders, or camera logic.

---

## Coordinate Spaces

Geometry passes through several coordinate spaces on its way to the screen:

```
Object Space → World Space → View Space → Clip Space → NDC → Screen Space
```

| Space | Description |
|-------|-------------|
| **Object space** | Local coordinates of a mesh, as authored |
| **World space** | Object placed in the scene via a model matrix |
| **View space** | World relative to the camera (camera is at origin) |
| **Clip space** | After projection — a 4D homogeneous space |
| **NDC** | After dividing by W — a fixed normalized cube the GPU maps to pixels |
| **Screen space** | Final pixel coordinates on the display |

---

## Handedness

The renderer uses a **right-handed coordinate system** in world and view space:

- **X** points right
- **Y** points up
- **Z** points out of the screen toward the viewer (negative Z goes into the scene)

This matches the glTF specification, so no axis conversion is needed when loading assets.

After projection, the GPU works in Vulkan's **left-handed NDC** space where Y points down and Z runs 0 (near) to 1 (far) into the screen. The projection matrix handles this conversion. The handedness difference between world space and NDC is an emergent consequence of two independent decisions:

- **Y is flipped** because screen pixels are indexed from the top-left downward. Vulkan adopted this directly in NDC so that the mapping to screen coordinates requires no implicit correction. OpenGL chose Y-up in NDC to feel more mathematical but then had to silently flip when writing to pixels — Vulkan makes it explicit instead.

- **Z is negated** because in right-handed view space, objects in front of the camera sit at negative Z. The projection matrix negates Z when producing clip coordinates, which is required so that the perspective divide (dividing by W, where W = −Z in view space) yields positive values. Positive W is necessary for the depth range 0..1 to be meaningful. This negation is what flips the Z axis and makes NDC left-handed.

The projection matrix is the single place that absorbs both of these conventions, so the rest of the codebase can work consistently in right-handed world space.

---

## Projection and Depth

### Reverse-Z

The renderer uses **reverse-Z depth**, a technique that improves floating-point precision for scenes with large view distances:

- The depth buffer is cleared to **0.0**, which represents the far plane
- Fragments closer to the camera have **higher** depth values (near = 1.0)
- Depth test passes when the incoming fragment depth is **greater than or equal to** the stored value

This requires the projection matrix to map near to 1 and far to 0. The benefit is that floating-point precision is concentrated near the camera, where it matters most.

### Depth Range

Vulkan uses a depth range of **0.0 to 1.0** (unlike OpenGL which uses -1.0 to 1.0).

---

## Face Culling

The renderer uses **back-face culling** to skip triangles whose back side faces the camera.

### Winding Order

A triangle's facing direction is determined by the order its vertices appear on screen (winding order):

- **Counter-clockwise (CCW)** = typically front-facing in OpenGL and glTF conventions
- **Clockwise (CW)** = typically back-facing

glTF meshes use CCW winding for front faces. The pipeline is configured to treat **counter-clockwise as front-facing** and cull **back faces**, consistent with glTF conventions. The Y-axis flip in the projection matrix does not reverse the winding order as seen by the rasterizer in this setup.

---

## Transformation Matrices

### Layout

Matrices are stored in **column-major** order, matching both the math library used on the CPU and GLSL's default on the GPU. No transposition is needed when uploading matrices to the GPU.

### Multiplication Order

Transformations are applied right-to-left, which is standard for column-major math:

```
final_position = Projection × View × Model × vertex
```

This reads as: transform the vertex into world space (Model), then into camera space (View), then into clip space (Projection).

### Normal Vectors

Normal vectors cannot be transformed with the same matrix as positions when non-uniform scaling is involved. The correct transform for normals is the **inverse-transpose** of the model matrix. This preserves the perpendicularity of normals to their surfaces after transformation.

---

## glTF Assets

glTF files use a **right-handed, Y-up** coordinate system with **counter-clockwise** front-face winding — consistent with this renderer's world space. Assets load without any axis conversion.

Node transforms in the glTF hierarchy are composed by multiplying parent and child matrices together (parent × child), accumulating the full world transform for each mesh.

---

## Further Reading

- **Coordinate systems and the rendering pipeline**
  https://learnopengl.com/Getting-started/Coordinate-Systems

- **Vulkan coordinate system differences from OpenGL**
  https://www.vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer

- **Depth precision and why reverse-Z helps**
  https://developer.nvidia.com/content/depth-precision-visualized

- **Normal matrix (inverse-transpose) explained**
  https://learnopengl.com/Lighting/Basic-Lighting

- **glTF coordinate system specification**
  https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units

- **Column-major vs row-major matrices in graphics**
  https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector
