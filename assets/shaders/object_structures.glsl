// assets/shaders/object_structures.glsl
// Buffer-reference types only, no descriptor bindings — safe to include from
// any stage. Includer must enable GL_EXT_buffer_reference.

// Must match Vertex in crates/engine/src/primitives.rs.
struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex vertices[];
};

// Must match GPUObjectData in crates/engine/src/primitives.rs.
struct ObjectData {
    mat4 model;
    mat4 normalMatrix;
    VertexBuffer vertexBuffer;
    uint materialIndex;
    uint _pad;
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer {
    ObjectData objects[];
};
