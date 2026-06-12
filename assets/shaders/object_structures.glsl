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
    vec4 boundsOrigin;
    vec4 boundsExtents;
    VertexBuffer vertexBuffer;
    uint materialIndex;
    uint indexCount;
    uint firstIndex;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer {
    ObjectData objects[];
};

// Must match VkDrawIndexedIndirectCommand.
struct DrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

layout(buffer_reference, std430) writeonly buffer DrawCommandBuffer {
    DrawIndexedIndirectCommand commands[];
};

layout(buffer_reference, std430) buffer DrawCountBuffer {
    uint count;     // read by vkCmdDrawIndexedIndirectCount at offset 0
    uint triangles; // drawn-triangle total, stats only
};
