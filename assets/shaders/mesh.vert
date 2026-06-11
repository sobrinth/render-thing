#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#include "object_structures.glsl"

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outWorldPos;
layout(location = 4) flat out uint outMaterialIndex;

layout(push_constant) uniform constants {
    ObjectBuffer objectBuffer;
} PushConstants;

void main() {
    // first_instance carries the object record index; instance count is always 1.
    ObjectData obj = PushConstants.objectBuffer.objects[gl_InstanceIndex];
    Vertex v = obj.vertexBuffer.vertices[gl_VertexIndex];

    vec4 worldPos = obj.model * vec4(v.position, 1.0f);
    gl_Position = sceneData.viewproj * worldPos;
    outColor = v.color.xyz;
    outUV = vec2(v.uv_x, v.uv_y);
    outNormal = normalize(mat3(obj.normalMatrix) * v.normal);
    outWorldPos = worldPos.xyz;
    outMaterialIndex = obj.materialIndex;
}
