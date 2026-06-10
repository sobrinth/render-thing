// assets/shaders/input_structures.glsl

layout(set = 0, binding = 0) uniform SceneData {
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
    vec4 cameraPos;
} sceneData;

// Must match MAX_TEXTURES in crates/engine/src/bindless.rs.
layout(set = 1, binding = 0) uniform sampler2D textures[4096];

// Must match GPUMaterial in crates/engine/src/bindless.rs.
struct Material {
    vec4 colorFactors;
    vec4 metalRoughFactors;
    uint colorTexIndex;
    uint metalRoughTexIndex;
    uint _pad0;
    uint _pad1;
};

layout(set = 1, binding = 1, std430) readonly buffer Materials {
    Material materials[];
} materialBuffer;
