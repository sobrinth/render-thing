#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec4 outFragColor;

void main() {
    float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);

    vec4 color = texture(colorTex, inUV) * materialData.colorFactors;
    vec3 ambient = color.rgb * sceneData.ambientColor.xyz;

    outFragColor = vec4(color.rgb * lightValue * sceneData.sunlightColor.w + ambient, color.a);
}
