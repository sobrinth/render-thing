#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inWorldPos;

layout(location = 0) out vec4 outFragColor;

void main() {
    vec4 color = texture(colorTex, inUV) * materialData.colorFactors * vec4(inColor, 1.0);

    // glTF metallic-roughness convention: G = roughness, B = metallic
    vec4 metalRough = texture(metalRoughTex, inUV);
    float roughness = clamp(metalRough.g * materialData.metalRoughFactors.y, 0.05, 1.0);
    float metallic  = clamp(metalRough.b * materialData.metalRoughFactors.x, 0.0,  1.0);

    vec3 N = normalize(inNormal);
    vec3 L = normalize(sceneData.sunlightDirection.xyz);
    vec3 V = normalize(sceneData.cameraPos.xyz - inWorldPos);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);

    // Diffuse: metallic surfaces absorb diffuse
    vec3 diffuse = color.rgb * (1.0 - metallic) * NdotL;

    // Blinn-Phong specular: roughness^2 maps [0,1] → shininess [1, 4096]
    float shininess = pow(2.0, 12.0 * (1.0 - roughness * roughness));
    float NdotH = max(dot(N, H), 0.0);
    float spec = (NdotL > 0.0) ? pow(NdotH, shininess) * (1.0 - roughness) : 0.0;
    // Dielectrics: white specular; metals: tinted specular
    vec3 specColor = mix(vec3(1.0), color.rgb, metallic);
    vec3 specular = specColor * spec;

    vec3 ambient = color.rgb * sceneData.ambientColor.xyz;
    float sunStrength = sceneData.sunlightColor.w;
    vec3 lit = (diffuse + specular) * sceneData.sunlightColor.rgb * sunStrength + ambient;

    outFragColor = vec4(lit, color.a);
}
