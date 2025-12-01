#version 450

layout (location = 0) out vec3 outColor;

void main() {
    // const triangle lol
    const vec3 positions[3] = vec3[3](
    vec3(1.f, 1.f, 0.f),
    vec3(-1.f, 1.f, 0.f),
    vec3(0.f, -1.f, 0.f)
    );

    // const colors lol
    const vec3 colors[3] = vec3[3](
    vec3(1.0f, 0.f, 0.f), // red
    vec3(0.f, 1.f, 0.f), // green
    vec3(0.f, 0.f, 1.f)// blue
    );

    // output the position of each vertex
    gl_Position = vec4(positions[gl_VertexIndex], 1.f);
    outColor = colors[gl_VertexIndex];
}