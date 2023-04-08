#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

const vec3 rect[6] = {
    vec3(-1.0, 1.0, 0.0),
    vec3(-1.0, -1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(-1.0, -1.0, 0.0),
    vec3(1.0, -1.0, 0.0)
};

const vec2 tex_c[6] = {
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 0.0)
};

out vec2 tex_coords;

void main() {
    tex_coords = tex_c[gl_VertexID];
    gl_Position = vec4(rect[gl_VertexID], 1.0);
}
