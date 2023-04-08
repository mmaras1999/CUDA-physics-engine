#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

layout(location = 0) in vec3 sphere_vertex;
layout(location = 1) in float particle_color;
layout(location = 2) in vec3 center_position;
layout(location = 3) uniform mat4 view_matrix;
layout(location = 4) uniform mat4 proj_matrix;
layout(location = 5) uniform vec3 camera_position;
layout(location = 6) uniform mat4 model_matrix;

out vec3 camera_relative_pos;
out float v_color;

void main() {
    vec4 pos = model_matrix * vec4(sphere_vertex + center_position, 1.0);
    camera_relative_pos = sphere_vertex + center_position - camera_position;
    v_color = particle_color;

    gl_Position = proj_matrix * view_matrix * pos;
}
