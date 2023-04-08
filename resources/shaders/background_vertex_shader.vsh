#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

layout(location = 0) in vec3 cube_data;
layout(location = 1) uniform mat4 view_matrix;
layout(location = 2) uniform mat4 proj_matrix;
layout(location = 3) uniform vec3 camera_position;

out vec4 pos;
out vec4 camera_relative_pos;

void main() {
    pos = vec4(cube_data, 1.0);
    camera_relative_pos = vec4(cube_data - camera_position, 1.0);
    gl_Position = proj_matrix * view_matrix * pos;
}
