#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

layout(location = 0) in vec3 object_vertex;
layout(location = 1) in float object_color;
layout(location = 2) in vec3 object_position;
layout(location = 3) in vec4 object_quat;
layout(location = 4) uniform mat4 view_matrix;
layout(location = 5) uniform mat4 proj_matrix;
layout(location = 6) uniform vec3 camera_position;
layout(location = 7) uniform mat4 model_matrix;

out vec3 camera_relative_pos;
out float v_color;

vec4 quat_conj(vec4 q)
{ 
    return vec4(-q.x, -q.y, -q.z, q.w); 
}
  
vec4 quat_mult(vec4 q1, vec4 q2)
{ 
    return vec4(
        (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y),
        (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x),
        (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w),
        (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z)
    );
}

vec4 rotate_vector(vec4 v, vec4 quat)
{
    // if (length(quat) != 1.0)
    // {
    //     return vec4(1.0, 0.5, 1.0, 1.0);
    // }

    quat = normalize(quat);
    v.w = 0.0;
    vec4 rotated = quat_mult(quat_mult(quat, v), quat_conj(quat));
    rotated.w = 1.0;
    return rotated;
}

void main() {
    vec4 model_pos = rotate_vector(model_matrix * vec4(object_vertex, 1.0), object_quat);
    vec4 pos = model_pos + vec4(object_position, 0.0);
    camera_relative_pos = pos.xyz - camera_position;
    v_color = object_color;

    gl_Position = proj_matrix * view_matrix * pos;
}
