#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

layout(location = 5) uniform vec3 object_color;

in vec4 camera_relative_pos;
out vec4 color;

void main(void) 
{
    float fading = pow(0.89, length(camera_relative_pos.xyz));
    
    color = mix(vec4(object_color, 1.0), vec4(0.0, 0.0, 0.0, 1.0), 1.0 - fading);
}
