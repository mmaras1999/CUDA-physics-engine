#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

in vec4 pos;
in vec4 camera_relative_pos;
out vec4 color;

void main(void) 
{
    float fading = pow(0.87, length(camera_relative_pos.xyz));
    
    float x_d = floor((pos.x + 0.0001) / 0.2);
    float y_d = floor((pos.y + 0.0001) / 0.2);
    float z_d = floor((pos.z + 0.0001) / 0.2);

    color = vec4(0.7, 0.7, 0.7, 1.0);

    if (mod(x_d + y_d + z_d, 2.0) < 0.5)
    {
        color = vec4(0.5, 0.5, 0.5, 1.0);
    }
    
    color = mix(color, vec4(0.0, 0.0, 0.0, 1.0), 1.0 - fading);
}
