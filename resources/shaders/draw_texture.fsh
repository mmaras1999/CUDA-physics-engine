#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

uniform sampler2D tex;
in vec2 tex_coords;
out vec4 color;

void main(void) 
{
    float z = texture(tex, tex_coords).x;

    if (z == 1.0)
    {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
       color = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), exp(z / 3.0) - 1.0);
    }
}
