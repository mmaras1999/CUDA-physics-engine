#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

uniform sampler2D depth_tex;
layout(location = 4) uniform vec2 viewport_size;
out vec4 color;

const float EPS = 1e-8;

void main(void) 
{
    vec2 tex_coord = vec2(gl_FragCoord.x / viewport_size.x, gl_FragCoord.y / viewport_size.y);

    if (gl_FragCoord.z - texture(depth_tex, tex_coord).x <= EPS)
    {
        discard;
    }

    color = vec4(1.0, 1.0, 1.0, 1.0);
}
