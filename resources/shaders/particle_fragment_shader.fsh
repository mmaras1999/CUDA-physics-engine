#version 330
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_shading_language_420pack : require

in vec3 camera_relative_pos;
in float v_color;
out vec4 color;

vec3 hsb2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0),6.0) - 3.0) - 1.0,
                     0.0,
                     1.0);

    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
    float fading = pow(0.87, length(camera_relative_pos.xyz));
    color = mix(
        vec4(hsb2rgb(vec3(v_color, 0.8, 0.8)), 1.0), 
        vec4(0.0, 0.0, 0.0, 1.0), 
        1.0 - fading
    );
}
