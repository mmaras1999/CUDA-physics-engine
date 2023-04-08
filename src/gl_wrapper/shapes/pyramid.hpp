#ifndef SHAPES_PYRAMID
#define SHAPES_PYRAMID

#include <epoxy/gl.h>
#include <epoxy/glx.h>

#include <vector>

namespace shapes
{
    std::vector <GLfloat> generatePyramid(int base_vertices, float base_radius, float height);
};

#endif
