#ifndef SHAPES_SPHERE
#define SHAPES_SPHERE

#include <vector>
#include <cmath>
#include <epoxy/gl.h>
#include <epoxy/glx.h>

namespace shapes
{
    std::vector <GLfloat> generateSphere(uint sectors, uint stacks, float radius);
}

#endif