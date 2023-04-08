#ifndef MODEL_UTILITIES
#define MODEL_UTILITIES

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <vector>

#include "geometry3d.hpp"

namespace ModelUtilities
{
    std::vector <GLfloat> calculateNormals(std::vector <GLfloat> vertices);
};

#endif
