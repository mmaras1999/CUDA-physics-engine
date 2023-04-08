#ifndef QUATERNION_UTILS
#define QUATERNION_UTILS

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

glm::quat RotationBetweenVectors(glm::vec3 start, glm::vec3 dest);

#endif