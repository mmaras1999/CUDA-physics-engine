#include "quaternion_utils.hpp"

glm::quat RotationBetweenVectors(glm::vec3 start, glm::vec3 dest)
{
	using namespace glm;

	start = normalize(start);
	dest = normalize(dest);
	
	float cosTheta = dot(start, dest);
	vec3 rotationAxis;
	
	if (cosTheta < -1 + 0.001f)
    {
		rotationAxis = cross(vec3(0.0f, 0.0f, 1.0f), start);
		if (length2(rotationAxis) < 0.01) 
			rotationAxis = cross(vec3(1.0f, 0.0f, 0.0f), start);
		
		rotationAxis = normalize(rotationAxis);
		return angleAxis(glm::radians(180.0f), rotationAxis);
	}

	rotationAxis = cross(start, dest);

	float s = sqrt( (1+cosTheta)*2 );
	float invs = 1 / s;

	return glm::normalize(quat(
		s * 0.5f, 
		rotationAxis.x * invs,
		rotationAxis.y * invs,
		rotationAxis.z * invs
	));
}