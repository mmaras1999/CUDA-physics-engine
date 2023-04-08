// CODE FROM https://github.com/gszauer/GamePhysicsCookbook
#include "geometry3d.hpp"

glm::vec3 Geometry3d::Triangle::getNormal()
{
	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	return glm::normalize(glm::cross(edge1, edge2));
}

Geometry3d::Plane::Plane() 
    : normal(1.0f, 0.0f, 0.0f), distance(0.0f)
{

}

Geometry3d::Plane::Plane(const glm::vec3& normal, float distance)
    : normal(normal), distance(distance)
{

}

Geometry3d::Plane::Plane(const Triangle& triangle)
{
	normal = glm::normalize(glm::cross(triangle.b - triangle.a, triangle.c - triangle.a));
	distance = glm::dot(normal, triangle.a);
}

Geometry3d::Line::Line(const glm::vec3& a, const glm::vec3& b)
    : start(a), end(b)
{

}

bool Geometry3d::pointInTriangle(const glm::vec3& point, const Triangle& triangle)
{
	glm::vec3 a = triangle.a - point;
	glm::vec3 b = triangle.b - point;
	glm::vec3 c = triangle.c - point;

	glm::vec3 normPBC = glm::cross(b, c);
	glm::vec3 normPCA = glm::cross(c, a);
	glm::vec3 normPAB = glm::cross(a, b);

	if (glm::dot(normPBC, normPCA) < 0.0f) {
		return false;
	}
	else if (glm::dot(normPBC, normPAB) < 0.0f) {
		return false;
	}

	return true;
}
    
glm::vec3 Geometry3d::closestPoint(const Plane& plane, const glm::vec3& point)
{
	float distance = glm::dot(plane.normal, point) - plane.distance;
	return point - plane.normal * distance;
}

glm::vec3 Geometry3d::closestPoint(const Line& line, const glm::vec3& point)
{
    glm::vec3 lVec = line.end - line.start;
	float t = glm::dot(point - line.start, lVec) / glm::dot(lVec, lVec);
	t = std::max(t, 0.0f);
	t = std::min(t, 1.0f);
	return line.start + lVec * t;
}

glm::vec3 Geometry3d::closestPoint(const Triangle& triangle, const glm::vec3& point)
{
    Plane plane(triangle);
	glm::vec3 closest = closestPoint(plane, point);

	// Closest point was inside triangle
	if (pointInTriangle(closest, triangle)) {
		return closest;
	}

	glm::vec3 c1 = closestPoint(Line(triangle.a, triangle.b), closest);
	glm::vec3 c2 = closestPoint(Line(triangle.b, triangle.c), closest);
	glm::vec3 c3 = closestPoint(Line(triangle.c, triangle.a), closest);

	float l1 = glm::length(closest - c1);
	float l2 = glm::length(closest - c2);
	float l3 = glm::length(closest - c3);

	if (l1 < l2 && l1 < l3) {
		return c1;
	}
	else if (l2 < l1 && l2 < l3) {
		return c2;
	}

	return c3;
}

bool Geometry3d::intersects(const Sphere& sphere, const Triangle& triangle)
{
    auto closest = closestPoint(triangle, sphere.center);
	
	return glm::length(closest - sphere.center) <= sphere.radius;
}

bool Geometry3d::intersects(const Sphere& s1, const Sphere& s2)
{
	float dist = glm::length(s1.center - s2.center);

	return dist <= s1.radius + s2.radius;
}
