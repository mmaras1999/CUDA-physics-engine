// CODE FROM https://github.com/gszauer/GamePhysicsCookbook
#ifndef GEOMETRY_3D
#define GEOMETRY_3D

#include <glm/glm.hpp>

namespace Geometry3d
{
    struct Triangle
    {
        glm::vec3 a, b, c;
        glm::vec3 getNormal();
    };
    
    struct Sphere
    {
        glm::vec3 center;
        float radius;
    };

    struct Plane
    {
        glm::vec3 normal;
        float distance; 

        Plane();
        Plane(const glm::vec3& normal, float distance);
        Plane(const Triangle& triangle);
    };

    struct Line
    {
        glm::vec3 start, end;

        Line(const glm::vec3& a, const glm::vec3& b);
    };

    bool pointInTriangle(const glm::vec3& point, const Triangle& triangle);
    
    glm::vec3 closestPoint(const Plane& plane, const glm::vec3& point);
    glm::vec3 closestPoint(const Line& line, const glm::vec3& point);
    glm::vec3 closestPoint(const Triangle& triangle, const glm::vec3& point);
    bool intersects(const Sphere& sphere, const Triangle& triangle);
    bool intersects(const Sphere& s1, const Sphere& s2);
};

#endif