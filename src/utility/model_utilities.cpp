#include "model_utilities.hpp"

std::vector <GLfloat> ModelUtilities::calculateNormals(std::vector <GLfloat> vertices)
{
    std::vector <GLfloat> normals;
    normals.reserve(vertices.size());

    for (int i = 0; i < (int)vertices.size(); i += 9)
    {
        glm::vec3 a = glm::vec3(vertices[i], vertices[i + 1], vertices[i + 2]);
        glm::vec3 b = glm::vec3(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
        glm::vec3 c = glm::vec3(vertices[i + 6], vertices[i + 7], vertices[i + 8]);

        Geometry3d::Triangle tri = {a, b, c};
        auto normal = glm::normalize(tri.getNormal());

        for (int j = 0; j < 3; ++j)
        {
            normals.push_back(normal.x);
            normals.push_back(normal.y);
            normals.push_back(normal.z);
        }
    }

    return normals;
}