#include <stdexcept>
#include <cmath>
#include "pyramid.hpp"

std::vector <GLfloat> shapes::generatePyramid(int base_vertices, float base_radius, float height)
{
    if (base_vertices <= 2)
    {
        throw new std::logic_error("Bad number of base vertices for pyramid!");
    }

    struct Vertex
    {
        float x, y, z;
    };

    Vertex top = {0.0f, height / 2.0f, 0.0f};
    Vertex base = {0.0f, -height / 2.0f, 0.0f};

    std::vector <Vertex> base_v;

    float radius = base_radius;
    for (float angle = 0.0f; angle < 2 * M_PI - 1e-4f; angle += 2.0f / base_vertices * M_PI)
    {
        base_v.push_back(Vertex{radius * sinf(angle), -height / 2.0f, radius * cosf(angle)});
    }

    std::vector <GLfloat> triangle_vertices;

    // create base
    for (int i = 0; i < base_vertices; ++i)
    {
        // middle vertex
        triangle_vertices.push_back(base.x);
        triangle_vertices.push_back(base.y);
        triangle_vertices.push_back(base.z);

        // side vertices
        triangle_vertices.push_back(base_v[i].x);
        triangle_vertices.push_back(base_v[i].y);
        triangle_vertices.push_back(base_v[i].z);

        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].x);
        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].y);
        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].z);
    }

    // create sides
    for (int i = 0; i < base_vertices; ++i)
    {
        // top vertex
        triangle_vertices.push_back(top.x);
        triangle_vertices.push_back(top.y);
        triangle_vertices.push_back(top.z);

        // side vertices
        triangle_vertices.push_back(base_v[i].x);
        triangle_vertices.push_back(base_v[i].y);
        triangle_vertices.push_back(base_v[i].z);

        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].x);
        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].y);
        triangle_vertices.push_back(base_v[(i + 1) % base_vertices].z);
    }

    return triangle_vertices;
}