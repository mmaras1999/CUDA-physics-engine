#include "sphere.hpp"

std::vector <GLfloat> shapes::generateSphere(uint sectors, uint stacks, float radius)
{
    struct vertex
    {
        float x, y, z;
    };

    std::vector<vertex> vertices;

    float sectorStep = 2 * M_PI / sectors;
    float stackStep = M_PI / stacks;

    // generate vertices
    for(int i = 0; i <= stacks; ++i)
    {
        float stackAngle = M_PI / 2 - i * stackStep; // starting from pi/2 to -pi/2
        float xy = radius * cosf(stackAngle);        // r * cos(u)
        float z = radius * sinf(stackAngle);         // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for(int j = 0; j <= sectors; ++j)
        {
            float sectorAngle = j * sectorStep;      // starting from 0 to 2pi

            // vertex position (x, y, z)
            float x = xy * cosf(sectorAngle);        // r * cos(u) * cos(v)
            float y = xy * sinf(sectorAngle);        // r * cos(u) * sin(v)
            vertices.push_back({x, y, z});
        }
    }

    std::vector<GLfloat> triangle_vertices;
    
    // triangulate
    for(int i = 0; i < stacks; ++i)
    {
        int k1 = i * (sectors + 1);  // beginning of current stack
        int k2 = k1 + sectors + 1;   // beginning of next stack

        for(int j = 0; j < sectors; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1 + 1
            if(i != 0)
            {
                // vertex k1
                triangle_vertices.push_back(vertices[k1].x);   
                triangle_vertices.push_back(vertices[k1].y);
                triangle_vertices.push_back(vertices[k1].z);
                // vertex k2
                triangle_vertices.push_back(vertices[k2].x);   
                triangle_vertices.push_back(vertices[k2].y);
                triangle_vertices.push_back(vertices[k2].z);
                // vertex k1 + 1
                triangle_vertices.push_back(vertices[k1 + 1].x);   
                triangle_vertices.push_back(vertices[k1 + 1].y);
                triangle_vertices.push_back(vertices[k1 + 1].z);
            }

            // k1 + 1 => k2 => k2 + 1
            if(i != (stacks - 1))
            {
                // vertex k1 + 1
                triangle_vertices.push_back(vertices[k1 + 1].x);   
                triangle_vertices.push_back(vertices[k1 + 1].y);
                triangle_vertices.push_back(vertices[k1 + 1].z);
                // vertex k2
                triangle_vertices.push_back(vertices[k2].x);   
                triangle_vertices.push_back(vertices[k2].y);
                triangle_vertices.push_back(vertices[k2].z);
                // vertex k2 + 1
                triangle_vertices.push_back(vertices[k2 + 1].x);   
                triangle_vertices.push_back(vertices[k2 + 1].y);
                triangle_vertices.push_back(vertices[k2 + 1].z);
            }
        }
    }

    return triangle_vertices;
}