#include "object_viewer.hpp"

#include <cmath>

#include "managers/input_manager.hpp"
#include "managers/resource_manager.hpp"

#include "utility/random_generator.hpp"
#include "utility/quaternion_utils.hpp"
#include "gl_wrapper/shapes/sphere.hpp"
#include "constants.hpp"

ObjectViewer::ObjectViewer(int screen_width, int screen_height, std::string object_path, const std::vector <glm::vec3>& particles)
    : Canvas(screen_width, screen_height),
      start_time(glfwGetTime()),
      rotation_speed(0.1f),
      object_center(0.0f, 0.0f, 0.0f),
      particle_view(false)
{
    glCreateVertexArrays(1, &objectVAO);
    glCreateVertexArrays(1, &particlesVAO);
    generateObject(object_path);
    generateParticles(particles);
    loadShaders();
}

void ObjectViewer::loadShaders()
{
    auto& resourceManager = ResourceManager::getInstance();
    resourceManager.loadShader(
        "simple",
        "simple_vertex_shader.vsh",
        "simple_fragment_shader.fsh"
    );

    resourceManager.loadShader(
        "particles",
        "simple_particle_vertex_shader.vsh",
        "simple_particle_fragment_shader.fsh"
    );
}

void ObjectViewer::generateParticles(const std::vector <glm::vec3>& particles)
{
    particles_size = particles.size();
    glBindVertexArray(particlesVAO);
    // GENERATE SPHERE
    glGenBuffers(1, &sphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    auto sphere_v = shapes::generateSphere(24, 16, PARTICLE_RADIUS);
    GLfloat sphere[sphere_v.size()];
    
    for (int i = 0; i < (int)sphere_v.size(); ++i)
    {
        sphere[i] = sphere_v[i];
    }

    sphere_vertices = sphere_v.size();

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(sphere),
        sphere,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // GENERATE PARTICLE POSITIONS
    glGenBuffers(1, &particles_position_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_VBO);
    float* positions = new float[3 * particles.size()];

    for (int i = 0; i < particles.size(); ++i)
    {
        positions[3 * i + 0] = particles[i].x;
        positions[3 * i + 1] = particles[i].y;
        positions[3 * i + 2] = particles[i].z;
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * 3 * particles.size(),
        positions,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // GENERATE PARTICLE COLORS
    glGenBuffers(1, &colorsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
    float* colors = new float[particles.size()];

    for (int i = 0; i < particles.size(); ++i)
    {
        colors[i] = RandomGenerator::getInstance().random_real();
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * particles.size(),
        colors,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    delete[] colors;
    delete[] positions;
}

void ObjectViewer::generateObject(std::string path)
{
    glCreateVertexArrays(1, &objectVAO);
    glBindVertexArray(objectVAO);
    glGenBuffers(1, &objectVBO);
    glBindBuffer(GL_ARRAY_BUFFER, objectVBO);
    
    auto [object_vertices, object_indices] = load_obj(path);

    object_vertices_count = object_indices.size();

    glm::vec3 min_vertices(1000000.0f), max_vertices(-1000000.0f);

    float* buff = new float[object_vertices_count * 3];
    for (int i = 0; i < object_vertices_count; ++i)
    {
        buff[3 * i + 0] = object_vertices[3 * (object_indices[i] - 1) + 0];
        buff[3 * i + 1] = object_vertices[3 * (object_indices[i] - 1) + 1];
        buff[3 * i + 2] = object_vertices[3 * (object_indices[i] - 1) + 2];

        auto v = glm::vec3(buff[3 * i + 0], buff[3 * i + 1], buff[3 * i + 2]);

        min_vertices = glm::min(min_vertices, v);
        max_vertices = glm::max(max_vertices, v);
    }

    auto diff = max_vertices - min_vertices;
    float max_s = std::max(diff.x, std::max(diff.y, diff.z));

    object_scale = glm::vec3((1.0f - PARTICLE_RADIUS / OBJECT_SCALE) / max_s) * OBJECT_SCALE;
    object_center = (max_vertices + min_vertices) * object_scale.x / 2.0f;
    
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GLfloat) * object_vertices_count * 3,
        buff,
        GL_STATIC_DRAW
    );

    delete[] buff;
}

void ObjectViewer::update(double dt)
{
    if (InputManager::getInstance().keyPressed(GLFW_KEY_TAB))
    {
        particle_view = !particle_view;
    }
}

void ObjectViewer::draw()
{
    auto camera_position = glm::vec3(4.0f, 0.0f, 3.0f);
    auto camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
    auto camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
        
    auto view_mat = glm::lookAt(
        camera_position,
        camera_target,
        camera_up
    );

    auto proj_mat = glm::perspective(glm::radians(45.0f), (float)size.first / size.second, 0.01f, 100.0f);

    if (!particle_view)
    {
        drawObject(camera_position, view_mat, proj_mat);
    }
    else
    {
        drawParticles(camera_position, view_mat, proj_mat);
    }
}

void ObjectViewer::drawObject(
    const glm::vec3& camera_position, 
    const glm::mat4& view_matrix, 
    const glm::mat4& proj_matrix)
{
    auto shader = ResourceManager::getInstance().getShader("simple");
    shader->use();

    float model_scale = 3.0f;
    glm::quat rotation = glm::angleAxis(rotation_speed * ((float)getTime() + 0.0001f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 model_matrix = glm::toMat4(rotation) * glm::scale(glm::translate(glm::mat4(1.0f), -object_center * model_scale), object_scale * model_scale);
    glm::vec3 color = glm::vec3(0.0f, 0.8f, 0.8f);
    
    glBindVertexArray(objectVAO);
    glBindBuffer(GL_ARRAY_BUFFER, objectVBO);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(model_matrix));
    glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(3, 1, GL_FALSE, glm::value_ptr(proj_matrix));
    glUniform3fv(4, 1, glm::value_ptr(camera_position));
    glUniform3fv(5, 1, glm::value_ptr(color));

    glDrawArrays(GL_TRIANGLES, 0, object_vertices_count);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void ObjectViewer::drawParticles(
    const glm::vec3& camera_position, 
    const glm::mat4& view_matrix, 
    const glm::mat4& proj_matrix)
{
    auto shader = ResourceManager::getInstance().getShader("particles");
    shader->use();

    glBindVertexArray(particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        1,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glBindBuffer(GL_ARRAY_BUFFER, particles_position_VBO);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glVertexAttribDivisor(1, 1);
    glVertexAttribDivisor(2, 1);

    float model_scale = 3.0f;
    glm::quat rotation = glm::angleAxis(rotation_speed * ((float)getTime() + 0.0001f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 model_matrix = glm::scale(glm::mat4_cast(rotation), glm::vec3(model_scale));

    glUniformMatrix4fv(3, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(4, 1, GL_FALSE, glm::value_ptr(proj_matrix));
    glUniform3fv(5, 1, glm::value_ptr(camera_position));
    glUniformMatrix4fv(6, 1, GL_FALSE, glm::value_ptr(model_matrix));

    glDrawArraysInstanced(GL_TRIANGLES, 0, sphere_vertices, particles_size);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

double ObjectViewer::getTime()
{
    return glfwGetTime() - start_time;
}

void ObjectViewer::freeBuffers()
{
    glDeleteBuffers(1, &objectVBO);
    glDeleteVertexArrays(1, &objectVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &particles_position_VBO);
    glDeleteVertexArrays(1, &particlesVAO);
    glDeleteBuffers(1, &colorsVBO);
}
