#include <cmath>

#include "particle_simulation_state.hpp"

#include "managers/input_manager.hpp"
#include "managers/resource_manager.hpp"

#include "gl_wrapper/shapes/sphere.hpp"

#include "utility/random_generator.hpp"
#include "utility/quaternion_utils.hpp"
#include "utility/cuda_utilities.hpp"
#include "constants.hpp"
#include "simulation_funcs.hpp"

ParticleSimulationState::ParticleSimulationState(int screen_width, int screen_height, const std::string& object_path, const std::string& start_config_path)
    : Canvas(screen_width, screen_height),
      ended(false), 
      start_time(glfwGetTime()),
      mouse_enabled(LOCK_MOUSE)
{
    glCreateVertexArrays(1, &backgroundVAO);
    glCreateVertexArrays(1, &particleVAO);
    generateBackground();
    generateParticles();
    loadShaders();

    #ifdef UNIFORM_GRID
    cudaMalloc((void**)&part_grid_hashes_values, sizeof(uint) * PARTICLE_AMOUNT);
    cudaCheckError();
    cudaMalloc((void**)&part_grid_hashes_keys, sizeof(uint) * PARTICLE_AMOUNT);
    cudaCheckError();
    cudaMalloc((void**)&bucket_begin, sizeof(int) * MAX_BUCKETS);
    cudaCheckError();
    cudaMalloc((void**)&bucket_end, sizeof(int) * MAX_BUCKETS);
    cudaCheckError();
    #endif
}

void ParticleSimulationState::loadShaders()
{
    auto& resourceManager = ResourceManager::getInstance();
    resourceManager.loadShader(
        "bg",
        "background_vertex_shader.vsh",
        "background_fragment_shader.fsh"
    );

    resourceManager.loadShader(
        "particles",
        "particle_vertex_shader.vsh",
        "particle_fragment_shader.fsh"
    );
}

void ParticleSimulationState::generateBackground()
{
    glGenBuffers(1, &backgroundVBO);
    float boundary_cube[9 * 2 * 3] = {
        // back wall
        0.0f, 0.0f, 0.0f,
        0.0f,  10.0f, 0.0f,
        10.0f, 0.0f, 0.0f,

        10.0f,  10.0f, 0.0f,
        10.0f, 0.0f, 0.0f,
        0.0f,  10.0f, 0.0f,

        // front wall
        // 0.0f, 0.0f,  10.0f,
        // 0.0f,  10.0f,  10.0f,
        //  10.0f, 0.0f,  10.0f,

        //  10.0f,  10.0f,  10.0f,
        //  10.0f, 0.0f,  10.0f,
        // 0.0f,  10.0f,  10.0f,
        
        // floor
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f,  10.0f,
        10.0f, 0.0f, 0.0f,

        10.0f, 0.0f,  10.0f,
        10.0f, 0.0f, 0.0f,
        0.0f, 0.0f,  10.0f,

        // top
        // 0.0f,  10.0f, 0.0f,
        // 0.0f,  10.0f,  10.0f,
        //  10.0f,  10.0f, 0.0f,

        //  10.0f,  10.0f,  10.0f,
        //  10.0f,  10.0f, 0.0f,
        // 0.0f,  10.0f,  10.0f,

        // left wall
        0.0f, 0.0f, 0.0f,
        0.0f,  10.0f, 0.0f,
        0.0f, 0.0f,  10.0f,

        0.0f,  10.0f,  10.0f,
        0.0f, 0.0f,  10.0f,
        0.0f,  10.0f, 0.0f,

        // right wall
        //  10.0f, 0.0f, 0.0f,
        //  10.0f,  10.0f, 0.0f,
        //  10.0f, 0.0f,  10.0f,

        //  10.0f,  10.0f,  10.0f,
        //  10.0f, 0.0f,  10.0f,
        //  10.0f,  10.0f, 0.0f,
    };
    int boundary_cube_vertices = 9 * 3 * 2;

    glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GLfloat) * boundary_cube_vertices,
        boundary_cube,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSimulationState::generateParticles()
{
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
    glGenBuffers(1, &positionVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    float positions[3 * PARTICLE_AMOUNT];

    for (int i = 0; i < PARTICLE_AMOUNT; ++i)
    {
        positions[3 * i + 0] = 0.3 + 2 * PARTICLE_RADIUS * (i % 20);
        positions[3 * i + 1] = 0.3 + 2 * PARTICLE_RADIUS * (i / 20 % 20);
        positions[3 * i + 2] = 0.3 + 2 * PARTICLE_RADIUS * (i / 400);
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(positions),
        positions,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // GENERATE PARTICLE COLORS
    glGenBuffers(1, &colorsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
    float colors[PARTICLE_AMOUNT];

    for (int i = 0; i < PARTICLE_AMOUNT; ++i)
    {
        colors[i] = RandomGenerator::getInstance().random_real();
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(colors),
        colors,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float momenta[3 * PARTICLE_AMOUNT] = {};

    cudaMalloc((void**)&d_momenta, sizeof(float) * 3 * PARTICLE_AMOUNT);
    cudaCheckError();
    cudaMemcpy(d_momenta, momenta, sizeof(momenta), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMalloc((void**)&d_forces, sizeof(float) * 3 * PARTICLE_AMOUNT);
    cudaCheckError();

    cudaGraphicsGLRegisterBuffer(&positions_resource, positionVBO, cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
}

void ParticleSimulationState::update(double dt)
{
    dt *= TIME_SPEED;
    auto iDivUp = [](int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); };

    dim3 blockSize(1024, 1);  // threads
    dim3 gridSize(iDivUp(PARTICLE_AMOUNT, blockSize.x), 1);

    size_t size = 0;
    float* d_positions;
    cudaGraphicsMapResources(1, &positions_resource);
    cudaCheckError();
    cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &size, positions_resource);
    cudaCheckError();

    #ifdef BRUTE_FORCE
    update_forces_brute_force <<<gridSize, blockSize>>> (
        d_positions,
        d_momenta,
        d_forces,
        PARTICLE_AMOUNT,
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
    #endif

    #ifdef UNIFORM_GRID
    calc_grid_hashes <<<gridSize, blockSize>>> (
        d_positions, 
        part_grid_hashes_values,
        part_grid_hashes_keys, 
        PARTICLE_AMOUNT,
        MAX_BUCKETS);
    cudaDeviceSynchronize();
    cudaCheckError();

    // radix sort on GPU

    thrust::device_ptr<uint> d_ptr_hashes_keys(part_grid_hashes_keys);
    thrust::device_ptr<uint> d_ptr_hashes_values(part_grid_hashes_values);
    thrust::sort_by_key(d_ptr_hashes_keys, 
                        d_ptr_hashes_keys + PARTICLE_AMOUNT, 
                        d_ptr_hashes_values);
    cudaDeviceSynchronize();
    cudaCheckError();

    thrust::device_ptr<int> dev_b_begin(bucket_begin);
    thrust::device_ptr<int> dev_b_end(bucket_end);
    thrust::fill(dev_b_begin, dev_b_begin + MAX_BUCKETS, PARTICLE_AMOUNT);
    cudaDeviceSynchronize();
    cudaCheckError();

    thrust::fill(dev_b_end, dev_b_end + MAX_BUCKETS, PARTICLE_AMOUNT);
    cudaDeviceSynchronize();
    cudaCheckError();

    calc_bucket_ranges <<<gridSize, blockSize>>> (
        part_grid_hashes_keys,
        bucket_begin,
        bucket_end,
        PARTICLE_AMOUNT,
        MAX_BUCKETS
    );
    cudaDeviceSynchronize();
    cudaCheckError();

    calc_grid_collisions <<<gridSize, blockSize>>> (
        d_positions,
        d_momenta,
        d_forces,
        part_grid_hashes_values,
        part_grid_hashes_keys,
        bucket_begin,
        bucket_end,
        PARTICLE_AMOUNT,
        MAX_BUCKETS,
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
    #endif

    update_physics <<<gridSize, blockSize>>> (
        d_positions,
        d_momenta,
        d_forces,
        PARTICLE_AMOUNT,
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
    
    cudaGraphicsUnmapResources(1, &positions_resource);
    cudaDeviceSynchronize();
}

void ParticleSimulationState::draw()
{
    auto camera_position = glm::vec3(3.0f, 1.0f, 3.0f);
    auto camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
    auto camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
        
    auto view_mat = glm::lookAt(
        camera_position,
        camera_target,
        camera_up
    );

    auto proj_mat = glm::perspective(glm::radians(45.0f), (float)size.first / size.second, 0.01f, 10.0f);

    drawBackground(camera_position, view_mat, proj_mat);
    drawParticles(camera_position, view_mat, proj_mat);
}

void ParticleSimulationState::drawBackground(
    const glm::vec3& camera_position, 
    const glm::mat4& view_matrix, 
    const glm::mat4& proj_matrix)
{
    auto shader = ResourceManager::getInstance().getShader("bg");
    shader->use();
    
    glBindVertexArray(backgroundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(proj_matrix));
    glUniform3fv(3, 1, glm::value_ptr(camera_position));
    glDrawArrays(GL_TRIANGLES, 0, 9 * 3 * 2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void ParticleSimulationState::drawParticles(
    const glm::vec3& camera_position, 
    const glm::mat4& view_matrix, 
    const glm::mat4& proj_matrix)
{
    auto shader = ResourceManager::getInstance().getShader("particles");
    shader->use();

    glBindVertexArray(particleVAO);
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

    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
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

    glUniformMatrix4fv(3, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(4, 1, GL_FALSE, glm::value_ptr(proj_matrix));
    glUniform3fv(5, 1, glm::value_ptr(camera_position));

    glDrawArraysInstanced(GL_TRIANGLES, 0, sphere_vertices, PARTICLE_AMOUNT);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

int ParticleSimulationState::simEnded()
{
    return ended;
}

double ParticleSimulationState::getTime()
{
    return glfwGetTime() - start_time;
}

void ParticleSimulationState::freeBuffers()
{
    glDeleteBuffers(1, &backgroundVBO);
    glDeleteVertexArrays(1, &backgroundVAO);
    glDeleteVertexArrays(1, &particleVAO);

    cudaGraphicsUnregisterResource(positions_resource);
    cudaCheckError();
    cudaFree(d_forces);
    cudaCheckError();
    cudaFree(d_momenta);
    cudaCheckError();

    glDeleteBuffers(1, &colorsVBO);
    glDeleteBuffers(1, &positionVBO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteVertexArrays(1, &particleVAO);

    #ifdef UNIFORM_GRID
    cudaFree(part_grid_hashes_values);
    cudaCheckError();
    cudaFree(part_grid_hashes_keys);
    cudaCheckError();
    cudaFree(bucket_begin);
    cudaCheckError();
    cudaFree(bucket_end);
    cudaCheckError();
    #endif
}
