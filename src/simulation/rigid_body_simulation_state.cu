#include <cmath>
#include <fstream>

#include "rigid_body_simulation_state.hpp"

#include "managers/input_manager.hpp"
#include "managers/resource_manager.hpp"

#include "gl_wrapper/shapes/sphere.hpp"

#include "utility/random_generator.hpp"
#include "utility/quaternion_utils.hpp"
#include "utility/cuda_utilities.hpp"
#include "constants.hpp"
#include "simulation_funcs.hpp"
#include "object_voxelizer.hpp"

RigidBodySimulationState::RigidBodySimulationState(int screen_width, int screen_height, const std::string object_path, std::string start_config_path)
    : Canvas(screen_width, screen_height),
      ended(false), 
      start_time(glfwGetTime()),
      mouse_enabled(LOCK_MOUSE)
{
    // voxelize the object
    auto voxelizer = ObjectVoxelizer(screen_width, screen_height, object_path);
    particles = voxelizer.generateVoxels();

    if (particles.size() == 0)
    {
        std::cout << "Error! No particles generated! Try increasing the object size or decreasing the particle size!";
        exit(1);
    }

    resources_mapped = false;

    glCreateVertexArrays(1, &backgroundVAO);
    glCreateVertexArrays(1, &objectVAO);
    generateBackground();
    generateParticles(object_path, start_config_path);
    loadShaders();

    #ifdef UNIFORM_GRID
    cudaMalloc((void**)&part_grid_hashes_values, sizeof(uint) * particle_amount);
    cudaCheckError();
    cudaMalloc((void**)&part_grid_hashes_keys, sizeof(uint) * particle_amount);
    cudaCheckError();
    cudaMalloc((void**)&bucket_begin, sizeof(int) * MAX_BUCKETS);
    cudaCheckError();
    cudaMalloc((void**)&bucket_end, sizeof(int) * MAX_BUCKETS);
    cudaCheckError();
    #endif

    start_time = glfwGetTime();
}

void RigidBodySimulationState::loadShaders()
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

    resourceManager.loadShader(
        "objects",
        "objects_vertex_shader.vsh",
        "objects_fragment_shader.fsh"
    );
}

void RigidBodySimulationState::generateBackground()
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

void RigidBodySimulationState::generateParticles(const std::string& object_path, const std::string& start_config_path)
{
    // READ CONFIGURATION FILE
    std::ifstream config_file(start_config_path);
    config_file >> object_amount;
    particle_amount = object_amount * particles.size();

    std::vector <glm::vec3> object_positions;

    for (int i = 0; i < object_amount; ++i)
    {
        float x, y, z;
        config_file >> x >> y >> z;

        object_positions.emplace_back(x, y, z);
    }

    config_file.close();

    // GENERATE MODEL
    glCreateVertexArrays(1, &objectVAO);
    glBindVertexArray(objectVAO);
    glGenBuffers(1, &objectVBO);
    glBindBuffer(GL_ARRAY_BUFFER, objectVBO);
    
    auto [object_vertices, object_indices] = load_obj(object_path);

    object_vertices_count = object_indices.size();

    glm::vec3 min_vertices(1000000.0f), max_vertices(-1000000.0f);
    object_mass_center = glm::vec3(0.0f);

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

    for (auto& particle : particles)
    {
        object_mass_center += particle;
    }

    object_mass_center /= particles.size();

    for (auto& particle : particles)
    {
        particle -= object_mass_center;
    }
    
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GLfloat) * object_vertices_count * 3,
        buff,
        GL_STATIC_DRAW
    );

    delete[] buff;
    
    // GENERATE OBJECT POSITIONS
    glGenBuffers(1, &positionVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    float* positions = new float[3 * object_amount];

    for (int i = 0; i < object_amount; ++i)
    {
        positions[3 * i + 0] = object_positions[i].x;
        positions[3 * i + 1] = object_positions[i].y;
        positions[3 * i + 2] = object_positions[i].z;
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * 3 * object_amount,
        positions,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    delete[] positions;

    // GENERATE OBJECT QUATERNIONS
    glGenBuffers(1, &quatVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quatVBO);
    float* quaternions = new float[4 * object_amount];
    for (int i = 0; i < object_amount; ++i)
    {
        quaternions[4 * i + 0] = 0.0f; 
        quaternions[4 * i + 1] = 0.0f;
        quaternions[4 * i + 2] = 0.0f;
        quaternions[4 * i + 3] = 1.0f;
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * 4 * object_amount,
        quaternions,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    delete[] quaternions;

    // GENERATE OBJECT COLORS
    glGenBuffers(1, &colorsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
    float* colors = new float[object_amount];

    for (int i = 0; i < object_amount; ++i)
    {
        colors[i] = RandomGenerator::getInstance().random_real();
    }

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * object_amount,
        colors,
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    delete[] colors;
        
    // CALCULATE PARTICLE PHYSICS
    float* object_momenta = new float[3 * object_amount];
    float* object_angular_momenta = new float[3 * object_amount];
    for (int i = 0; i < 3 * object_amount; ++i)
    {
        object_momenta[i] = 0.0f;
        object_angular_momenta[i] = 0.0f;
    }

    cudaMalloc((void**)&d_particle_momenta, sizeof(float) * 3 * particle_amount);
    cudaCheckError();
    cudaMalloc((void**)&d_particle_positions, sizeof(float) * 3 * particle_amount);
    cudaCheckError();
    cudaMalloc((void**)&d_forces, sizeof(float) * 3 * particle_amount);
    cudaCheckError();
    cudaMalloc((void**)&d_object_momenta, sizeof(float) * 3 * object_amount);
    cudaCheckError();
    cudaMalloc((void**)&d_object_angular_momenta, sizeof(float) * 3 * object_amount);
    cudaCheckError();
    cudaMalloc((void**)&d_particle_relative_positions, sizeof(float) * 3 * particles.size());
    cudaCheckError();

    float* particles_rel_pos = new float[3 * particles.size()];
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        particles_rel_pos[3 * i + 0] = particles[i].x;
        particles_rel_pos[3 * i + 1] = particles[i].y;
        particles_rel_pos[3 * i + 2] = particles[i].z;
    }

    cudaMemcpy(d_particle_relative_positions, particles_rel_pos, sizeof(float) * 3 * particles.size(), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_object_momenta, object_momenta, sizeof(float) * 3 * object_amount, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_object_angular_momenta, object_angular_momenta, sizeof(float) * 3 * object_amount, cudaMemcpyHostToDevice);
    cudaCheckError();
    delete[] particles_rel_pos;
    delete[] object_momenta;
    delete[] object_angular_momenta;

    glm::mat3 inertia_tensor(0.0f);

    for (auto& particle : particles)
    {
        float* p = &particle.x;

        inertia_tensor[0][0] += (p[1] * p[1] + p[2] * p[2]);
        inertia_tensor[0][1] += -(p[0] * p[1]);
        inertia_tensor[0][2] += -(p[0] * p[2]);

        inertia_tensor[1][0] += -(p[0] * p[1]);
        inertia_tensor[1][1] += (p[0] * p[0] + p[2] * p[2]);
        inertia_tensor[1][2] += -p[1] * p[2];

        inertia_tensor[2][0] += -(p[0] * p[2]);
        inertia_tensor[2][1] += -p[1] * p[2];
        inertia_tensor[2][2] += (p[0] * p[0] + p[1] * p[1]);
    }

    inertia_tensor = glm::inverse(PARTICLE_MASS * inertia_tensor);
    cudaMalloc((void**)&d_inertia_tensor, sizeof(float) * 3 * 3);
    cudaCheckError();
    cudaMemcpy(d_inertia_tensor, glm::value_ptr(inertia_tensor), sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaGraphicsGLRegisterBuffer(&positions_resource, positionVBO, cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
    cudaGraphicsGLRegisterBuffer(&quat_resource, quatVBO, cudaGraphicsRegisterFlagsNone);
    cudaCheckError();

    std::cout << "Initialization done!" << std::endl;
}

void RigidBodySimulationState::update(double dt)
{
    dt *= TIME_SPEED;
    
    static const auto iDivUp = [](int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); };
    static const dim3 blockSize(1024, 1);  // threads
    static const dim3 gridSize(iDivUp(particle_amount, blockSize.x), 1);
    static const dim3 gridSize2(iDivUp(object_amount, blockSize.x), 1);
    size_t size = 0;

    if(!resources_mapped)
    {
        cudaGraphicsMapResources(1, &positions_resource);
        cudaCheckError();
        cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &size, positions_resource);
        cudaCheckError(); 
        cudaGraphicsMapResources(1, &quat_resource);
        cudaCheckError();
        cudaGraphicsResourceGetMappedPointer((void**)&d_quats, &size, quat_resource);
        cudaCheckError();

        resources_mapped = true;
    }

    calculate_particle_values <<<gridSize, blockSize>>> (
        d_positions,
        d_quats,
        d_object_momenta,
        d_object_angular_momenta,
        d_particle_relative_positions,
        d_particle_positions,
        d_particle_momenta,
        d_inertia_tensor,
        object_amount,
        particles.size()
    );
    cudaDeviceSynchronize();
    cudaCheckError();

    #ifdef BRUTE_FORCE
    update_forces_brute_force <<<gridSize, blockSize>>> (
        d_particle_positions,
        d_particle_momenta,
        d_forces,
        particle_amount,
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
    #endif

    #ifdef UNIFORM_GRID
    calc_grid_hashes <<<gridSize, blockSize>>> (
        d_particle_positions, 
        part_grid_hashes_values,
        part_grid_hashes_keys, 
        particle_amount,
        MAX_BUCKETS);
    cudaDeviceSynchronize();
    cudaCheckError();

    // radix sort on GPU

    thrust::device_ptr<uint> d_ptr_hashes_keys(part_grid_hashes_keys);
    thrust::device_ptr<uint> d_ptr_hashes_values(part_grid_hashes_values);
    thrust::sort_by_key(d_ptr_hashes_keys, 
                        d_ptr_hashes_keys + particle_amount, 
                        d_ptr_hashes_values);
    cudaDeviceSynchronize();
    cudaCheckError();

    thrust::device_ptr<int> dev_b_begin(bucket_begin);
    thrust::device_ptr<int> dev_b_end(bucket_end);
    thrust::fill(dev_b_begin, dev_b_begin + MAX_BUCKETS, particle_amount);
    cudaDeviceSynchronize();
    cudaCheckError();

    thrust::fill(dev_b_end, dev_b_end + MAX_BUCKETS, particle_amount);
    cudaDeviceSynchronize();
    cudaCheckError();

    calc_bucket_ranges <<<gridSize, blockSize>>> (
        part_grid_hashes_keys,
        bucket_begin,
        bucket_end,
        particle_amount,
        MAX_BUCKETS
    );
    cudaDeviceSynchronize();
    cudaCheckError();

    calc_grid_collisions <<<gridSize, blockSize>>> (
        d_particle_positions,
        d_particle_momenta,
        d_forces,
        part_grid_hashes_values,
        part_grid_hashes_keys,
        bucket_begin,
        bucket_end,
        particle_amount,
        MAX_BUCKETS,
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
    #endif

    update_object_momenta <<<gridSize, blockSize>>> (
        d_forces,
        d_particle_relative_positions,
        d_quats,
        d_object_momenta,
        d_object_angular_momenta,
        object_amount,
        particles.size(),
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();

    update_physics_with_rotation <<<gridSize2, blockSize>>> (
        d_positions,
        d_quats,
        d_object_momenta,
        d_object_angular_momenta,
        d_inertia_tensor,
        object_amount,
        particles.size(),
        dt
    );
    cudaDeviceSynchronize();
    cudaCheckError();
}

void RigidBodySimulationState::draw()
{
    if(resources_mapped)
    {    
        cudaGraphicsUnmapResources(1, &positions_resource);
        cudaCheckError();
        cudaGraphicsUnmapResources(1, &quat_resource);
        cudaCheckError();

        resources_mapped = false;
    }

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
    drawObjects(camera_position, view_mat, proj_mat);
}

void RigidBodySimulationState::drawBackground(
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

void RigidBodySimulationState::drawObjects(
    const glm::vec3& camera_position, 
    const glm::mat4& view_matrix, 
    const glm::mat4& proj_matrix)
{
    auto shader = ResourceManager::getInstance().getShader("objects");
    shader->use();

    auto model_matrix = glm::scale(glm::translate(glm::mat4(1.0f), -object_center - object_mass_center), object_scale);

    glBindVertexArray(objectVAO);
    glBindBuffer(GL_ARRAY_BUFFER, objectVBO);
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

    glBindBuffer(GL_ARRAY_BUFFER, quatVBO);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(
        3,
        4,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glVertexAttribDivisor(1, 1);
    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);

    glUniformMatrix4fv(4, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(5, 1, GL_FALSE, glm::value_ptr(proj_matrix));
    glUniform3fv(6, 1, glm::value_ptr(camera_position));
    glUniformMatrix4fv(7, 1, GL_FALSE, glm::value_ptr(model_matrix));

    glDrawArraysInstanced(GL_TRIANGLES, 0, object_vertices_count, object_amount);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

int RigidBodySimulationState::simEnded()
{
    return ended;
}

double RigidBodySimulationState::getTime()
{
    return glfwGetTime() - start_time;
}

void RigidBodySimulationState::freeBuffers()
{
    if(resources_mapped)
    {    
        cudaGraphicsUnmapResources(1, &positions_resource);
        cudaCheckError();
        cudaGraphicsUnmapResources(1, &quat_resource);
        cudaCheckError();

        resources_mapped = false;
    }

    glDeleteBuffers(1, &backgroundVBO);
    glDeleteVertexArrays(1, &backgroundVAO);

    glDeleteVertexArrays(1, &objectVAO);
    glDeleteBuffers(1, &objectVBO);
    glDeleteBuffers(1, &positionVBO);
    glDeleteBuffers(1, &quatVBO);
    glDeleteBuffers(1, &colorsVBO);

    cudaGraphicsUnregisterResource(positions_resource);
    cudaCheckError();
    cudaGraphicsUnregisterResource(quat_resource);
    cudaFree(d_forces);
    cudaCheckError();
    cudaFree(d_particle_positions);
    cudaCheckError();
    cudaFree(d_particle_relative_positions);
    cudaCheckError();
    cudaFree(d_particle_momenta);
    cudaCheckError();
    cudaFree(d_object_momenta);
    cudaCheckError();
    cudaFree(d_object_angular_momenta);
    cudaCheckError();

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
