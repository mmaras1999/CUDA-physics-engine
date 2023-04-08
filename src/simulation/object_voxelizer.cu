#include "object_voxelizer.hpp"
#include "managers/resource_manager.hpp"
#include "cuda_utilities.hpp"

#include <cuda_gl_interop.h>
#include <iomanip>

__global__ void voxelsInside(int* is_voxel_inside, int side_len, float* textures_data, int texture_width, int texture_height, float particle_radius)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    uint voxel_x = id % side_len;
    uint voxel_y = id / side_len;

    float voxel_pos_x = -0.5f + particle_radius + voxel_x * particle_radius * 2.0f;
    float voxel_pos_y = -0.5f + particle_radius + voxel_y * particle_radius * 2.0f;
    float voxel_pos_z = 0.0f + particle_radius;

    int intersection_cnt = 0; 
    int texture_id = 0;
    uint texture_id_x = (voxel_pos_x + 0.5f) * texture_width;
    uint texture_id_y = (voxel_pos_y + 0.5f) * texture_height;

    for (uint voxel_z = 0; voxel_z < side_len - 1; ++voxel_z)
    {
        while (texture_id < MAX_DEPTH_PEELING and textures_data[texture_width * texture_height * texture_id + texture_id_x + texture_id_y * texture_width] <= voxel_pos_z)
        {
            ++intersection_cnt;
            ++texture_id;
        }

        if (intersection_cnt % 2 == 1 and (texture_id == MAX_DEPTH_PEELING or textures_data[texture_width * texture_height * texture_id + texture_id_x + texture_id_y * texture_width] == 1.0f))
        {
            intersection_cnt = 0;
        }

        is_voxel_inside[voxel_x + voxel_y * side_len + voxel_z * side_len * side_len] = intersection_cnt % 2;
        voxel_pos_z += particle_radius * 2.0f;
    }
}

ObjectVoxelizer::ObjectVoxelizer(int screen_width, int screen_height, std::string object_path)
    : screen_width(screen_width), screen_height(screen_height)
{
    glCreateFramebuffers(1, &FBO);
    glCreateRenderbuffers(1, &color_RBO);

    for (int i = 0; i <= MAX_DEPTH_PEELING; ++i)
    {
        depth_textures[i].internal_format = GL_DEPTH_COMPONENT32F;
        depth_textures[i].image_format = GL_DEPTH_COMPONENT;
        depth_textures[i].wrap_s = GL_CLAMP_TO_EDGE;
        depth_textures[i].wrap_t = GL_CLAMP_TO_EDGE;
        depth_textures[i].filter_min = GL_NEAREST;
        depth_textures[i].filter_max = GL_NEAREST;
        depth_textures[i].data_type = GL_FLOAT;
        depth_textures[i].generate(screen_width, screen_height, NULL, false);
    }

    generateObject(object_path);

    ResourceManager::getInstance().loadShader(
        "depth_peeling",
        "depth_peeling_vertex.vsh",
        "depth_peeling_fragment.fsh"
    );

    ResourceManager::getInstance().loadShader(
        "debug_tex_draw",
        "draw_texture.vsh",
        "draw_texture.fsh"
    );
}

ObjectVoxelizer::~ObjectVoxelizer()
{
    glDeleteFramebuffers(1, &FBO);
    glDeleteRenderbuffers(1, &color_RBO);
    
    for (int i = 0; i < MAX_DEPTH_PEELING; ++i)
    {
        depth_textures[i].free();
    }

    glDeleteVertexArrays(1, &objectVAO);
    glDeleteBuffers(1, &objectVBO);
}

std::vector <glm::vec3> ObjectVoxelizer::generateVoxels(GLFWwindow* glfw_window)
{
    std::vector <glm::vec3> particles;
    float* texture_pointers[MAX_DEPTH_PEELING];

    // prepare camera
    auto camera_position = glm::vec3(0.0f, 0.0f, 0.5f);
    auto camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
    auto camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
        
    auto view_mat = glm::lookAt(
        camera_position,
        camera_target,
        camera_up
    );

    auto proj_mat = glm::ortho(-0.5f, 0.5f, -0.5f, 0.5f, -1e-4f, 1.0f);

    // prepare framebuffer
    glDisable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);

    for (int i = 1; i <= MAX_DEPTH_PEELING; ++i)
    {
        texture_pointers[i - 1] = new float[depth_textures[i].width * depth_textures[i].height];

        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glBindRenderbuffer(GL_RENDERBUFFER, color_RBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB16F, screen_width, screen_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_RBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_textures[i].ID, 0);
        
        if (int z = glCheckFramebufferStatus(GL_FRAMEBUFFER); z != GL_FRAMEBUFFER_COMPLETE)
            throw new std::runtime_error("ERROR:: Failed to initialize voxelizing FBO!");

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        depth_textures[i - 1].bind();

        // draw object
        auto shader = ResourceManager::getInstance().getShader("depth_peeling");
        shader->use();

        glm::mat4 model_matrix = glm::scale(glm::translate(glm::mat4(1.0f), -object_center), object_scale);
        
        glBindVertexArray(objectVAO);
        glBindBuffer(GL_ARRAY_BUFFER, objectVBO);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(model_matrix));
        glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(view_mat));
        glUniformMatrix4fv(3, 1, GL_FALSE, glm::value_ptr(proj_mat));
        float viewport_size[2] = {(float)screen_width, (float)screen_height};
        glUniform2fv(4, 1, viewport_size);
        glDrawArrays(GL_TRIANGLES, 0, object_vertices_count);

        glActiveTexture(GL_TEXTURE0);
        depth_textures[i - 1].unbind();

        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        glActiveTexture(GL_TEXTURE0);
        depth_textures[i].bind();
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, texture_pointers[i - 1]);

        // if (i == 3)
        // {
        //     shader = ResourceManager::getInstance().getShader("debug_tex_draw");
        //     shader->use();
        //     glDrawArrays(GL_TRIANGLES, 0, 6);

        //     while(true)
        //     {
        //         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //         glDrawArrays(GL_TRIANGLES, 0, 6);
        //         glfwSwapBuffers(glfw_window);
        //         glfwPollEvents();
        //     }
        // }

        glBindVertexArray(0);
    }

    glEnable(GL_MULTISAMPLE);

    // calculate voxels
    const float voxel_side_length = PARTICLE_RADIUS / OBJECT_SCALE * 2.0f;
    const int voxels_per_dim = 1.0f / voxel_side_length;

    int texture_size = depth_textures[0].width * depth_textures[0].height;
    int* is_voxel_inside = nullptr;
    int* is_voxel_inside_cpu = new int[voxels_per_dim * voxels_per_dim * voxels_per_dim ];
    float* textures_data = nullptr;
    cudaMalloc((void**)&is_voxel_inside, sizeof(int) * voxels_per_dim * voxels_per_dim * voxels_per_dim);
    cudaCheckError();
    cudaMalloc((void**)&textures_data, sizeof(float) * texture_size * MAX_DEPTH_PEELING);
    cudaCheckError();

    for (int i = 0; i < MAX_DEPTH_PEELING; ++i)
    {
        cudaMemcpy(textures_data + i * texture_size, texture_pointers[i], sizeof(float) * texture_size, cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    dim3 block_size(1024, 1);
    int wx = (voxels_per_dim * voxels_per_dim + block_size.x - 1) / block_size.x;
    dim3 grid_size = dim3(wx, 1);

    voxelsInside<<<grid_size, block_size>>>(is_voxel_inside, voxels_per_dim, textures_data, depth_textures[0].width, depth_textures[0].height, PARTICLE_RADIUS / OBJECT_SCALE);

    cudaMemcpy(is_voxel_inside_cpu, is_voxel_inside, sizeof(int) * voxels_per_dim * voxels_per_dim * voxels_per_dim, cudaMemcpyDeviceToHost);
    cudaCheckError();

    for (int x = 0; x < voxels_per_dim; ++x)
    {
        for (int y = 0; y < voxels_per_dim; ++y)
        {
            for (int z = 0; z < voxels_per_dim; ++z)
            {
                if (is_voxel_inside_cpu[x + y * voxels_per_dim + z * voxels_per_dim * voxels_per_dim])
                {
                    particles.emplace_back(
                       (-0.5f + PARTICLE_RADIUS / OBJECT_SCALE + x * PARTICLE_RADIUS / OBJECT_SCALE * 2.0f) * OBJECT_SCALE, 
                       (-0.5f + PARTICLE_RADIUS / OBJECT_SCALE + y * PARTICLE_RADIUS / OBJECT_SCALE * 2.0f) * OBJECT_SCALE, 
                       -1 * (-0.5f + PARTICLE_RADIUS / OBJECT_SCALE + z * PARTICLE_RADIUS / OBJECT_SCALE * 2.0f) * OBJECT_SCALE);  
                }
            }
        }
    }

    for (int i = 0; i < MAX_DEPTH_PEELING; ++i)
        delete[] texture_pointers[i];
    delete[] is_voxel_inside_cpu;

    return particles;
}

void ObjectVoxelizer::generateObject(std::string path)
{
    glCreateVertexArrays(1, &objectVAO);
    glBindVertexArray(objectVAO);
    glGenBuffers(1, &objectVBO);
    glBindBuffer(GL_ARRAY_BUFFER, objectVBO);
    
    auto [object_vertices, object_indices] = load_obj(path);

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
        object_mass_center += v;
    }

    auto diff = max_vertices - min_vertices;
    float max_s = std::max(diff.x, std::max(diff.y, diff.z));

    object_scale = glm::vec3((1.0f - PARTICLE_RADIUS / OBJECT_SCALE) / max_s);
    object_center = (max_vertices + min_vertices) * object_scale.x / 2.0f;
    object_mass_center /= object_vertices_count * object_scale.x;
    
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GLfloat) * object_vertices_count * 3,
        buff,
        GL_STATIC_DRAW
    );

    delete[] buff;
}