#ifndef PARTICLE_SIMULATION_STATE
#define PARTICLE_SIMULATION_STATE

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include "constants.hpp"

#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "gl_wrapper/canvas.hpp"

class ParticleSimulationState : public Canvas {
public:
    ParticleSimulationState(int screen_width, int screen_height, const std::string& object_path, const std::string& start_config_path);
    void update(double dt) override;
    void draw() override;
    int simEnded();
    double getTime();
    virtual void freeBuffers() override;

private:
    bool mouse_enabled;
    int ended;
    double start_time;

    // background
    uint backgroundVAO;
    uint backgroundVBO;

    // particles
    uint particleVAO;
    uint sphereVBO;
    uint sphere_vertices;
    uint positionVBO;
    uint colorsVBO;
    float* d_forces;
    float* d_momenta;

    // uniform grid
    #ifdef UNIFORM_GRID
    uint* part_grid_hashes_values;
    uint* part_grid_hashes_keys;
    int* bucket_begin;
    int* bucket_end;
    #endif

    cudaGraphicsResource_t positions_resource;

    void generateBackground();
    void generateParticles();
    void loadShaders();
    void drawBackground(const glm::vec3& camera_position, 
                        const glm::mat4& view_matrix, 
                        const glm::mat4& proj_matrix);
    void drawParticles(const glm::vec3& camera_position, 
                       const glm::mat4& view_matrix, 
                       const glm::mat4& proj_matrix);
};

#endif