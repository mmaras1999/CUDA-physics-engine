#ifndef RIGID_BODY_SIMULATION_STATE
#define RIGID_BODY_SIMULATION_STATE

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
#include <string>
#include <vector>

#include "gl_wrapper/canvas.hpp"

class RigidBodySimulationState : public Canvas {
public:
    RigidBodySimulationState(int screen_width, int screen_height, const std::string object_path, std::string start_config_path);
    void update(double dt) override;
    void draw() override;
    int simEnded();
    double getTime();
    virtual void freeBuffers() override;

private:
    bool mouse_enabled;
    int ended;
    double start_time;

    // object data
    std::vector <glm::vec3> particles;
    uint particle_amount;
    uint object_amount;
    uint object_vertices_count;
    glm::vec3 object_mass_center;
    glm::vec3 object_center;
    glm::vec3 object_scale;

    // background
    uint backgroundVAO;
    uint backgroundVBO;

    // particles and objects
    uint objectVAO;
    uint objectVBO;
    uint positionVBO;
    uint quatVBO;
    uint colorsVBO;

    // physics
    float* d_inertia_tensor;
    float* d_forces;
    float* d_particle_positions;
    float* d_particle_relative_positions;
    float* d_particle_momenta;
    float* d_object_momenta;
    float* d_object_angular_momenta;
    float* d_positions;
    float* d_quats;

    bool resources_mapped;

    // uniform grid
    #ifdef UNIFORM_GRID
    uint* part_grid_hashes_values;
    uint* part_grid_hashes_keys;
    int* bucket_begin;
    int* bucket_end;
    #endif

    cudaGraphicsResource_t positions_resource;
    cudaGraphicsResource_t quat_resource;

    void generateBackground();
    void generateParticles(const std::string& object_path, const std::string& start_config_path);
    void loadShaders();
    void drawBackground(const glm::vec3& camera_position, 
                        const glm::mat4& view_matrix, 
                        const glm::mat4& proj_matrix);
    void drawObjects(const glm::vec3& camera_position, 
                       const glm::mat4& view_matrix, 
                       const glm::mat4& proj_matrix);
};

#endif