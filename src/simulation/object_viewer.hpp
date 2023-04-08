#ifndef OBJECT_VIEWER
#define OBJECT_VIEWER

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include "constants.hpp"
#include "utility/load_obj.hpp"

#include "gl_wrapper/canvas.hpp"

class ObjectViewer : public Canvas {
public:
    ObjectViewer(int screen_width, int screen_height, std::string object_path, const std::vector <glm::vec3>& particles);
    void update(double dt) override;
    void draw() override;
    double getTime();
    virtual void freeBuffers() override;

private:
    double start_time;
    float rotation_speed;
    glm::vec3 object_center;
    glm::vec3 object_scale;
    bool particle_view;

    // object
    uint objectVAO;
    uint objectVBO;
    uint object_vertices_count;
    // particles
    uint particlesVAO;
    uint sphereVBO;
    uint colorsVBO;
    uint particles_position_VBO;
    uint sphere_vertices;
    uint particles_size;

    void generateObject(std::string path);
    void generateParticles(const std::vector <glm::vec3>& particles);
    void loadShaders();
    void drawObject(const glm::vec3& camera_position, 
                    const glm::mat4& view_matrix, 
                    const glm::mat4& proj_matrix);
    void drawParticles(const glm::vec3& camera_position, 
                       const glm::mat4& view_matrix, 
                       const glm::mat4& proj_matrix);
};

#endif