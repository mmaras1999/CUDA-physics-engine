#ifndef OBJECT_VOXELIZER
#define OBJECT_VOXELIZER

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
#include "gl_wrapper/texture.hpp"

class ObjectVoxelizer
{
public:
    ObjectVoxelizer(int screen_width, int screen_height, std::string object_path);
    ~ObjectVoxelizer();

    std::vector <glm::vec3> generateVoxels(GLFWwindow* glfw_window = nullptr);

private:
    int screen_width, screen_height;
    uint FBO;
    uint color_RBO, depth_RBO;
    uint objectVAO;
    uint objectVBO;
    Texture depth_textures[MAX_DEPTH_PEELING + 1];
    uint object_vertices_count;

    glm::vec3 object_center;
    glm::vec3 object_scale;
    glm::vec3 object_mass_center;

    void generateObject(std::string path);
};

#endif