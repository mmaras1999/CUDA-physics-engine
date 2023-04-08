#ifndef RESOURCE_MANAGER
#define RESOURCE_MANAGER

#include "gl_wrapper/shader.hpp"

#include <map>
#include <string>
#include <memory>

class ResourceManager
{
public:
    static ResourceManager& getInstance();
    void loadShader(std::string name, const char* vertex_shader_file,
                    const char* fragment_shader_file,
                    const char* geometry_shader_file = nullptr);
    std::shared_ptr<Shader> getShader(std::string name);
    void unloadShaders();
private:
    ResourceManager();
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    std::map <std::string, std::shared_ptr<Shader> > loaded_shaders;
};

#endif
