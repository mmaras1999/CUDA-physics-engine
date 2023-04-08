#include "resource_manager.hpp"
#include <GLFW/glfw3.h>
#include <iostream>

ResourceManager& ResourceManager::getInstance()
{
    static ResourceManager instance;
    return instance;
}

ResourceManager::ResourceManager() {}

void ResourceManager::loadShader(std::string name, 
                                 const char* vertex_shader_file,
                                 const char* fragment_shader_file, 
                                 const char* geometry_shader_file)
{
    if (loaded_shaders.count(name))
        return;

    loaded_shaders.emplace(name, std::make_shared<Shader>(
        vertex_shader_file, 
        fragment_shader_file, 
        geometry_shader_file)
    );
}

std::shared_ptr<Shader> ResourceManager::getShader(std::string name)
{
    if (!loaded_shaders.count(name))
        throw std::runtime_error("Error: Tried to load shader " + name + "that is not loaded!");
    return loaded_shaders[name];
}

void ResourceManager::unloadShaders()
{
    loaded_shaders.clear();
}
