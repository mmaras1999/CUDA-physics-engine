#include "shader.hpp"
#include "constants.hpp"

#include <epoxy/gl.h>
#include <epoxy/glx.h>

#include <stdexcept>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>

Shader::Shader()
{
    ID = 0;
}

Shader::Shader(const char* vertex_shader_file, const char* fragment_shader_file,
               const char* geometry_shader_file)
{
    std::string vertex_shader_data;
    std::string fragment_shader_data;
    std::string geometry_shader_data;

    try
    {
        vertex_shader_data = loadFromFile(vertex_shader_file);
        fragment_shader_data = loadFromFile(fragment_shader_file);
        if (geometry_shader_file)
            geometry_shader_data = loadFromFile(geometry_shader_file);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error while reading shader files: " << std::endl;
        std::cerr << e.what() << std::endl;
    }

    const char* temp = vertex_shader_data.c_str();
    GLuint v_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v_shader, 1, &temp, NULL);
    glCompileShader(v_shader);
    checkCompileErrors(v_shader, "VERTEX");

    temp = fragment_shader_data.c_str();
    GLuint f_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f_shader, 1, &temp, NULL);
    glCompileShader(f_shader);
    checkCompileErrors(f_shader, "FRAGMENT");

    GLuint g_shader = 0;
    
    if (geometry_shader_file)
    {
        temp = geometry_shader_data.c_str();
        g_shader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(g_shader, 1, &temp, NULL);
        glCompileShader(g_shader);
        checkCompileErrors(g_shader, "GEOMETRY");
    }
    
    this->ID = glCreateProgram();
    glAttachShader(this->ID, v_shader);
    glAttachShader(this->ID, f_shader);
    if (g_shader)
        glAttachShader(this->ID, g_shader);

    glLinkProgram(this->ID);
    checkCompileErrors(this->ID, "PROGRAM");
    
    glDeleteShader(v_shader);
    glDeleteShader(f_shader);
    if (g_shader)
        glDeleteShader(g_shader);
}

std::string Shader::loadFromFile(const char* file_name)
{
    std::string file_path = SHADER_DIR + std::string(file_name);
    std::ifstream file_stream(file_path);
    std::stringstream ss;

    if (file_stream.is_open())
    {
        ss << file_stream.rdbuf();
        file_stream.close();
        return ss.str();
    }
    else
    {
        throw std::runtime_error("Error: Shader file " + file_path + " not found!");
    }
}

void Shader::checkCompileErrors(unsigned int object, std::string type)
{
    int success;
    char infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(object, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(object, 1024, NULL, infoLog);
            std::cout << "| ERROR::SHADER: Compile-time error: Type: " << type << "\n"
                << infoLog << "\n -- --------------------------------------------------- -- "
                << std::endl;
        }
    }
    else
    {
        glGetProgramiv(object, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(object, 1024, NULL, infoLog);
            std::cout << "| ERROR::Shader: Link-time error: Type: " << type << "\n"
                << infoLog << "\n -- --------------------------------------------------- -- "
                << std::endl;
        }
    }
}

void Shader::use() const
{
    glUseProgram(ID);
}

Shader::~Shader()
{
    if (ID) glDeleteProgram(ID);
}
