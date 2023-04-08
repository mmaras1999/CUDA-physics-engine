#ifndef SHADER
#define SHADER

#include <string>

class Shader
{
public:
    unsigned int ID = 0;

    Shader();
    Shader(const char* vertex_shader_file, const char* fragment_shader_name,
           const char* geometry_shader_file = nullptr);
    ~Shader();
    void use() const;
private:
    std::string loadFromFile(const char* file_name);
    void checkCompileErrors(unsigned int object, std::string type);
};

#endif
