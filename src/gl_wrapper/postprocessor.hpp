#ifndef POST_PROCESSOR
#define POST_PROCESSOR

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <glm/glm.hpp>

#include "texture.hpp"
#include "shader.hpp"
#include "types.hpp"

class PostProcessor
{
public:
    Texture texture;
    uint width, height;

    PostProcessor(uint width, uint height);
    
    void begin();
    void end();
    void begin_render();
    void end_render();
    void free();
private:

    uint msaa_fbo, fbo;
    uint color_rbo, depth_rbo;
    uint vao, vbo;
    
    void initRenderData();
};

#endif