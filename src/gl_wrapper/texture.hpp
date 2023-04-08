#ifndef TEXTURE
#define TEXTURE

#include <epoxy/gl.h>
#include <epoxy/glx.h>

#include "types.hpp"

class Texture
{
public:
    uint ID;
    uint width, height; 
    
    uint internal_format, image_format;

    uint wrap_s, wrap_t;
    uint filter_min, filter_max;
    uint data_type;
    bool multisample;

    Texture();
    
    void generate(unsigned int width, unsigned int height, unsigned char* data, bool floating);
    void bind() const;
    void unbind() const;
    void free();
};

#endif