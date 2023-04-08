#include "texture.hpp"

Texture::Texture()
    : width(0), height(0), 
      internal_format(GL_RGBA), image_format(GL_RGBA), 
      wrap_s(GL_REPEAT), wrap_t(GL_REPEAT), 
      filter_min(GL_LINEAR), filter_max(GL_LINEAR),
      data_type(GL_UNSIGNED_BYTE),
      multisample(false)
{
    glGenTextures(1, &ID);
}

void Texture::generate(unsigned int width, unsigned int height, unsigned char* data, bool floating)
{
    this->width = width;
    this->height = height;

    // create Texture
    if (!floating)
    {
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, image_format, data_type, data);
        // set Texture wrap and filter modes
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_min);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_max);
        // unbind texture
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, image_format, GL_FLOAT, data);
        // set Texture wrap and filter modes
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_min);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_max);
        // unbind texture
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void Texture::bind() const
{
    glBindTexture(GL_TEXTURE_2D, ID);
}

void Texture::unbind() const
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::free()
{
    glDeleteTextures(1, &ID);
}