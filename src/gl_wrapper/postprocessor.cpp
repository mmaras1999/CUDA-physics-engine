#include "postprocessor.hpp"

#include <exception>
#include <string>
#include <stdexcept>

PostProcessor::PostProcessor(uint width, uint height) 
    : texture(), width(width), height(height)
{
    glGenFramebuffers(1, &msaa_fbo);
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &color_rbo);
    glGenRenderbuffers(1, &depth_rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, color_rbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGB16F, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_rbo);
    
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw new std::runtime_error("ERROR: Failed to initialize postprocessing MSAA_FBO!");
   
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    texture.internal_format = GL_RGB16F;
    texture.generate(width, height, NULL, true);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.ID, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw new std::runtime_error("ERROR:: Failed to initialize postprocessing FBO!");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    initRenderData();
}

void PostProcessor::begin()
{
    glEnable(GL_MULTISAMPLE);
    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

void PostProcessor::end()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PostProcessor::begin_render()
{
    glBindVertexArray(vao);
    texture.bind();
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
}

void PostProcessor::end_render()
{
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    texture.unbind();
    glBindVertexArray(0);
}

void PostProcessor::initRenderData()
{
    float vertices[] = {
        // position   // texture position
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,

        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PostProcessor::free()
{
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteRenderbuffers(1, &depth_rbo);
    glDeleteRenderbuffers(1, &color_rbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &msaa_fbo);
}