// ====================================================
// Code heavily based on:
// AGL3 Ver.3  14.I.2020 (c) A. Łukaszewski
// ====================================================
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "window.hpp"
#include "managers/input_manager.hpp"
#include "managers/resource_manager.hpp"

void AGLErrors(const char *comment) 
{
   GLenum er;
   while (er=glGetError())
      fprintf(stderr, "\nOpenGL ERROR: 0x%04x    =============%s===\n", er, comment);
}

//* ======= Constructor and destructor ======= *//
MyWindow::MyWindow(int width, int height, const char* name, int fullscreen) 
    : width(width), height(height), window_position({0, 0}), aspect_ratio((float)width / height), 
      should_be_closed(false), current_viewport({0, 0, width, height})
{
    // initialize GLFW
    if (!glfwInit())
    {
        glfwTerminate();
        throw std::runtime_error("Error: Can't initialize GLFW!");
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GL_FALSE);

    if (GL_VER > 32)
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_VER / 10);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_VER % 10);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    }

    // setup monitors
    int monitor_cnt;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_cnt);

    if (!monitor_cnt)
        throw std::runtime_error("Error: No monitors found!");
    
    glfw_monitor = monitors[0];

    // create window
    if (fullscreen)
    {
        const GLFWvidmode* mode = glfwGetVideoMode(glfw_monitor);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

        glfw_window = glfwCreateWindow(mode->width, mode->height, name, glfw_monitor, NULL);
    }
    else
    {
        glfw_window = glfwCreateWindow(width, height, name, nullptr, nullptr);
    }

    if (!glfw_window)
    {
        glfwTerminate();
        throw std::runtime_error("Error: Can't create window!");
    }

    glfwMakeContextCurrent(glfw_window);
    glfwGetWindowSize(glfw_window, &(this->width), &(this->height));
    glfwGetWindowPos(glfw_window, &window_position.first,  &window_position.second);
    InputManager::getInstance().update_screen_size(this->width, this->height);

    if (LOCK_MOUSE)
    {
        glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        glfwSetCursorPos(glfw_window, width / 2, height / 2);
    }

    // setup callbacks
    glfwSetWindowUserPointer(glfw_window, this);
    glfwSetErrorCallback(MyWindow::callbackError);
    glfwSetKeyCallback(glfw_window, MyWindow::callbackKey);
    glfwSetScrollCallback(glfw_window, MyWindow::callbackScroll);
    glfwSetWindowSizeCallback(glfw_window, MyWindow::callbackResize);
    glfwSetCursorPosCallback(glfw_window, MyWindow::callbackMousePos);
    glfwSetMouseButtonCallback(glfw_window, MyWindow::callbackMouseButton);

    glfwSwapInterval(1);
    glEnable(GL_MULTISAMPLE); 
}

MyWindow::~MyWindow() 
{
    for (auto& c : canvases)
    {
        c->freeBuffers();
    }

    ResourceManager::getInstance().unloadShaders();
    glfwDestroyWindow(glfw_window);
    glfwTerminate();
}

//* ======= Main loop ======= *//

void MyWindow::run()
{
    resetViewport();
    GLuint vertexArrayID;
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
    glEnable(GL_DEPTH_TEST);

    if (ENABLE_TRANSPARENCY)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    
    double previous_time = glfwGetTime();
    double previous_drawing_time = glfwGetTime();

    while (!glfwWindowShouldClose(glfw_window) and
           !should_be_closed)     
    {
        double current_time = glfwGetTime();
        double delta_time = current_time - previous_time;

        if (current_time - previous_drawing_time >= REFRESH_RATE) 
        {
            update(delta_time, true);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            previous_drawing_time = current_time;
            
            AGLErrors("starting drawing");
            draw();
            AGLErrors("ending drawing");

            glfwSwapBuffers(glfw_window);
            previous_time = current_time;
            InputManager::getInstance().update_events();
        }
        else if (delta_time >= MIN_ENGINE_UPDATE)
        {
            update(delta_time, false);
            previous_time = current_time;
            // InputManager::getInstance().update_events();
        }
        
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &vertexArrayID);
}

void MyWindow::update(double dt, bool log)
{
    for (auto& canvas : canvases)
    {
        canvas->update(dt);
    }

    if (LOCK_MOUSE)
        glfwSetCursorPos(glfw_window, width / 2, height / 2);
}

void MyWindow::draw()
{
    for (auto& canvas : canvases)
    {
        auto offset = canvas->getOffset();
        auto size = canvas->getSize();
        setViewport(offset.first, offset.second, size.first, size.second);
        canvas->draw();
        resetViewport();
    }
}

//* ======= Callbacks ======= *//

void MyWindow::callbackError(int error, const char* description) 
{
    std::cerr << "GLFW error: " << description << std::endl;
}

void MyWindow::callbackResize(GLFWwindow* window, int new_w, int new_h) 
{
    MyWindow *ptr = static_cast<MyWindow*>(glfwGetWindowUserPointer(window));
    if (ptr)
        ptr->resize(new_w, new_h);
}

void MyWindow::callbackKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    InputManager::getInstance().addKeyboardEvent(key, scancode, action, mods);
}

void MyWindow::callbackMouseButton(GLFWwindow* window, int button, int action, int mods) 
{
    InputManager::getInstance().addMouseButtonEvent(button, action, mods);
}

void MyWindow::callbackScroll(GLFWwindow* window, double xp, double yp) 
{
    // scrolled
}

void MyWindow::callbackMousePos(GLFWwindow* window, double xp, double yp)
{
    InputManager::getInstance().mouseMoved(xp, yp);
}

//* ======= Viewports ======= *//

void MyWindow::setViewport(int x, int y, int width, int height)
{
    current_viewport = {x, y, width, height};
    glViewport(x, y, width, height);
}

void MyWindow::setViewport(Viewport viewport)
{
    current_viewport = viewport;
    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
}

Viewport MyWindow::getViewport() const
{
    return current_viewport;
}

void MyWindow::resetViewport()
{
    current_viewport = {0, 0, width, height};
    glViewport(0, 0, width, height);
    InputManager::getInstance().update_screen_size(width, height);
}

//* ======= Others ======= *//

void MyWindow::resize(int new_width, int new_height)
{
    for (auto& canvas : canvases)
    {
        canvas->resize(width, height, new_width, new_height);
    }

    width = new_width;
    height = new_height;

    resetViewport();
}

void MyWindow::close()
{
    should_be_closed = true;
}

bool MyWindow::isFullscreen() 
{
   return glfwGetWindowMonitor(glfw_window) != nullptr;
}

void MyWindow::setFullscreen(bool fullscreen) 
{
    if (isFullscreen() != fullscreen)
    {
        if (fullscreen) 
        {
            glfwGetWindowPos(glfw_window, &old_window_position.first, &old_window_position.second);
            glfwGetWindowSize(glfw_window, &old_width, &old_height);

            const GLFWvidmode* vidmode = glfwGetVideoMode(glfw_monitor);
            glfwSetWindowMonitor(glfw_window, glfw_monitor, 0, 0, vidmode->width, vidmode->height, FPS);
        } 
        else 
        {
            glfwSetWindowMonitor(glfw_window, nullptr, old_window_position.first, old_window_position.second, 
                                 old_width, old_height, 0);
            glfwShowWindow(glfw_window);
        }

        glfwGetWindowPos(glfw_window, &window_position.first, &window_position.second);
        resetViewport();
    }
}

