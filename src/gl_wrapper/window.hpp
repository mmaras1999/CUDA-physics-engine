#ifndef WINDOW
#define WINDOW

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <GLFW/glfw3.h>

#include <string>
#include <memory>
#include <vector>

#include "../types.hpp"
#include "../constants.hpp"
#include "canvas.hpp"

struct Viewport
{
    int x, y, width, height;

    float getAspectRatio()
    {
        return (float)width / height;
    }
};

void AGLErrors(const char* comment);

class MyWindow 
{
public:
    int width, height;
    int old_width, old_height;
    std::pair <int, int> window_position;
    std::pair <int, int> old_window_position;
    
    float aspect_ratio;

    MyWindow(int width, int height, const char* name, int fullscreen);
    ~MyWindow();
    
    void run();
    virtual void update(double dt, bool log = false);
    virtual void draw();
    void close();

    void setViewport(int x, int y, int width, int height);
    Viewport getViewport() const;

    bool isFullscreen();

protected:
    std::vector <std::shared_ptr<Canvas> > canvases;

    GLFWwindow* glfw_window;
    GLFWmonitor* glfw_monitor;
    bool should_be_closed;
    Viewport current_viewport;

    virtual void resize(int new_width, int new_height);

    void setViewport(Viewport);
    void resetViewport();

    static void callbackError(int error, const char* description);
    static void callbackResize(GLFWwindow* window, int cx, int cy);
    static void callbackKey(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void callbackMouseButton(GLFWwindow* window, int button, int action, int mods);
    static void callbackScroll(GLFWwindow* window, double xp, double yp);
    static void callbackMousePos(GLFWwindow* window, double xp, double yp);

    void setFullscreen(bool fullscreen);
};

#endif
