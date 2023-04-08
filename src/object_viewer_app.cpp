#include "object_viewer_app.hpp"
#include "managers/input_manager.hpp"
#include "simulation/object_voxelizer.hpp"

#include <iostream>
#include <string>

ObjectViewerApp::ObjectViewerApp(int width, int height, const char* name, int fullscreen, std::string path)
    : MyWindow(width, height, name, fullscreen), 
      obj_view(nullptr)
{
    auto voxelizer = ObjectVoxelizer(width, height, path);
    auto particles = voxelizer.generateVoxels(glfw_window);

    std::cout << "Particles: " << particles.size() << std::endl;
    std::cout << "Example particle: " << particles[7].x << " " << particles[7].y << " " << particles[7].z << std::endl;

    obj_view = std::make_shared<ObjectViewer>(width, height, path, particles);
    canvases.push_back(obj_view);
}

void ObjectViewerApp::resize(int new_width, int new_height)
{
    MyWindow::resize(new_width, new_height);
}

void ObjectViewerApp::update(double dt, bool log)
{
    auto& input_manager = InputManager::getInstance();

    if (input_manager.isKeyDown(GLFW_KEY_ESCAPE))
    {
        close();
    }

    if (input_manager.keyPressed(GLFW_KEY_F))
    {
        setFullscreen(!isFullscreen());
    }

    MyWindow::update(dt);
}

void ObjectViewerApp::draw()
{
    if (!DISABLE_DRAWING)
    {
        MyWindow::draw();
        obj_view->draw();
    }
}
