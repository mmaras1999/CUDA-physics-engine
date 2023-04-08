#include "rigid_body_app.hpp"
#include "managers/input_manager.hpp"
#include "simulation/object_voxelizer.hpp"
#include <iostream>

RigidBodyApp::RigidBodyApp(int width, int height, const char* name, int fullscreen, const std::string& object_path, const std::string& start_config_path)
    : MyWindow(width, height, name, fullscreen), 
      simulation_state(nullptr)
{
    simulation_state = std::make_shared<RigidBodySimulationState>(width, height, object_path, start_config_path);
    canvases.push_back(simulation_state);
}

void RigidBodyApp::resize(int new_width, int new_height)
{
    MyWindow::resize(new_width, new_height);
}

void RigidBodyApp::update(double dt, bool log)
{
    simulation_state->update(dt);

    previous_updates.push_back(dt);

    if (previous_updates.size() > FPS_COUNTER_UPDATES)
    {
        previous_updates.pop_front();
    }

    if (log)
    {
        double UPS = 0.0;

        for (const auto& d : previous_updates)
            UPS += d;

        UPS /= previous_updates.size();

        std::cout << "Updates per second: " << 1.0 / UPS << std::endl;
    }

    auto& input_manager = InputManager::getInstance();

    if (input_manager.isKeyDown(GLFW_KEY_ESCAPE))
    {
        close();
    }

    if (input_manager.keyPressed(GLFW_KEY_F))
    {
        setFullscreen(!isFullscreen());
    }

    if (int end = simulation_state->simEnded(); end)
    {
        std::cout << "Simulation took " << simulation_state->getTime() << " seconds" << std::endl;
        close();
    }

    MyWindow::update(dt, log);
}

void RigidBodyApp::draw()
{
    if (!DISABLE_DRAWING)
    {
        MyWindow::draw();
        simulation_state->draw();
    }
}
