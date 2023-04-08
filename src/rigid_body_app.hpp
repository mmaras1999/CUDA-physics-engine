#ifndef RIGID_BODY_APP
#define RIGID_BODY_APP

#include "gl_wrapper/window.hpp"
#include "simulation/rigid_body_simulation_state.hpp"

#include <deque>

class RigidBodyApp : public MyWindow
{
public:
    RigidBodyApp(int width, int height, const char* name, int fullscreen, const std::string& object_path, const std::string& start_config_path);
    void update(double dt, bool log = false) override;
    void draw() override;
protected:
    std::shared_ptr<RigidBodySimulationState> simulation_state;
    std::deque <double> previous_updates;

    void resize(int new_width, int new_height) override;
};

#endif
