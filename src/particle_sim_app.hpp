#ifndef APP
#define APP

#include "gl_wrapper/window.hpp"
#include "simulation/particle_simulation_state.hpp"

#include <deque>

class ParticleSimApp : public MyWindow
{
public:
    ParticleSimApp(int width, int height, const char* name, int fullscreen, const std::string& object_path, const std::string& start_config_path);
    void update(double dt, bool log = false) override;
    void draw() override;
protected:
    std::shared_ptr<ParticleSimulationState> simulation_state;
    std::deque <double> previous_updates;

    void resize(int new_width, int new_height) override;
};

#endif
