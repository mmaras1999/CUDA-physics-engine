#ifndef OBJECT_VIEWER_APP
#define OBJECT_VIEWER_APP

#include "gl_wrapper/window.hpp"
#include "simulation/object_viewer.hpp"

#include <string>

class ObjectViewerApp : public MyWindow
{
public:
    ObjectViewerApp(int width, int height, const char* name, int fullscreen, std::string path);
    void update(double dt, bool log = false) override;
    void draw() override;
protected:
    std::shared_ptr<ObjectViewer> obj_view;

    void resize(int new_width, int new_height) override;
};

#endif
