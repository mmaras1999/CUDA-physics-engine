#include "canvas.hpp"

Canvas::Canvas(int screen_width, int screen_height)
{
    offset = {0, 0};
    size = {screen_width, screen_height};
}

void Canvas::draw() {}
void Canvas::update(double dt) {}
void Canvas::resize(int old_width, int old_height,
                    int new_width, int new_height)
{
    size = {new_width, new_height};
}

std::pair <int, int> Canvas::getOffset()
{
    return offset;
}

std::pair <int, int> Canvas::getSize()
{
    return size;
}

void Canvas::freeBuffers() {};
