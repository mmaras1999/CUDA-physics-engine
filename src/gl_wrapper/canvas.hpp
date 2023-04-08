#ifndef CANVAS
#define CANVAS

#include <iostream>

class Canvas
{
public:
    Canvas(int screen_width, int screen_height);
    virtual void update(double dt);
    virtual void resize(int old_width, int old_height, 
                        int new_width, int new_height);
    virtual void draw();
    virtual void freeBuffers();
    
    std::pair <int, int> getOffset();
    std::pair <int, int> getSize();
    
protected:
    std::pair <int, int> offset;
    std::pair <int, int> size; 
};

#endif
