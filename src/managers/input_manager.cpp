#include "input_manager.hpp"
#include "constants.hpp"
#include <GLFW/glfw3.h>
#include <iostream>

InputManager& InputManager::getInstance()
{
    static InputManager instance;
    return instance;
}

InputManager::InputManager()
{
    mouse_pos = {0, 0};
    prev_mouse_pos = mouse_pos;
    mouse_pos_change = {0, 0};
    current_screen_size = {0, 0};
    mouse_pos_changed = false;
}

void InputManager::update_events()
{
    // clear old
    keyboard_events.clear();
    mouse_button_events.clear();
    mouse_pos_changed = false;

    keyboard_events.insert(
        keyboard_events_queue.begin(), 
        keyboard_events_queue.end()
    );

    mouse_button_events.insert(
        mouse_button_events_queue.begin(),
        mouse_button_events_queue.end()
    );

    mouse_pos_changed = mouse_pos_changed_queue;

    keyboard_events_queue.clear();
    mouse_button_events_queue.clear();
    mouse_pos_changed_queue = false;
}

void InputManager::addKeyboardEvent(int key, int scancode, int action, int mods)
{
    addKeyboardEvent(KeyboardEvent{key, scancode, action, mods});
}

void InputManager::addKeyboardEvent(KeyboardEvent event)
{
    if (event.action == GLFW_REPEAT)
        return;

    keyboard_events_queue.insert(event);

    if (event.action == GLFW_PRESS)
    {
        pressed_keys.insert(event.key);
    }
    else if (event.action == GLFW_RELEASE)
    {
        pressed_keys.erase(event.key);
    }
}

void InputManager::addMouseButtonEvent(int button, int action, int mods)
{
    addMouseButtonEvent(MouseButtonEvent{button, action, mods});
}

void InputManager::addMouseButtonEvent(MouseButtonEvent event)
{
    if (event.action == GLFW_REPEAT)
        return;
    
    mouse_button_events_queue.insert(event);

    if (event.action == GLFW_PRESS)
    {
        pressed_mouse_buttons.insert(event.button);
    }
    else if (event.action == GLFW_RELEASE)
    {
        pressed_mouse_buttons.erase(event.button);
    }
}

bool InputManager::isKeyDown(int key)
{
    return pressed_keys.count(key);
}

bool InputManager::isKeyUp(int key)
{
    return !pressed_keys.count(key);
}

bool InputManager::isMouseButtonDown(int button)
{
    return pressed_mouse_buttons.count(button);
}

bool InputManager::isMouseButtonUp(int button)
{
    return !pressed_mouse_buttons.count(button);
}

bool InputManager::keyPressed(int key)
{
    auto x = keyboard_events.find({key, 0, 0, 0});

    if (x == keyboard_events.end())
        return 0;
        
    return x->action == GLFW_PRESS;
}

bool InputManager::keyReleased(int key)
{
    auto x = keyboard_events.find({key, 0, 0, 0});

    if (x == keyboard_events.end())
        return 0;
    
    return x->action == GLFW_RELEASE;
}

bool InputManager::mouseButtonPressed(int button)
{
    auto x = mouse_button_events.find({button, 0, 0});

    if (x == mouse_button_events.end())
        return 0;
    
    return x->action == GLFW_PRESS;
}

bool InputManager::mouseButtonReleased(int button)
{
    auto x = mouse_button_events.find({button, 0, 0});

    if (x == mouse_button_events.end())
        return 0;
    
    return x->action == GLFW_RELEASE;
}

void InputManager::mouseMoved(int x, int y)
{
    prev_mouse_pos = mouse_pos;
    mouse_pos = {x, y};
    mouse_pos_changed_queue = true;

    if(LOCK_MOUSE)
    {
        mouse_pos_change = {x - current_screen_size.first / 2, y - current_screen_size.second / 2};
        mouse_pos = {current_screen_size.first / 2, current_screen_size.second / 2};
        prev_mouse_pos = mouse_pos;
    }
}

bool InputManager::isMouseMoved()
{
    return mouse_pos_changed;
}

std::pair <int, int> InputManager::getMousePosChange()
{
    return mouse_pos_change;
}

std::pair <int, int> InputManager::getMousePos()
{
    return mouse_pos;
}

std::pair <int, int> InputManager::getPrevMousePos()
{
    return prev_mouse_pos;
}

void InputManager::update_screen_size(int screen_width, int screen_height)
{
    current_screen_size = {screen_width, screen_height};
}
