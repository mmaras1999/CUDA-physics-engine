#ifndef INPUT_MANAGER
#define INPUT_MANAGER

#include <set>

struct KeyboardEvent
{
    int key, scancode, action, mods;

    bool operator< (const KeyboardEvent& rhs) const 
    {
       return key < rhs.key;
    }
};

struct MouseButtonEvent
{
    int button, action, mods;

    bool operator< (const MouseButtonEvent& rhs) const
    {
        return button < rhs.button;
    }
};

class InputManager
{
    public:
        static InputManager& getInstance();
        
        void addKeyboardEvent(int key, int scancode, int action, int mods);
        void addKeyboardEvent(KeyboardEvent event);

        void addMouseButtonEvent(int button, int action, int mods);
        void addMouseButtonEvent(MouseButtonEvent event);

        void mouseMoved(int x, int y);

        void update_events();

        bool isKeyDown(int key);
        bool isKeyUp(int key);

        bool isMouseButtonDown(int button);
        bool isMouseButtonUp(int button);

        bool keyPressed(int key);
        bool keyReleased(int key);

        bool mouseButtonPressed(int button);
        bool mouseButtonReleased(int button);

        bool isMouseMoved();
        std::pair <int, int> getMousePos();
        std::pair <int, int> getPrevMousePos();
        std::pair <int, int> getMousePosChange();

        void update_screen_size(int screen_width, int screen_height);

    private:
        std::set <KeyboardEvent> keyboard_events;
        std::set <KeyboardEvent> keyboard_events_queue;
        std::set <MouseButtonEvent> mouse_button_events;
        std::set <MouseButtonEvent> mouse_button_events_queue;
        std::set <int> pressed_keys;
        std::set <int> pressed_mouse_buttons;

        std::pair <int, int> current_screen_size;
        std::pair <int, int> prev_mouse_pos;
        std::pair <int, int> mouse_pos;
        std::pair <int, int> mouse_pos_change;
        bool mouse_pos_changed = false;
        bool mouse_pos_changed_queue = false;
        bool mouse_disabled = false;

        InputManager();
        InputManager(const InputManager&) = delete;
        InputManager& operator=(const InputManager&) = delete;
};

#endif
