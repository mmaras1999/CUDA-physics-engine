#ifndef CONSTANTS
#define CONSTANTS

#include <string>

const int GL_VER = 33;
const int FPS = 60;
const float REFRESH_RATE = 1.0f / FPS;
const bool LOCK_MOUSE = false;
const std::string SHADER_DIR = "resources/shaders/";
const bool ENABLE_TRANSPARENCY = false;
const int FPS_COUNTER_UPDATES = 10;
const bool DISABLE_DRAWING = false;
const bool ALLOW_RESIZE = true;
const bool DRAW_PARTICLES = true;
const float MIN_ENGINE_UPDATE = 1; // synchronize with graphics

// CONSTANTS FOR INITIALIZATION
const int MAX_DEPTH_PEELING = 20;

// CONSTANTS FOR SIMULATION
#define METER 1.0f
#define GRAVITY 9.8f
const int PARTICLE_AMOUNT = 8000;
#define PARTICLE_RADIUS 0.00625f
#define PARTICLE_RADIUS_SIM 0.00624f
#define PARTICLE_MASS 0.05f
#define TIME_SPEED 0.5f
#define OBJECT_SCALE 0.3f
#define MAX_FORCE 100.0f

#define SPRING_COEFF 700.0f
#define DAMPING_COEFF 10.0f
#define FRICTION_COEFF 0.1f

#define BORDER_SPRING_COEFF 2050.0f
#define BORDER_DAMPING_COEFF 50.0f
#define BORDER_FRICTION_COEFF 0.5f

// set collisions type
#define UNIFORM_GRID
// #define BRUTE_FORCE

#ifdef UNIFORM_GRID
#define GRID_SIDE_LEN 0.0125f
#define GRID_MAX_X 3.0f
#define GRID_MAX_Y 3.0f
#define GRID_MAX_Z 3.0f
// particles outside of those buckets will be ignored
constexpr int MAX_BUCKETS = (GRID_MAX_X / GRID_SIDE_LEN) *  // x
                            (GRID_MAX_Y / GRID_SIDE_LEN) *  // y
                            (GRID_MAX_Z / GRID_SIDE_LEN);   // z
#endif

#endif
