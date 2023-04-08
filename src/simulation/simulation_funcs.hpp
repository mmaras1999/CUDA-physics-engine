#ifndef SIMULATION_FUNCS
#define SIMULATION_FUNCS

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <GLFW/glfw3.h>
#include "constants.hpp"

#include <cuda_gl_interop.h>
#include <thrust/sort.h>

__global__ void update_physics(
    float* positions, 
    float* momenta,
    float* forces, 
    int particle_num, 
    float dt
);

__global__ void update_physics_with_rotation(
    float* object_positions,
    float* object_quat,
    float* object_momenta,
    float* object_angular_momenta,
    float* inertia_tensor,
    int object_amount,
    int particles_amount,
    float dt
);

__global__ void calculate_particle_values(
    float* object_positions,
    float* object_quats,
    float* object_momenta,
    float* object_angular_momenta,
    float* particle_rel_pos,
    float* particle_positions,
    float* particle_momenta,
    float* inertia_tensor,
    int object_amount,
    int particle_amount
);

__global__ void update_object_momenta(
    float* forces,
    float* particle_rel_pos,
    float* object_quats,
    float* object_momenta,
    float* object_angular_momenta,
    int object_amount,
    int particle_amount,
    float dt
);

__forceinline__ __device__ void check_border_collisions(
    float* positions,
    float* momenta,
    float* forces, 
    int id,
    float dt
);

__forceinline__ __device__ void check_particle_collisions(
    float* positions,
    float* momenta,
    float* forces,
    int id1,
    int id2,
    float dt
);

#ifdef BRUTE_FORCE
__global__ void update_forces_brute_force(
    float* positions,
    float* momenta,
    float* forces, 
    int particle_num,
    float dt
);
#endif

#ifdef UNIFORM_GRID
__global__ void calc_grid_hashes(
    float* positions,
    uint* grid_hashes_values,
    uint* grid_hashes_keys,
    int particle_num,
    int max_buckets
);

__global__ void calc_bucket_ranges(
    uint* grid_hashes_keys,
    int* bucket_begin,
    int* bucket_end,
    int particle_num,
    int max_buckets
);

__global__ void calc_grid_collisions(
    float* positions,
    float* momenta,
    float* forces,
    uint* grid_hashes_values,
    uint* grid_hashes_keys,
    int* bucket_begin,
    int* bucket_end,
    int particle_num,
    int max_buckets,
    float dt
);
#endif

#endif
