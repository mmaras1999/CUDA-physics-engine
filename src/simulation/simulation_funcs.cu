#include "simulation_funcs.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

// PARTICLES ONLY
__global__ void update_physics(
    float* positions, 
    float* momenta,
    float* forces, 
    int particle_num, 
    float dt)
{
    uint id  = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < particle_num)
    {
        // update momenta
        momenta[3 * id + 0] += forces[3 * id + 0] * dt;
        momenta[3 * id + 1] += forces[3 * id + 1] * dt;
        momenta[3 * id + 2] += forces[3 * id + 2] * dt;

        // update positions
        positions[3 * id + 0] += momenta[3 * id + 0] * dt / PARTICLE_MASS * METER;
        positions[3 * id + 1] += momenta[3 * id + 1] * dt / PARTICLE_MASS * METER;
        positions[3 * id + 2] += momenta[3 * id + 2] * dt / PARTICLE_MASS * METER;
    }
}

// OBJECT PHYSICS
__global__ void update_physics_with_rotation(
    float* object_positions,
    float* object_quats,
    float* object_momenta,
    float* object_angular_momenta,
    float* inertia_tensor,
    int object_amount,
    int particles_amount,
    float dt)
{
    uint id  = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < object_amount)
    {
        object_positions[3 * id + 0] += object_momenta[3 * id + 0] / (PARTICLE_MASS * particles_amount) * dt * METER;
        object_positions[3 * id + 1] += object_momenta[3 * id + 1] / (PARTICLE_MASS * particles_amount) * dt * METER;
        object_positions[3 * id + 2] += object_momenta[3 * id + 2] / (PARTICLE_MASS * particles_amount) * dt * METER;

        glm::mat3 in_ten = glm::mat3(inertia_tensor[0], inertia_tensor[1], inertia_tensor[2], 
                                     inertia_tensor[3], inertia_tensor[4], inertia_tensor[5],
                                     inertia_tensor[6], inertia_tensor[7], inertia_tensor[8]);
        glm::vec3 object_angular_momentum = glm::vec3(object_angular_momenta[3 * id + 0],
                                                      object_angular_momenta[3 * id + 1],
                                                      object_angular_momenta[3 * id + 2]);
        glm::quat object_quat = glm::normalize(glm::quat(object_quats[4 * id + 3],
                                                         object_quats[4 * id + 0],
                                                         object_quats[4 * id + 1],
                                                         object_quats[4 * id + 2]));

        auto rot_mat = glm::toMat3(object_quat);
        in_ten = rot_mat * in_ten * glm::transpose(rot_mat);

        auto angular_velocity = in_ten * object_angular_momentum;
        auto angle = glm::length(angular_velocity * dt * 0.5f);
        glm::vec3 a = glm::normalize(angular_velocity * dt * 0.5f);

        if (angle >= 0.0001f)
        {
            object_quat = glm::normalize(glm::normalize(glm::quat(cos(angle), a * sin(angle))) * object_quat);
            
            object_quats[4 * id + 0] = object_quat.x;
            object_quats[4 * id + 1] = object_quat.y;
            object_quats[4 * id + 2] = object_quat.z;
            object_quats[4 * id + 3] = object_quat.w;
        }

        object_angular_momenta[3 * id + 0] *= 0.97f;
        object_angular_momenta[3 * id + 1] *= 0.97f;
        object_angular_momenta[3 * id + 2] *= 0.97f;
    }
}

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
    int particle_amount)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < particle_amount * object_amount)
    {
        uint particle_id = id % particle_amount;
        uint object_id = id / particle_amount;

        glm::vec3 object_pos = glm::vec3(object_positions[3 * object_id + 0],
                                         object_positions[3 * object_id + 1],
                                         object_positions[3 * object_id + 2]);

        glm::quat object_quat = glm::normalize(glm::quat(object_quats[4 * object_id + 3],
                                                         object_quats[4 * object_id + 0],
                                                         object_quats[4 * object_id + 1],
                                                         object_quats[4 * object_id + 2]));

        glm::vec3 object_momentum = glm::vec3(object_momenta[3 * object_id + 0],
                                              object_momenta[3 * object_id + 1],
                                              object_momenta[3 * object_id + 2]);

        glm::vec3 object_angular_momentum = glm::vec3(object_angular_momenta[3 * object_id + 0],
                                                      object_angular_momenta[3 * object_id + 1],
                                                      object_angular_momenta[3 * object_id + 2]);

        glm::vec3 pos = glm::vec3(particle_rel_pos[3 * particle_id + 0],
                                  particle_rel_pos[3 * particle_id + 1],
                                  particle_rel_pos[3 * particle_id + 2]);

        glm::mat3 in_t = glm::mat3(inertia_tensor[0], inertia_tensor[1], inertia_tensor[2], 
                                   inertia_tensor[3], inertia_tensor[4], inertia_tensor[5],
                                   inertia_tensor[6], inertia_tensor[7], inertia_tensor[8]);

        auto rot_mat = glm::toMat3(glm::normalize(object_quat));
        pos = object_quat * pos;
        auto particle_pos = object_pos + pos;

        in_t = rot_mat * in_t * glm::transpose(rot_mat);
        auto angular_velocity = in_t * object_angular_momentum;

        angular_velocity = glm::cross(angular_velocity, pos);

        particle_positions[3 * id + 0] = particle_pos.x;
        particle_positions[3 * id + 1] = particle_pos.y;
        particle_positions[3 * id + 2] = particle_pos.z;

        particle_momenta[3 * id + 0] = object_momentum.x / particle_amount + angular_velocity.x * PARTICLE_MASS;
        particle_momenta[3 * id + 1] = object_momentum.y / particle_amount + angular_velocity.y * PARTICLE_MASS;
        particle_momenta[3 * id + 2] = object_momentum.z / particle_amount + angular_velocity.z * PARTICLE_MASS;
    }
}

__global__ void update_object_momenta(
    float* forces,
    float* particle_rel_pos,
    float* object_quats,
    float* object_momenta,
    float* object_angular_momenta,
    int object_amount,
    int particle_amount,
    float dt)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < object_amount * particle_amount)
    {
        uint particle_id = id % particle_amount;
        uint object_id = id / particle_amount;

        atomicAdd(&object_momenta[3 * object_id + 0], forces[3 * id + 0] * dt);
        atomicAdd(&object_momenta[3 * object_id + 1], forces[3 * id + 1] * dt);
        atomicAdd(&object_momenta[3 * object_id + 2], forces[3 * id + 2] * dt);
        
        glm::vec3 pos = glm::vec3(particle_rel_pos[3 * particle_id + 0],
                                  particle_rel_pos[3 * particle_id + 1],
                                  particle_rel_pos[3 * particle_id + 2]);

        glm::quat object_quat = glm::normalize(glm::quat(object_quats[4 * object_id + 3],
                                                         object_quats[4 * object_id + 0],
                                                         object_quats[4 * object_id + 1],
                                                         object_quats[4 * object_id + 2]));
        pos = object_quat * pos;

        glm::vec3 f = glm::vec3(forces[3 * id + 0],
                                forces[3 * id + 1],
                                forces[3 * id + 2]);
        
        auto torque = glm::cross(pos, f);

        atomicAdd(&object_angular_momenta[3 * object_id + 0], torque.x * dt);
        atomicAdd(&object_angular_momenta[3 * object_id + 1], torque.y * dt);
        atomicAdd(&object_angular_momenta[3 * object_id + 2], torque.z * dt);
    }
}

// PARTICLE COLLISIONS HANDLING
__forceinline__ __device__ void check_border_collisions(
    float* positions,
    float* momenta,
    float* forces, 
    int id,
    float dt)
{
    float velocities[3] = {
        momenta[3 * id + 0] / PARTICLE_MASS,
        momenta[3 * id + 1] / PARTICLE_MASS,
        momenta[3 * id + 2] / PARTICLE_MASS
    };

    float new_pos[3] = {
        positions[3 * id + 0], //+ velocities[0] * dt,
        positions[3 * id + 1], // + velocities[1] * dt,
        positions[3 * id + 2], // + velocities[2] * dt
    };

    // collision with borders
    if (new_pos[1] < PARTICLE_RADIUS_SIM)
    {   
        // ignore gravity
        forces[3 * id + 1] += GRAVITY * PARTICLE_MASS;

        // repulsive force
        forces[3 * id + 1] += BORDER_SPRING_COEFF * (PARTICLE_RADIUS_SIM - new_pos[1]) / METER;
        // damping force
        forces[3 * id + 1] += -BORDER_DAMPING_COEFF * velocities[1];
        // tangential force (friction)
        forces[3 * id + 0] += -BORDER_FRICTION_COEFF * velocities[0];
        forces[3 * id + 2] += -BORDER_FRICTION_COEFF * velocities[2];
    }

    if (new_pos[0] < PARTICLE_RADIUS_SIM)
    {   
        // repulsive force
        forces[3 * id + 0] += BORDER_SPRING_COEFF * (PARTICLE_RADIUS_SIM - new_pos[0]) / METER;
        // damping force
        forces[3 * id + 0] += -BORDER_DAMPING_COEFF * velocities[0];
        // tangential force (friction)
        forces[3 * id + 1] += -BORDER_FRICTION_COEFF * velocities[1];
        forces[3 * id + 2] += -BORDER_FRICTION_COEFF * velocities[2];
    }

    if (new_pos[2] < PARTICLE_RADIUS_SIM)
    {   
        // repulsive force
        forces[3 * id + 2] += BORDER_SPRING_COEFF * (PARTICLE_RADIUS_SIM - new_pos[2]) / METER;
        // damping force
        forces[3 * id + 2] += -BORDER_DAMPING_COEFF * velocities[2];
        // tangential force (friction)
        forces[3 * id + 0] += -BORDER_FRICTION_COEFF * velocities[0];
        forces[3 * id + 1] += -BORDER_FRICTION_COEFF * velocities[1];
    }
}

__forceinline__ __device__ void check_particle_collisions(
    float* positions,
    float* momenta,
    float* forces,
    int id1,
    int id2,
    float dt)
{
    float velocities[3] = {
        momenta[3 * id1 + 0] / PARTICLE_MASS,
        momenta[3 * id1 + 1] / PARTICLE_MASS,
        momenta[3 * id1 + 2] / PARTICLE_MASS
    };

    if (id1 != id2)
    {
        float d = 0.0f;
        float pos_rel[3] = {
            positions[3 * id2 + 0] - positions[3 * id1 + 0],
            positions[3 * id2 + 1] - positions[3 * id1 + 1],
            positions[3 * id2 + 2] - positions[3 * id1 + 2]
        };

        for (int k = 0; k < 3; ++k)
        {
            d += pos_rel[k] * pos_rel[k];
        }

        d = sqrt(d);

        if (d < 2 * PARTICLE_RADIUS_SIM and d > 0.000001f)
        {
            float vel_rel[3] = {
                momenta[3 * id2 + 0] / PARTICLE_MASS - velocities[0],
                momenta[3 * id2 + 1] / PARTICLE_MASS - velocities[1],
                momenta[3 * id2 + 2] / PARTICLE_MASS - velocities[2]
            };

            // repulsive and damping force
            for (int k = 0; k < 3; ++k)
            {
                forces[3 * id1 + k] += -SPRING_COEFF * (2 * PARTICLE_RADIUS_SIM - d) * pos_rel[k] / d / METER;
                forces[3 * id1 + k] += DAMPING_COEFF * vel_rel[k];
            }

            float dot_p = 0.0f;

            for (int k = 0; k < 3; ++k)
                dot_p += vel_rel[k] * pos_rel[k] / d;

            // friction force
            for (int k = 0; k < 3; ++k)
            {
                forces[3 * id1 + k] += FRICTION_COEFF * (vel_rel[k] - dot_p * pos_rel[k] / d);
            }
        }
    }
}

#ifdef BRUTE_FORCE
__global__ void update_forces_brute_force(
    float* positions, 
    float* momenta,
    float* forces, 
    int particle_num, 
    float dt)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < particle_num)
    {
        forces[3 * id + 0] = 0.0f;
        forces[3 * id + 1] = -GRAVITY * PARTICLE_MASS;
        forces[3 * id + 2] = 0.0f;

        for (int j = 0; j < particle_num; ++j)
        {
            check_particle_collisions(positions, momenta, forces, id, j, dt);
        }

        check_border_collisions(positions, momenta, forces, id, dt);


        forces[3 * id + 0] = max(forces[3 * id + 0], -MAX_FORCE);
        forces[3 * id + 0] = min(forces[3 * id + 0], MAX_FORCE);
        forces[3 * id + 1] = max(forces[3 * id + 1], -MAX_FORCE);
        forces[3 * id + 1] = min(forces[3 * id + 1], MAX_FORCE);
        forces[3 * id + 2] = max(forces[3 * id + 2], -MAX_FORCE);
        forces[3 * id + 2] = min(forces[3 * id + 2], MAX_FORCE);
    }
}
#endif 

#ifdef UNIFORM_GRID
__global__ void calc_grid_hashes(
    float* positions,
    uint* grid_hashes_values,
    uint* grid_hashes_keys,
    int particle_num,
    int max_buckets)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < particle_num)
    {
        int x = floor(positions[3 * id + 0] / GRID_SIDE_LEN);
        int y = floor(positions[3 * id + 1] / GRID_SIDE_LEN);
        int z = floor(positions[3 * id + 2] / GRID_SIDE_LEN);

        int max_x = round(GRID_MAX_X / GRID_SIDE_LEN);
        int max_y = round(GRID_MAX_Y / GRID_SIDE_LEN);
        int max_z = round(GRID_MAX_Z / GRID_SIDE_LEN);

        grid_hashes_values[id] = id;

        if (x < max_x and y < max_y and z < max_z)
        {
            grid_hashes_keys[id] = x + y * max_x + z * max_x * max_y;
        }
        else
        {
            grid_hashes_keys[id] = max_buckets;
        }
    }
}

__global__ void calc_bucket_ranges(
    uint* grid_hashes_keys,
    int* bucket_begin,
    int* bucket_end,
    int particle_num,
    int max_buckets)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < particle_num and grid_hashes_keys[id] < max_buckets)
    {
        if (id == 0)
        {
            bucket_begin[grid_hashes_keys[id]] = 0;
        }
        else if (grid_hashes_keys[id - 1] != grid_hashes_keys[id])
        {
            bucket_begin[grid_hashes_keys[id]] = id;
        }

        if (id == particle_num - 1)
        {
            bucket_end[grid_hashes_keys[id]] = particle_num;
        }
        else if (grid_hashes_keys[id + 1] != grid_hashes_keys[id])
        {
            bucket_end[grid_hashes_keys[id]] = id + 1;
        }
    }
}

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
    float dt)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < particle_num)
    {
        uint id = grid_hashes_values[i];
        uint grid_id = grid_hashes_keys[i];

        forces[3 * id + 0] = 0.0f;
        forces[3 * id + 1] = -GRAVITY * PARTICLE_MASS;
        forces[3 * id + 2] = 0.0f;

        if (grid_id >= max_buckets)
            return;

        int max_x = round(GRID_MAX_X / GRID_SIDE_LEN);
        int max_y = round(GRID_MAX_Y / GRID_SIDE_LEN);

        for (int xx = -1; xx <= 1; ++xx)
        {
            for (int yy = -1; yy <= 1; ++yy)
            {
                for (int zz = -1; zz <= 1; ++zz)
                {
                    int bucket = grid_id 
                                + xx
                                + yy * max_x 
                                + zz * max_x * max_y;
                    
                    if (bucket >= 0 and bucket < max_buckets)
                    {                        
                        for (int j = bucket_begin[bucket];
                                 j < bucket_end[bucket];
                                 ++j)
                        {
                            check_particle_collisions(positions, 
                                                      momenta, 
                                                      forces, 
                                                      id, 
                                                      grid_hashes_values[j], 
                                                      dt);
                        }
                    }
                }
            }
        }

        check_border_collisions(positions, momenta, forces, id, dt);

        forces[3 * id + 0] = max(forces[3 * id + 0], -MAX_FORCE);
        forces[3 * id + 0] = min(forces[3 * id + 0], MAX_FORCE);
        forces[3 * id + 1] = max(forces[3 * id + 1], -MAX_FORCE);
        forces[3 * id + 1] = min(forces[3 * id + 1], MAX_FORCE);
        forces[3 * id + 2] = max(forces[3 * id + 2], -MAX_FORCE);
        forces[3 * id + 2] = min(forces[3 * id + 2], MAX_FORCE);
    }
}
#endif
