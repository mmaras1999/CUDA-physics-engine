# **CUDA-physics-engine**
Simple CUDA physics engine based on GPU Gems 3 Physics Simulation

Full documentation is available in Polish language only.

### **Short description**

The project implements a simulation of rigid bodies. The collisions are calculated using a particle system. Usually the constants needs to be adjusted correctly for the best results.

The simulations uses similar technique for collision detection as described in [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus) or [Simon Green's tutorial](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf). 

In addition to the simulator, I provided a simple class that enables to transform any .obj object into particles.

### **Requirements**

You'll need:
- C++ compiler (e.g. g++)
- CUDA library with nvcc compiler
- OpenGL for visualization
- CUDA Thrust (available with the CUDA library)

### **Usage**
Create a build directory and use `make` command to compile the project. To run the project, use the following command:

```
./particle_simulation <path_to_obj_file:string> <path_to_config_file:string>
```

`path_to_obj_file` specifies the object that will be simulated,

`path_to_config_file` specifies the configuration file with the initial object setup.

There are three very simple setups provided, but you can easily create your own configuration files. The first line in such file is the number of objects, in the next `n` lines are three real numbers - the starting position of each object.


