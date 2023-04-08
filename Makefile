# Thanks to Job Vranish (https://spin.atomicobject.com/2016/08/26/makefile-c-projects/)
TARGET_EXEC := particle_simulation

CXXFLAGS := -std=c++17 -O3

BUILD_DIR := ./build
SRC_DIRS := ./src

# Find all the C++ files we want to compile
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
CUDA_SRCS := $(shell find $(SRC_DIRS) -name '*.cu')

# String substitution for every C/C++ file.
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
CUDA_OBJS := $(CUDA_SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
DEPS := $(OBJS:.o=.d) $(CUDA_OBJS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Generate Makefiles
# These files will have .d instead of .o as the output.
OPENGLFLAGS := -lGL -lglfw -lm
CUDAFLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcurand -arch=sm_61 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
CPPFLAGS := $(INC_FLAGS) -MMD -MP
NVCCFLAGS := $(INC_FLAGS) -MMD -MP

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS) $(CUDA_OBJS)
	nvcc $(OBJS) $(CUDA_OBJS) -o $@ $(LDFLAGS) -lstdc++ -lepoxy $(OPENGLFLAGS) $(CUDAFLAGS)
	rm -f particle_simulation
	ln -s build/particle_simulation particle_simulation
	cp -r resources $(dir $@)/resources

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	nvcc $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@ $(CUDAFLAGS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f particle_simulation

-include $(DEPS)