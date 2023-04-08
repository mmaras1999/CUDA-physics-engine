#include "cuda_utilities.hpp"
#include <exception>
#include <stdexcept>

void cudaCheckError()
{
    cudaError_t e = cudaGetLastError();

    if (e != cudaSuccess) 
    {
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
        throw std::runtime_error("cuda_error");
    }
}