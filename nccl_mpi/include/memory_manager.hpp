#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

class MemoryManager {
public:
    static MemoryManager& instance() {
        static MemoryManager instance;
        return instance;
    }

    // Allocate memory on GPU, set to ones using CPU, and return GPU pointer
    float* allocate_and_initialize(size_t size) {
        // Allocate CPU memory
        std::vector<float> host_memory(size, 1.0f);

        // Allocate GPU memory
        float* device_memory;
        cudaMalloc(&device_memory, size * sizeof(float));

        // Copy initialized data from CPU to GPU
        cudaMemcpy(device_memory, host_memory.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Store allocated pointer for cleanup
        gpu_pointers.push_back(device_memory);

        return device_memory;
    }

    // Reset GPU memory to ones using CPU
    void reset_to_ones(float* device_memory, size_t size) {
        // Allocate CPU memory with ones
        std::vector<float> host_memory(size, 1.0f);

        // Copy initialized data from CPU to GPU
        cudaMemcpy(device_memory, host_memory.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Cleanup all allocated memory
    void cleanup() {
        for (auto ptr : gpu_pointers) {
            cudaFree(ptr);
        }
        gpu_pointers.clear();
    }

private:
    MemoryManager() = default;
    ~MemoryManager() {
        cleanup();
    }

    // Disallow copy and assignment
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    std::vector<float*> gpu_pointers;
};

#endif // MEMORY_MANAGER_H

