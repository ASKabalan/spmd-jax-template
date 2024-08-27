
#ifndef NCCL_RUNNER_H
#define NCCL_RUNNER_H

#include "mpi_ops.h"
#include "perfostep.hpp"
#include <string>
#include <nccl.h>
#include <cuda_runtime.h>

class NCCLRunner {
public:
    NCCLRunner(const MPIOps &mpiOps, const std::string &begin_str, const std::string &end_str, int factor, const std::string &algo);
    ~NCCLRunner();
    void run();

private:
    MPIOps mpiOps;
    size_t begin_size;
    size_t end_size;
    int factor;
    std::string algo;

    void* d_data;       // Device memory for data
    void* d_recv_data;  // Device memory for receive buffer
    size_t current_size;

    ncclComm_t comm;
    cudaStream_t stream;

    size_t parse_size(const std::string &size_str);
    double calculate_bandwidth(size_t size, double time_ms);
    void run_nccl_allreduce(size_t msg_size);
    void run_nccl_allgather(size_t msg_size);
    void run_nccl_in_place_allreduce(size_t msg_size);
    void run_nccl_out_of_place_allreduce(size_t msg_size);
    void run_nccl_in_place_allgather(size_t msg_size);
    void run_nccl_out_of_place_allgather(size_t msg_size);
    void allocate_memory(size_t size);
    void free_memory();
    void fill_ones(void* data, size_t size);
    void initialize_nccl();
    void finalize_nccl();
};

#endif // NCCL_RUNNER_H
