#ifndef RUNNER_H
#define RUNNER_H

#include "mpi_ops.h"
#include <string>
#include <nccl.h>
#include <cuda_runtime.h>

class Runner {
public:
    Runner(const MPIOps &mpiOps, const std::string &begin_str, const std::string &end_str, int factor, const std::string &algo, const std::string &mode);
    ~Runner();

    void run();

private:
    const MPIOps &mpiOps;
    size_t begin_size;
    size_t end_size;
    int factor;
    std::string algo;
    std::string mode;
    size_t current_size;

    ncclComm_t comm;
    cudaStream_t stream;
    void *d_data;
    void *d_recv_data;

    size_t parse_size(const std::string &size_str);
    double calculate_bandwidth(size_t size, double time_ms);

    void allocate_memory(size_t size);
    void free_memory();
    void fill_ones(void* data, size_t size);

    void run_nccl_in_place_allreduce(size_t msg_size);
    void run_nccl_out_of_place_allreduce(size_t msg_size);
    void run_mpi_in_place_allreduce(size_t msg_size);
    void run_mpi_out_of_place_allreduce(size_t msg_size);

    void initialize_nccl();
    void finalize_nccl();
};

#endif // RUNNER_H

