#include "runner.h"
#include <iostream>
#include <chrono>

Runner::Runner(const MPIOps &mpiOps, const std::string &begin_str, const std::string &end_str, int factor, const std::string &algo, const std::string &mode)
    : mpiOps(mpiOps), begin_size(parse_size(begin_str)), end_size(parse_size(end_str)), factor(factor), algo(algo), mode(mode), current_size(begin_size) {

    if (mode == "nccl") {
        initialize_nccl();
    }
}

Runner::~Runner() {
    if (mode == "nccl") {
        finalize_nccl();
    }
    free_memory();
}

size_t Runner::parse_size(const std::string &size_str) {
    return std::stoull(size_str);
}

double Runner::calculate_bandwidth(size_t size, double time_ms) {
    double time_s = time_ms / 1000.0;
    return (size * 2.0 / (1024 * 1024 * 1024)) / time_s; // GB/s
}

void Runner::allocate_memory(size_t size) {
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_recv_data, size);
}

void Runner::free_memory() {
    cudaFree(d_data);
    cudaFree(d_recv_data);
}

void Runner::fill_ones(void* data, size_t size) {
    float *ptr = static_cast<float*>(data);
    for (size_t i = 0; i < size / sizeof(float); ++i) {
        ptr[i] = 1.0f;
    }
}

void Runner::run_nccl_in_place_allreduce(size_t msg_size) {
    allocate_memory(msg_size);
    fill_ones(d_data, msg_size);

    auto start = std::chrono::high_resolution_clock::now();

    ncclAllReduce(d_data, d_recv_data, msg_size / sizeof(float), ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double bandwidth = calculate_bandwidth(msg_size, duration.count() * 1000); // Convert to milliseconds

    std::cout << "[NCCL] In-place AllReduce: Size " << msg_size << " bytes, Bandwidth " << bandwidth << " GB/s" << std::endl;
}

void Runner::run_nccl_out_of_place_allreduce(size_t msg_size) {
    allocate_memory(msg_size);
    fill_ones(d_data, msg_size);

    auto start = std::chrono::high_resolution_clock::now();

    ncclAllReduce(d_data, d_recv_data, msg_size / sizeof(float), ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double bandwidth = calculate_bandwidth(msg_size, duration.count() * 1000); // Convert to milliseconds

    std::cout << "[NCCL] Out-of-place AllReduce: Size " << msg_size << " bytes, Bandwidth " << bandwidth << " GB/s" << std::endl;
}

void Runner::run_mpi_in_place_allreduce(size_t msg_size) {
    allocate_memory(msg_size);
    fill_ones(d_data, msg_size);

    auto start = std::chrono::high_resolution_clock::now();

    mpiOps.allreduce_in_place(d_data, msg_size);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double bandwidth = calculate_bandwidth(msg_size, duration.count() * 1000); // Convert to milliseconds

    std::cout << "[MPI] In-place AllReduce: Size " << msg_size << " bytes, Bandwidth " << bandwidth << " GB/s" << std::endl;
}

void Runner::run_mpi_out_of_place_allreduce(size_t msg_size) {
    allocate_memory(msg_size);
    fill_ones(d_data, msg_size);

    auto start = std::chrono::high_resolution_clock::now();

    mpiOps.allreduce_out_of_place(d_data, d_recv_data, msg_size);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double bandwidth = calculate_bandwidth(msg_size, duration.count() * 1000); // Convert to milliseconds

    std::cout << "[MPI] Out-of-place AllReduce: Size " << msg_size << " bytes, Bandwidth " << bandwidth << " GB/s" << std::endl;
}

void Runner::initialize_nccl() {
    ncclUniqueId id;
    if (mpiOps.get_rank() == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaSetDevice(mpiOps.get_local_rank());
    cudaStreamCreate(&stream);
    ncclCommInitRank(&comm, mpiOps.get_size(), id, mpiOps.get_rank());
}

void Runner::finalize_nccl() {
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
}

void Runner::run() {
    while (current_size <= end_size) {
        if (algo == "allreduce") {
            if (mode == "nccl") {
                run_nccl_in_place_allreduce(current_size);
                run_nccl_out_of_place_allreduce(current_size);
            } else if (mode == "mpi") {
                run_mpi_in_place_allreduce(current_size);
                run_mpi_out_of_place_allreduce(current_size);
            }
        }
        current_size *= factor;
    }
}

