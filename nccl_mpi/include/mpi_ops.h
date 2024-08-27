#ifndef MPI_OPS_H
#define MPI_OPS_H

#include <mpi.h>
#include <cuda_runtime.h>

class MPIOps {
public:
    static MPIOps& instance() {
        static MPIOps instance;
        return instance;
    }

    void mpi_init(int& argc, char**& argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    void mpi_finalize() {
        MPI_Finalize();
    }

    int get_rank() const {
        return rank;
    }

    int get_size() const {
        return size;
    }

    void mpi_allgather(const void* sendbuf, size_t length, void* recvbuf, bool inplace) {
        if (inplace) {
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf, length, MPI_FLOAT, MPI_COMM_WORLD);
        } else {
            MPI_Allgather(sendbuf, length, MPI_FLOAT, recvbuf, length, MPI_FLOAT, MPI_COMM_WORLD);
        }
    }

    void mpi_allreduce(const void* sendbuf, void* recvbuf, size_t length, bool inplace) {
        if (inplace) {
            MPI_Allreduce(MPI_IN_PLACE, recvbuf, length, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        } else {
            MPI_Allreduce(sendbuf, recvbuf, length, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    void mpi_alltoall(const void* sendbuf, size_t length, void* recvbuf, bool inplace) {
        if (inplace) {
            MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf, length, MPI_FLOAT, MPI_COMM_WORLD);
        } else {
            MPI_Alltoall(sendbuf, length, MPI_FLOAT, recvbuf, length, MPI_FLOAT, MPI_COMM_WORLD);
        }
    }

    void mpi_sendrecv(const void* sendbuf, size_t send_length, int dest, 
                      void* recvbuf, size_t recv_length, int source, bool inplace) {
        if (inplace) {
            MPI_Sendrecv(MPI_IN_PLACE, send_length, MPI_FLOAT, dest, 0,
                         recvbuf, recv_length, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Sendrecv(sendbuf, send_length, MPI_FLOAT, dest, 0,
                         recvbuf, recv_length, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

private:
    MPIOps() = default;
    int rank = -1;
    int size = -1;
};

#endif // MPI_OPS_H

