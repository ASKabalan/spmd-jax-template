cmake_minimum_required(VERSION 3.18)
project(CUFFT_Bench LANGUAGES CXX CUDA)

add_subdirectory(nccl_mpi)


find_package(CUDAToolkit REQUIRED VERSION 12)
set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MPI_INCLUDE_DIR ${NVHPC_MPI_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})

# add executable
add_executable(nccl_mpi_bench src/main.cc)
target_link_libraries(nccl_mpi_bench PRIVATE nccl_mpi)
set_target_properties(nccl_mpi_bench PROPERTIES CUDA_ARCHITECTURES "80;89")

target_link_libraries(nccl_mpi_bench  PRIVATE nccl_mpi NVHPC::CUDA NVHPC::MPI NVHPC::NCCL)
target_include_directories(nccl_mpi_bench PRIVATE 
                           include
                           ${NCCL_MPI_INCLUDE_DIR}
                           ${NVHPC_CUDA_INCLUDE_DIR}
                           ${NVHPC_MATH_INCLUDE_DIR}
                           ${NVHPC_MPI_INCLUDE_DIR}
                           ${NVHPC_NCCL_INCLUDE_DIR}
                          )

