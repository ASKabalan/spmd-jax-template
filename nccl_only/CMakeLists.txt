cmake_minimum_required(VERSION 3.18)
project(CUFFT_Bench LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED VERSION 12)
set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)

string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MPI_INCLUDE_DIR ${NVHPC_MPI_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})

# Set source files
set(SOURCES src/main.cc src/nccl.cu src/mpi.cu)

# Add the executable
add_executable(spmd_template ${SOURCES})
target_link_libraries(spmd_template )
# CUDA architecture 70 80 89
set_target_properties(spmd_template PROPERTIES CUDA_ARCHITECTURES "80;89")

# Include directories for the executable
target_link_libraries(spmd_template PRIVATE NVHPC::CUDA NVHPC::MPI NVHPC::NCCL)
target_include_directories(spmd_template PRIVATE 
                           include
                           ${NVHPC_CUDA_INCLUDE_DIR}
                           ${NVHPC_MATH_INCLUDE_DIR}
                           ${NVHPC_MPI_INCLUDE_DIR}
                           ${NVHPC_NCCL_INCLUDE_DIR}
                          )

