cmake_minimum_required(VERSION 3.18)
project(NCCL_API_Bench LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (NOT SKBUILD)
  # message(FATAL_ERROR "SKBUILD is not set")
endif()

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(CUDAToolkit REQUIRED VERSION 12)
set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MPI_INCLUDE_DIR ${NVHPC_MPI_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
# Set source files

# Add the executable
# CUDA architecture 70 80 89
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
# # Detect the installed nanobind package and import it into CMake
find_package(nanobind CONFIG REQUIRED) 

# Include directories for the executable
set(SOURCES lib/src/extensions.cc lib/src/gpu_ops.cu)
nanobind_add_module(nccl_mpi_lib STABLE_ABI NB_STATIC ${SOURCES})
# add_library(nccl_mpi_lib SHARED ${SOURCES})
set_target_properties(nccl_mpi_lib PROPERTIES CUDA_ARCHITECTURES "70;80;89")
target_link_libraries(nccl_mpi_lib  PRIVATE NVHPC::CUDA NVHPC::MPI NVHPC::NCCL)
target_include_directories(nccl_mpi_lib PUBLIC 
                           lib/include
                           ${XLA_DIR}
                           ${NVHPC_CUDA_INCLUDE_DIR}
                           ${NVHPC_MATH_INCLUDE_DIR}
                           ${NVHPC_MPI_INCLUDE_DIR}
                           ${NVHPC_NCCL_INCLUDE_DIR}
                          )

install(TARGETS nccl_mpi_lib LIBRARY DESTINATION ops)


