cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME get_set)

# ------------------------ CUDA ------------------------
set(CUDA_ROOT /usr/local/cuda-12)
set(CUDA_INC ${CUDA_ROOT}/include)
set(CUDA_LIB ${CUDA_ROOT}/lib64)
set(CUDA_LIBRARIES # "libcuda.so" # -> Linuxだと不要
"libcudart.so" "libcusparse.so" "libcusolver.so")
# set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)

project(main
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

# Find cuDSS
# find_package(cudss 0.4.0 REQUIRED)

# [FIXME]
set(cudss_ROOT /home/goto/LIB/cuDSS/0.4.0)
set(cudss_INCLUDE_DIR ${cudss_ROOT}/include)
set(cudss_LIBRARY_DIR ${cudss_ROOT}/lib)

message(STATUS ${cudss_INCLUDE_DIR})
message(STATUS ${cudss_LIBRARY_DIR})

add_executable(main main.cpp)
# add_executable(main sample.cpp)

target_include_directories(main PRIVATE ${CUDA_INC} ${cudss_INCLUDE_DIR})

target_link_directories(main PRIVATE ${CUDA_LIB} ${cudss_LIBRARY_DIR})

target_link_libraries(main PRIVATE ${CUDA_LIBRARIES} cudss)
