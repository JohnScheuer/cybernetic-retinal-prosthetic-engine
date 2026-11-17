cmake_minimum_required(VERSION 3.22)
project(cybernetic_retinal_prosthetic_engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC -Xcompiler -march=native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(retinal_prosthetic_engine
    src/main.cpp
    src/cuda_stimulation_pattern.cu
)

set_target_properties(retinal_prosthetic_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(retinal_prosthetic_engine PRIVATE CUDA::cudart)
