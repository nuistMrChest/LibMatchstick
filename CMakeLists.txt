cmake_minimum_required(VERSION 3.24)

project(
    LibMatchstick
    VERSION 2.0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(matchstick SHARED
    src/cpp/activation.cu
    src/cpp/layer.cu
    src/cpp/loss.cu
    src/cpp/matrix.cu
    src/cpp/network.cu
    src/cpp/tensor_3d.cu
    src/cpp/tensor_4d.cu

    src/c_api/matrix_capi.cu
    src/c_api/network_capi.cu
    src/c_api/tensor_3d_capi.cu
    src/c_api/tensor_4d_capi.cu
)

set_target_properties(matchstick PROPERTIES
    OUTPUT_NAME "matchstick"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(matchstick
    BEFORE PRIVATE
        ${PROJECT_SOURCE_DIR}/internal
        ${PROJECT_SOURCE_DIR}/src/c_api
)

target_include_directories(matchstick
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matchstick
    PRIVATE
        CUDA::cudart
)

install(DIRECTORY include/
    DESTINATION include
)
