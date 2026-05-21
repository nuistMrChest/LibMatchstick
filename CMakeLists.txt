cmake_minimum_required(VERSION 3.16)

project(matchstick LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(
	matchstick SHARED
	src/activation.cu
	src/layer.cu
	src/losses.cu
	src/matrix.cu
	src/network.cu
	src/tensor_3d.cu
	src/tensor_4d.cu
)

target_include_directories(
	matchstick PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(matchstick PROPERTIES
    CUDA_ARCHITECTURES native
)
