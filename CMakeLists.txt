cmake_minimum_required(VERSION 3.15)
project(MNIST_ANN)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "/O2")

add_definitions(-DNOMINMAX)

include_directories(external/eigen)
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(mnist_ann ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_ann wsock32 ws2_32 OpenMP::OpenMP_CXX)
    target_compile_options(mnist_ann PRIVATE /openmp)
else()
    target_link_libraries(mnist_ann wsock32 ws2_32)
endif()