cmake_minimum_required(VERSION 3.15)
project(MNIST_ANN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-03 -march=native")

include_directories(external/eigen)
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(mnist_ann ${SOURCES})

target_link_libraries(mnist_ann wsock32 ws2_32)