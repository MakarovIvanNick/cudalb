cmake_minimum_required(VERSION 3.16)
project(cudalb LANGUAGES CUDA CXX)

set(OpenCV_DIR "../opencv/build_with_cuda")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cudalb kernel.cu)
target_link_libraries(cudalb ${OpenCV_LIBS})