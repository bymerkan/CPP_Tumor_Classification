cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-cpp)

set(CMAKE_PREFIX_PATH /Users/computer/Documents/torch-cpp/libtorch)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Torch REQUIRED)
include_directories(${TORCH_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable( torch-cpp
                src/main.cpp
                src/customDataset.cpp)

target_link_libraries(torch-cpp ${OpenCV_LIBS})

target_link_libraries(torch-cpp "${TORCH_LIBRARIES}")


set_property(TARGET torch-cpp PROPERTY CXX_STANDARD 20)
