cmake_minimum_required(VERSION 3.15)
project(tracker_cpp)

set(CMAKE_CXX_STANDARD 17)

# Path to Eigen (adjust if needed)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../external/eigen-5.0.0)

# pybind11
find_package(pybind11 REQUIRED)

# Build the Python module
pybind11_add_module(tracker_cpp
    kalman.cpp
    gating.cpp
    association.cpp
    bindings.cpp
)
