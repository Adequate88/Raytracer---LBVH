#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance. This is modified from the original Ray Tracing Weekend CMake File.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

project ( LBVHGPU LANGUAGES CXX )

set ( CMAKE_CXX_STANDARD          20 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS        OFF )

# Find OpenCL
find_package(OpenCL REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(src)
include_directories(src/utils)

add_subdirectory(external)
add_subdirectory(src)



option(EVALUATE "Measure metrics of engine" ON)
