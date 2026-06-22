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

option(EVALUATE "Measure metrics of engine" OFF)
if(EVALUATE)
  add_compile_definitions(EVALUATE)
endif()

option(RGP_CAPTURE "Emit deterministic RGP/RADV SQTT capture frames at startup" OFF)
if(RGP_CAPTURE)
  add_compile_definitions(RGP_CAPTURE)
endif()

add_subdirectory(external)
add_subdirectory(src)

# Compile GLSL shaders to SPIRV

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )


file(GLOB_RECURSE GLSL_INCLUDE_FILES "${PROJECT_SOURCE_DIR}/shaders/*.glsl")

foreach(GLSL ${GLSL_SOURCE_FILES})
  message("Starting to build shaders")
  get_filename_component(FILENAME ${GLSL} NAME_WE)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILENAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V -I${PROJECT_SOURCE_DIR}/shaders ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL} ${GLSL_INCLUDE_FILES})
  list (APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(Shaders DEPENDS ${SPIRV_BINARY_FILES})

add_dependencies(engine Shaders)

