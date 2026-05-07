add_library(raytracer STATIC
  camera.cpp
)

target_link_libraries(raytracer PRIVATE bvh OpenCL::OpenCL)
target_include_directories(raytracer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



