add_library(raytracer STATIC
  aabb_refactored.cpp
  interval.cpp
)

target_link_libraries(raytracer PRIVATE bvh engine OpenCL::OpenCL)
target_include_directories(raytracer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



