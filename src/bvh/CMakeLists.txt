
add_library(bvh STATIC
  lbvh_gpu.cpp
  lbvh_cpu.cpp
  bvh_utils.cpp
)

target_include_directories(bvh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(bvh PRIVATE OpenCL::OpenCL)
target_link_libraries(bvh PRIVATE raytracer)



