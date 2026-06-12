
add_library(bvh STATIC
  lbvh_gpu.cpp
  lbvh_cpu.cpp
)

target_include_directories(bvh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(bvh PUBLIC OpenCL::OpenCL)
target_link_libraries(bvh PRIVATE raytracer)



