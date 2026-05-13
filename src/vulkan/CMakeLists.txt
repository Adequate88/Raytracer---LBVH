
add_library(vulkan_engine STATIC
  vulkan_engine.cpp
)

target_include_directories(vulkan_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



