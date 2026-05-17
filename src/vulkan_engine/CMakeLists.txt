
add_library(vulkan_engine STATIC
  vulkan_engine.cpp
)

target_include_directories(vulkan_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(vulkan_engine PUBLIC Vulkan::Vulkan)
target_link_libraries(vulkan_engine PUBLIC SDL3::SDL3)


