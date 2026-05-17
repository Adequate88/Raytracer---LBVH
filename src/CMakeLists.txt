add_executable(engine
  utils/stb_image.cpp
  main.cpp
)


if(EVALUATE)
  message("Metrics enabled")
  add_compile_definitions(EVALUATE)
else()
  message("Metrics disabled")
endif()

add_subdirectory(bvh)
add_subdirectory(raytracer)
add_subdirectory(vulkan_engine)

target_link_libraries(engine PRIVATE raytracer)
target_link_libraries(engine PRIVATE bvh)
target_link_libraries(engine PRIVATE vulkan_engine)
target_link_libraries(engine PUBLIC fmt)


