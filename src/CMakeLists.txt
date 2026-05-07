
add_executable(engine
  utils/stb_image.cpp
  main.cpp
)

add_subdirectory(bvh)
add_subdirectory(raytracer)

target_link_libraries(engine PRIVATE raytracer)
target_link_libraries(engine PRIVATE bvh)


