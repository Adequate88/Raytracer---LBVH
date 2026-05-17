find_package(Vulkan REQUIRED)

add_subdirectory(vk-bootstrap)
add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)
add_subdirectory(fmt)

