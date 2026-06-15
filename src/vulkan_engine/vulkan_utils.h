#pragma once

#include "vulkan_engine.h"
#include <fstream>
#include <stdexcept>

inline uint32_t find_memory_type(
    VkMemoryPropertyFlags properties,
    uint32_t desired_type) { // TODO DELETE THIS OR THE ONE IN
                             // vulkan_engine.cpp, just rewrote it for practice

  VkPhysicalDeviceMemoryProperties gpuProperties;
  vkGetPhysicalDeviceMemoryProperties(VulkanEngine::Get()._gpu, &gpuProperties);

  for (uint32_t mem_index = 0; mem_index < gpuProperties.memoryTypeCount;
       mem_index++) {
    if ((1u << mem_index) & desired_type &&
        (properties & gpuProperties.memoryTypes[mem_index].propertyFlags) ==
            properties) {
      return mem_index;
    }
  }

  fmt::println("COULD NOT FIND DESIRED MEMORY TYPE");
  throw std::runtime_error("Could not find memory type");
  return -1;
}

inline VkShaderModule load_shader(const std::string &path, VkDevice &device) {

  std::ifstream file(path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    fmt::println("Failed to open shader");
    throw std::runtime_error("Failed to open shader");
  }

  size_t size = file.tellg();
  file.seekg(0);

  std::vector<uint32_t> data(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(data.data()), size);

  VkShaderModuleCreateInfo moduleInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = data.data()};

  VkShaderModule module;
  VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));

  return module;
}
