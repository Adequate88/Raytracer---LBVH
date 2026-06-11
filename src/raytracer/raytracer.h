#pragma once

#include "vulkan_engine.h"
#include "vulkan_engine/vulkan_engine.h"
#include <cstdint>
#include <vulkan/vulkan_core.h>

class Raytracer {
public:
  Raytracer(VulkanEngine &engine);

  VkDescriptorSetLayout _setLayout;
  VkDescriptorSet _descriptorSet;
  VkDescriptorPool _descriptorPool;
  VkPipelineLayout _layout;
  VkPipeline _pipeline;

  VkDeviceMemory _renderTargetMemory;

  void initRaytracer();
  void recordBuffer(uint32_t image_index);

private:
  VulkanEngine &_engine;

  VkImage _renderTarget;
  VkImageView _renderTargetView;

  void createRenderTarget();
  void createBuffers();
  void createDescriptors();
  void createShaderModule();
  void createPipeline();
};
