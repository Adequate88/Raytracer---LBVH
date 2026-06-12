#pragma once

#include "vulkan_engine.h"
#include "vulkan_engine/vulkan_engine.h"
#include <cstdint>
#include <iterator>
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
  VkDeviceMemory _sceneMemory;

  void initRaytracer(const void *data, size_t size);
  void recordBuffer(uint32_t image_index);

private:
  VulkanEngine &_engine;

  VkImage _renderTarget;
  VkImageView _renderTargetView;

  VkBuffer _sceneBuffer;

  void createRenderTarget();
  void createDescriptors();
  void createSceneBuffer(const void *data, size_t size);
  void createShaderModule();
  void createPipeline();
};
