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

  // The scene (primitive) and BVH node buffers are owned by the Bvh and passed
  // in here so the descriptor set can bind them. Build the BVH before calling.
  void initRaytracer(const void *cameraData, VkBuffer sceneBuffer,
                     VkBuffer bvhBuffer);
  void recordBuffer(uint32_t image_index);
  void recordRenderTime();
  void forceRerender() { _rendered = false; }

private:
  VulkanEngine &_engine;
  const void *_cameraConstants;

  VkImage _renderTarget;
  VkImageView _renderTargetView;

  VkBuffer _sceneBuffer;
  VkBuffer _BvhBuffer;

  VkQueryPool _timestampPool;

  bool _rendered = false;

  void createTimestampPool();
  void createRenderTarget();
  void createDescriptors();

  void createShaderModule();
  void createPipeline();
};
