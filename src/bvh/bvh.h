#pragma once

#include "vulkan_engine.h"
#include <vulkan/vulkan_core.h>

#define DEVICE _engine._device
#define BVH_BYTES                                                              \
  48 // 48 Bytes = 4 Bytes * 6 (AABB) + 4 Bytes * 4 (Left, Right, Parent,
     // PrimIdx)
#define BVH_KERNELS 5

class Bvh {
public:
  Bvh(VulkanEngine &engine);

  void init(size_t N, VkBuffer &primBuffer);
  void build();

private:
  VulkanEngine &_engine;

  VkBuffer _bvhBuffer;
  VkDeviceMemory _bvhMemory;

  VkDescriptorSetLayout _bvhDescriptorLayout;
  VkDescriptorPool _bvhDescriptorPool;
  VkDescriptorSet _bvhDescriptor;
  VkPipelineLayout _bvhPipelineLayout;
  std::vector<VkPipeline> _bvhPipelines;

  void createBvhBuffer(size_t N);
  void createDescriptor(VkBuffer &primBuffer);
  void createPipelines();
};
