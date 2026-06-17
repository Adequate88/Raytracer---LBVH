#pragma once

#include "vulkan_engine.h"
#include "vulkan_types.h"
#include <vulkan/vulkan_core.h>

#define DEVICE _engine._device
#define NODE_STRUCT_BYTES                                                      \
  64 // 64 Bytes = 4 Bytes * 6 (AABB) + 4 Bytes * 4 (Left, Right, Parent,
     // PrimIdx) + 12 bytes of padding
#define BVH_KERNELS 11

struct bufferMemory {
  VkBuffer buffer;
  VkDeviceMemory memory;
  size_t size;
};

class Bvh {
public:
  Bvh(VulkanEngine &engine);

  void init(size_t N, VkBuffer &primBuffer);
  void build();

private:
  VulkanEngine &_engine;

  bufferMemory bvhBuffer;
  bufferMemory primBboxBuffer;
  bufferMemory mortonCodesBuffer;
  bufferMemory primIndicesBuffer;
  bufferMemory worldBboxBuffer;
  bufferMemory histogramBuffer;
  bufferMemory scannedGramBuffer;
  bufferMemory globSumBuffer;
  bufferMemory outputMortonCodeBuffer;
  bufferMemory outputPrimIndicesBuffer;

  VkDescriptorSetLayout _bvhDescriptorLayout;
  VkDescriptorPool _bvhDescriptorPool;
  VkDescriptorSet _bvhDescriptor;
  VkPipelineLayout _bvhPipelineLayout;
  std::vector<VkPipeline> _bvhPipelines;

  void createBvhBuffers(size_t N);
  void createDescriptor(VkBuffer &primBuffer);
  void createPipelines();
};
