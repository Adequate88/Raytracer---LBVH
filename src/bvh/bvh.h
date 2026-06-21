#pragma once

#include "vulkan_engine.h"
#include "vulkan_types.h"
#include <cstddef>
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
  Bvh(VulkanEngine &engine, size_t primitiveCount);

  void init(const void *data, size_t size);
  void build();

  // The Bvh owns the scene (primitive) buffer and the node buffer; the
  // raytracer binds both of these for traversal.
  VkBuffer sceneBufferHandle() const { return sceneBuffer.buffer; }
  VkBuffer bvhBufferHandle() const { return bvhBuffer.buffer; }

private:
  VulkanEngine &_engine;

  size_t _primitiveCount;
  size_t groups;
  size_t histogramElems;
  size_t numBlocks;

  bufferMemory sceneBuffer;
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
  VkDescriptorSet _bvhDescriptors[2]; // [0] = normal, [1] = ping-pong swapped
  VkPipelineLayout _bvhPipelineLayout;
  std::vector<VkPipeline> _bvhPipelines;

  void createBvhBuffers();
  void createSceneBuffer(const void *data, size_t size);
  void createDescriptor();
  void createPipelines();
};
