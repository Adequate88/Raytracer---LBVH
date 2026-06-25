#pragma once

#include "vulkan_engine.h"
#include "vulkan_types.h"
#include <cstddef>
#include <vulkan/vulkan_core.h>

#define DEVICE _engine._device
#define NODE_STRUCT_BYTES                                                      \
  64 // 64 Bytes = 4 Bytes * 6 (AABB) + 4 Bytes * 4 (Left, Right, Parent,
     // PrimIdx) + 12 bytes of padding
#define BVH_KERNELS 12

struct bufferMemory {
  VkBuffer buffer;
  VkDeviceMemory memory;
  size_t size;
};

struct BvhTraversalBuffers {
  VkBuffer corners[6]; // 1x,1y,1z,2x,2y,2z
  VkBuffer leftChilds;
  VkBuffer rightChilds;
  VkBuffer primIndices;
  int primitiveCount;
};

class Bvh {
public:
  Bvh(VulkanEngine &engine, size_t primitiveCount);

  void init(const void *data, size_t size);
  void build();

  // The Bvh owns the scene (primitive) buffer and the node buffer; the
  // raytracer binds both of these for traversal.
  VkBuffer sceneBufferHandle() const { return sceneBuffer.buffer; }
  BvhTraversalBuffers traversalHandles() const {
    return {{cornerBuffers[0].buffer, cornerBuffers[1].buffer,
             cornerBuffers[2].buffer, cornerBuffers[3].buffer,
             cornerBuffers[4].buffer, cornerBuffers[5].buffer},
            leftChildsBuffer.buffer,
            rightChildsBuffer.buffer,
            primIndicesBuffer.buffer,
            (int)_primitiveCount};
  }

private:
  VulkanEngine &_engine;

  size_t _primitiveCount;
  size_t groups;
  size_t histogramElems;
  size_t numBlocks;

  bufferMemory sceneBuffer;
  bufferMemory mortonCodesBuffer;
  bufferMemory primIndicesBuffer;
  bufferMemory worldBboxBuffer;
  bufferMemory histogramBuffer;
  bufferMemory scannedGramBuffer;
  bufferMemory globSumBuffer;
  bufferMemory outputMortonCodeBuffer;
  bufferMemory outputPrimIndicesBuffer;

  bufferMemory cornerBuffers[6];    // 11-16
  bufferMemory leftChildsBuffer;    // 17
  bufferMemory rightChildsBuffer;   // 18
  bufferMemory parentsBuffer;       // 19
  bufferMemory atomicCountersBuffer; // 20
  bufferMemory cornerOutBuffers[6]; // 21-26

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
