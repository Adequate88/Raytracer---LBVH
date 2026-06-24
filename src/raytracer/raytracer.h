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
  // Wavefront variables:
  VkDescriptorSetLayout _wavefrontGenerateDescriptorSetLayout;
  VkDescriptorSetLayout _wavefrontUnifiedDescriptorSetLayout;
  VkDescriptorSetLayout _wavefrontFinalizeDescriptorSetLayout;
  VkDescriptorSet _wavefrontGenerateDescriptorSet;
  VkDescriptorSet _wavefrontUnifiedDescriptorSets[2];
  VkDescriptorSet _wavefrontFinalizeDescriptorSet;
  VkDescriptorPool _wavefrontDescriptorPool;

  VkPipelineLayout _wavefrontGenerateLayout;
  VkPipelineLayout _wavefrontUnifiedLayout;
  VkPipelineLayout _wavefrontFinalizeLayout;
  VkPipeline _wavefrontGeneratePipeline;
  VkPipeline _wavefrontExtendPipeline;
  VkPipeline _wavefrontShadePipeline;
  VkPipeline _wavefrontDispatchPipeline;
  VkPipeline _wavefrontFinalizePipeline;

  VkDeviceMemory _wavefrontRayBuffersMemory[2];
  VkDeviceMemory _wavefrontPathStateBuffersMemory[2];
  VkDeviceMemory _wavefrontHitRecordBufferMemory;
  VkDeviceMemory _wavefrontFinalRadianceBufferMemory;
  VkDeviceMemory _wavefrontNextRayCountBuffersMemory[2];
  VkDeviceMemory _wavefrontDispatchBufferMemory;

  VkMemoryBarrier2 _wavefrontMemoryBarrier;
  VkDependencyInfo _wavefrontMemoryDependency;
  // End of wavefront variables

  VkDeviceMemory _renderTargetMemory;
  VkDeviceMemory _sceneMemory;

  // The scene (primitive) and BVH node buffers are owned by the Bvh and passed
  // in here so the descriptor set can bind them. Build the BVH before calling.
  void initRaytracer(const void *cameraData, VkBuffer sceneBuffer,
                     VkBuffer bvhBuffer);
  void recordBuffer(uint32_t image_index);
  void recordRenderTime();

  void recordWavefrontBuffer(uint32_t image_index);
  VkBuffer &sceneBuffer() { return _sceneBuffer; }
  void forceRerender() { _rendered = false; }

private:
  VulkanEngine &_engine;
  const void *_cameraConstants;

  VkImage _renderTarget;
  VkImageView _renderTargetView;

  VkBuffer _sceneBuffer;
  // Wavefront variables:
  VkBuffer _wavefrontRayBuffers[2];
  VkBuffer _wavefrontPathStateBuffers[2];
  VkBuffer _wavefrontHitRecordBuffer;
  VkBuffer _wavefrontFinalRadianceBuffer;
  VkBuffer _wavefrontNextRayCountBuffers[2];
  VkBuffer _wavefrontDispatchBuffer;
  // End of wavefront variables

  VkBuffer _BvhBuffer;

  VkQueryPool _timestampPool;

  bool _rendered = false;

  void createTimestampPool();
  void createRenderTarget();
  void createDescriptors();

  void createShaderModule();
  void createPipeline();
  // Wavefront creation functions
  void createWavefrontBuffers();
  void createWavefrontDescriptors();
  void createWavefrontPipelines();
  void createWavefrontMemoryBarrier();
};
