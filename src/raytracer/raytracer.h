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
  VkDescriptorSetLayout _wavefrontExtendDescriptorSetLayout;
  VkDescriptorSetLayout _wavefrontShadeDescriptorSetLayout;
  VkDescriptorSetLayout _wavefrontFinalizeDescriptorSetLayout;
  VkDescriptorSet _wavefrontGenerateDescriptorSet;
  VkDescriptorSet _wavefrontExtendDescriptorSets[2];
  VkDescriptorSet _wavefrontShadeDescriptorSets[2];
  VkDescriptorSet _wavefrontFinalizeDescriptorSet;
  VkDescriptorPool _wavefrontDescriptorPool;

  VkPipelineLayout _wavefrontGenerateLayout;
  VkPipelineLayout _wavefrontExtendLayout;
  VkPipelineLayout _wavefrontShadeLayout;
  VkPipelineLayout _wavefrontFinalizeLayout;
  VkPipeline _wavefrontGeneratePipeline;
  VkPipeline _wavefrontExtendPipeline;
  VkPipeline _wavefrontShadePipeline;
  VkPipeline _wavefrontFinalizePipeline;

  VkDeviceMemory _wavefrontRayBuffersMemory[2];
  VkDeviceMemory _wavefrontPathStateBuffersMemory[2];
  VkDeviceMemory _wavefrontHitRecordBufferMemory;
  VkDeviceMemory _wavefrontFinalRadianceBufferMemory;
  VkDeviceMemory _wavefrontNextRayCountBufferMemory;

  VkMemoryBarrier2 _wavefrontMemoryBarrier;
  VkDependencyInfo _wavefrontMemoryDependency;
  // End of wavefront variables

  VkDeviceMemory _renderTargetMemory;
  VkDeviceMemory _sceneMemory;

  void initRaytracer(const void *data, size_t size, const void *cameraData);
  void recordBuffer(uint32_t image_index);

  void recordWavefrontBuffer(uint32_t image_index);

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
  VkBuffer _wavefrontNextRayCountBuffer;
  // End of wavefront variables

  void createRenderTarget();
  void createDescriptors();
  void createSceneBuffer(const void *data, size_t size);
  void createShaderModule();
  void createPipeline();
  // Wavefront creation functions
  void createWavefrontBuffers();
  void createWavefrontDescriptors();
  void createWavefrontPipelines();
  void createMemoryBarrier();
};
