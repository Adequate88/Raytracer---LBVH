#include "raytracer.h"
#include "metrics_macros.h"
#include "vulkan_engine.h"
#include "vulkan_types.h"
#include "vulkan_utils.h"
#include <cstdint>

#include <vulkan/vulkan_core.h>

Raytracer::Raytracer(VulkanEngine &engine) : _engine(engine) {}

void Raytracer::initRaytracer(const void *cameraData, VkBuffer sceneBuffer,
                              VkBuffer bvhBuffer) {
  _cameraConstants = cameraData;
  _sceneBuffer = sceneBuffer;
  _BvhBuffer = bvhBuffer;

  createTimestampPool();
  createRenderTarget();
  createDescriptors();
  createPipeline();

  // Wavefront
  createWavefrontBuffers();
  createWavefrontDescriptors();
  createWavefrontPipelines();
  createWavefrontMemoryBarrier();
}

void Raytracer::createTimestampPool() {
  VkQueryPoolCreateInfo queryInfo{.sType =
                                      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = 2};
  VK_CHECK(
      vkCreateQueryPool(_engine._device, &queryInfo, nullptr, &_timestampPool));
}

void Raytracer::createRenderTarget() {
  uint32_t queueFamilies[2] = {_engine._queueFamilyIndex,
                               _engine._computeQueueFamilyIndex};
  bool concurrent =
      _engine._queueFamilyIndex != _engine._computeQueueFamilyIndex;

  VkImageCreateInfo imgInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {_engine._windowExtent.width, _engine._windowExtent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode =
          concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = concurrent ? 2u : 0u,
      .pQueueFamilyIndices = concurrent ? queueFamilies : nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  VK_CHECK(vkCreateImage(_engine._device, &imgInfo, nullptr, &_renderTarget));

  VkMemoryRequirements targetRequirements;
  vkGetImageMemoryRequirements(_engine._device, _renderTarget,
                               &targetRequirements);
  uint32_t mem_type = find_memory_type(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       targetRequirements.memoryTypeBits);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = targetRequirements.size,
                                 .memoryTypeIndex = mem_type};
  VK_CHECK(vkAllocateMemory(_engine._device, &allocInfo, nullptr,
                            &_renderTargetMemory));
  VK_CHECK(vkBindImageMemory(_engine._device, _renderTarget,
                             _renderTargetMemory, 0));

  VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = _renderTarget,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

  VK_CHECK(vkCreateImageView(_engine._device, &viewInfo, nullptr,
                             &_renderTargetView));
}

void Raytracer::createDescriptors() {
  VkDescriptorSetLayoutBinding renderBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding sceneBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding BvhBinding{
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding layoutBindings[3] = {renderBinding, sceneBinding,
                                                    BvhBinding};

  VkDescriptorSetLayoutCreateInfo setInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = layoutBindings};
  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &setInfo, nullptr,
                                       &_setLayout));

  // Pool
  VkDescriptorPoolSize renderPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .descriptorCount = 1};
  VkDescriptorPoolSize scenePoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     .descriptorCount = 1};
  VkDescriptorPoolSize BvhPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   .descriptorCount = 1};

  VkDescriptorPoolSize poolSizes[3] = {renderPoolSize, scenePoolSize,
                                       BvhPoolSize};
  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 3,
      .pPoolSizes = poolSizes};

  VK_CHECK(vkCreateDescriptorPool(_engine._device, &poolInfo, nullptr,
                                  &_descriptorPool));

  VkDescriptorSetAllocateInfo descriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_setLayout};

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &descriptorAllocInfo,
                                    &_descriptorSet));

  VkDescriptorImageInfo descriptorImageInfo{
      .imageView = _renderTargetView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

  VkWriteDescriptorSet writeRender{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &descriptorImageInfo};

  VkDescriptorBufferInfo descriptorBufferInfo{_sceneBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeScene{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorBufferInfo};

  VkDescriptorBufferInfo BvhDescriptorBufferInfo{_BvhBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeBvh{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = _descriptorSet,
                                .dstBinding = 2,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType =
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .pBufferInfo = &BvhDescriptorBufferInfo};

  VkWriteDescriptorSet writeSets[3] = {writeRender, writeScene, writeBvh};
  vkUpdateDescriptorSets(_engine._device, 3, writeSets, 0, nullptr);
}

void Raytracer::createPipeline() {

  VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = 64}; // Currently hard-coded 64 Bytes

  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_setLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange};

  VK_CHECK(
      vkCreatePipelineLayout(_engine._device, &layoutInfo, nullptr, &_layout));

  VkShaderModule toyShader =
      load_shader("shaders/toyshader.spv", _engine._device);

  VkPipelineShaderStageCreateInfo shaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = toyShader,
      .pName = "main"};
  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = shaderStageInfo,
      .layout = _layout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
                                    &pipelineInfo, nullptr, &_pipeline));

  vkDestroyShaderModule(_engine._device, toyShader, nullptr);
}

void Raytracer::recordBuffer(uint32_t image_index) {
  VkCommandBuffer computeCmd =
      _engine._computeCommandBuffers[_engine.frame_index];
  VkCommandBuffer graphicsCmd = _engine._commandBuffers[_engine.frame_index];

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  // Compute submission: the raytrace dispatch (runs on the 60s compute ring)
  VK_CHECK(vkBeginCommandBuffer(computeCmd, &beginInfo));

  if (!_rendered) {
    vkCmdResetQueryPool(computeCmd, _timestampPool, 0, 2);

    vkCmdPushConstants(computeCmd, _layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64,
                       _cameraConstants);

    _engine.transition_image_layout(
        computeCmd, _renderTarget, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL, {}, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, _layout,
                            0, 1, &_descriptorSet, 0, nullptr);

    vkCmdWriteTimestamp2(computeCmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         _timestampPool, 0);

    vkCmdDispatch(computeCmd, (_engine._windowExtent.width + 15) / 16,
                  (_engine._windowExtent.height + 15) / 16, 1);

    vkCmdWriteTimestamp2(computeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         _timestampPool, 1);

    _rendered = true;
  }
  VK_CHECK(vkEndCommandBuffer(computeCmd));
  // Graphics submission: blit to swapchain + present (waits on computeFinished)
  VK_CHECK(vkBeginCommandBuffer(graphicsCmd, &beginInfo));

  _engine.transition_image_layout(
      graphicsCmd, _renderTarget, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, {}, VK_ACCESS_2_TRANSFER_READ_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  _engine.transition_image_layout(
      graphicsCmd, _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {},
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  VkImageBlit blitRegion{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0},
                     {(int32_t)_engine._windowExtent.width,
                      (int32_t)_engine._windowExtent.height, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0},
                     {(int32_t)_engine._swapchainExtent.width,
                      (int32_t)_engine._swapchainExtent.height, 1}}};
  vkCmdBlitImage(
      graphicsCmd, _renderTarget, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

  _engine.transition_image_layout(
      graphicsCmd, _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, {}, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  VK_CHECK(vkEndCommandBuffer(graphicsCmd));
}

void Raytracer::recordRenderTime() {
  uint64_t timestamps[2];
  VK_CHECK(vkGetQueryPoolResults(
      _engine._device, _timestampPool, 0, 2, sizeof(timestamps), timestamps,
      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

  float ms = static_cast<float>(timestamps[1] - timestamps[0]) *
             _engine._timestampPeriod * 1e-6f;
  METRIC_SET_VALUE("Total Rendering Time", ms);
}

// Wavefront
void Raytracer::createWavefrontBuffers() {
  // SAMPLES = 1
  const VkDeviceSize nrOfPaths =
      _engine._windowExtent.width * _engine._windowExtent.height * 1;

  // Ray buffers
  _engine.allocate_buffer(
      _wavefrontRayBuffers[0], _wavefrontRayBuffersMemory[0],
      nrOfPaths * 32, // Based on ray.glsl
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  _engine.allocate_buffer(
      _wavefrontRayBuffers[1], _wavefrontRayBuffersMemory[1],
      nrOfPaths * 32, // Based on ray.glsl
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Path State buffers
  _engine.allocate_buffer(
      _wavefrontPathStateBuffers[0], _wavefrontPathStateBuffersMemory[0],
      nrOfPaths * 48, // Based on path_state.glsl
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  _engine.allocate_buffer(
      _wavefrontPathStateBuffers[1], _wavefrontPathStateBuffersMemory[1],
      nrOfPaths * 48, // Based on path_state.glsl
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Hit Record buffer
  _engine.allocate_buffer(
      _wavefrontHitRecordBuffer, _wavefrontHitRecordBufferMemory,
      nrOfPaths * 64, // Based on triangle.glsl
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Final Radiance buffer
  _engine.allocate_buffer(
      _wavefrontFinalRadianceBuffer, _wavefrontFinalRadianceBufferMemory,
      nrOfPaths * 16, // vec3 + padding from std430
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Next Ray Count buffers
  _engine.allocate_buffer(
    _wavefrontNextRayCountBuffers[0], _wavefrontNextRayCountBuffersMemory[0], 
    sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  _engine.allocate_buffer(
      _wavefrontNextRayCountBuffers[1], _wavefrontNextRayCountBuffersMemory[1],
      sizeof(uint32_t),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  _engine.allocate_buffer(
      _wavefrontDispatchBuffer, _wavefrontDispatchBufferMemory,
      sizeof(uint32_t) * 3, 
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void Raytracer::createWavefrontDescriptors() {
  // Descriptor Set Layout
  // Generate
  VkDescriptorSetLayoutBinding renderGenerateBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding raysGenerateBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding pathStateGenerateBinding{
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding layoutGenerateBindings[3] = {
      renderGenerateBinding, raysGenerateBinding, pathStateGenerateBinding};

  VkDescriptorSetLayoutCreateInfo generateSetInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = layoutGenerateBindings};

  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &generateSetInfo,
                                       nullptr,
                                       &_wavefrontGenerateDescriptorSetLayout));

  // Extend
  VkDescriptorSetLayoutBinding raysExtendBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding hitRecordsExtendBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding sceneExtendBinding{
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding lastRayCountExtendBinding{
      .binding = 3,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

  VkDescriptorSetLayoutBinding layoutExtendBindings[4] = {
      raysExtendBinding, hitRecordsExtendBinding, sceneExtendBinding,
      lastRayCountExtendBinding };

  VkDescriptorSetLayoutCreateInfo extendSetInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4,
      .pBindings = layoutExtendBindings};

  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &extendSetInfo, nullptr,
                                       &_wavefrontExtendDescriptorSetLayout));

  // Shade
  VkDescriptorSetLayoutBinding raysInShadeBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding pathStatesInShadeBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding hitRecordsShadeBinding{
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding lastRayCountShadeBinding{
      .binding = 3,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

  VkDescriptorSetLayoutBinding raysOutShadeBinding{
      .binding = 4,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding pathStatesOutShadeBinding{
      .binding = 5,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding finalRadianceShadeBinding{
      .binding = 6,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding nextRayCountShadeBinding{
      .binding = 7,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding layoutShadeBindings[8] = {
      raysInShadeBinding,        pathStatesInShadeBinding,
      hitRecordsShadeBinding,    lastRayCountShadeBinding,
      raysOutShadeBinding,       pathStatesOutShadeBinding, 
      finalRadianceShadeBinding, nextRayCountShadeBinding};

  VkDescriptorSetLayoutCreateInfo shadeSetInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 8,
      .pBindings = layoutShadeBindings};

  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &shadeSetInfo, nullptr,
                                       &_wavefrontShadeDescriptorSetLayout));

  // Dispatch
  VkDescriptorSetLayoutBinding nextRayCountDispatchBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

  VkDescriptorSetLayoutBinding dispatchSizeDispatchBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

  VkDescriptorSetLayoutBinding layoutDispatchBindings[2] = {
      nextRayCountDispatchBinding, dispatchSizeDispatchBinding};

  VkDescriptorSetLayoutCreateInfo dispatchSetInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = layoutDispatchBindings };

  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &dispatchSetInfo, nullptr,
      &_wavefrontDispatchDescriptorSetLayout));

  // Finalize
  VkDescriptorSetLayoutBinding renderFinalizeBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding finalRadianceFinalizeBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding layoutFinalizeBindings[2] = {
      renderFinalizeBinding, finalRadianceFinalizeBinding};

  VkDescriptorSetLayoutCreateInfo finalizeSetInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = layoutFinalizeBindings};

  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &finalizeSetInfo,
                                       nullptr,
                                       &_wavefrontFinalizeDescriptorSetLayout));

  // Pool
  VkDescriptorPoolSize renderPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .descriptorCount = 2};
  VkDescriptorPoolSize buffersPoolSize{
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 31};

  VkDescriptorPoolSize poolSizes[2] = {renderPoolSize, buffersPoolSize};
  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 8,
      .poolSizeCount = 2,
      .pPoolSizes = poolSizes};

  VK_CHECK(vkCreateDescriptorPool(_engine._device, &poolInfo, nullptr,
                                  &_wavefrontDescriptorPool));

  // Allocate
  // Generate
  VkDescriptorSetAllocateInfo generateDescriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _wavefrontDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_wavefrontGenerateDescriptorSetLayout};

  VK_CHECK(vkAllocateDescriptorSets(_engine._device,
                                    &generateDescriptorAllocInfo,
                                    &_wavefrontGenerateDescriptorSet));

  // Extend (needs 2 for the in and output buffers switching)
  VkDescriptorSetAllocateInfo extendDescriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _wavefrontDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_wavefrontExtendDescriptorSetLayout};

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &extendDescriptorAllocInfo,
                                    &_wavefrontExtendDescriptorSets[0]));

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &extendDescriptorAllocInfo,
                                    &_wavefrontExtendDescriptorSets[1]));

  // Shade (needs 2 for the in and output buffers switching)
  VkDescriptorSetAllocateInfo shadeDescriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _wavefrontDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_wavefrontShadeDescriptorSetLayout};

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &shadeDescriptorAllocInfo,
                                    &_wavefrontShadeDescriptorSets[0]));

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &shadeDescriptorAllocInfo,
                                    &_wavefrontShadeDescriptorSets[1]));

  // Dispatch (needs 2 for the raycount buffer switching)
  VkDescriptorSetAllocateInfo dispatchDescriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _wavefrontDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_wavefrontDispatchDescriptorSetLayout };

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &dispatchDescriptorAllocInfo,
      &_wavefrontDispatchDescriptorSets[0]));

  VK_CHECK(vkAllocateDescriptorSets(_engine._device, &dispatchDescriptorAllocInfo,
      &_wavefrontDispatchDescriptorSets[1]));

  // Finalize
  VkDescriptorSetAllocateInfo finalizeDescriptorAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _wavefrontDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_wavefrontFinalizeDescriptorSetLayout};

  VK_CHECK(vkAllocateDescriptorSets(_engine._device,
                                    &finalizeDescriptorAllocInfo,
                                    &_wavefrontFinalizeDescriptorSet));

  // Populate
  // Generate
  VkDescriptorImageInfo descriptorGenerateImageInfo{
      .imageView = _renderTargetView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

  VkWriteDescriptorSet writeGenerateRender{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontGenerateDescriptorSet,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &descriptorGenerateImageInfo};

  VkDescriptorBufferInfo descriptorGenerateRaysBufferInfo{
      _wavefrontRayBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeGenerateRays{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontGenerateDescriptorSet,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorGenerateRaysBufferInfo};

  VkDescriptorBufferInfo descriptorGeneratePathStatesInfo{
      _wavefrontPathStateBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeGeneratePathStates{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontGenerateDescriptorSet,
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorGeneratePathStatesInfo};

  VkWriteDescriptorSet writeGenerateSets[3] = {
      writeGenerateRender, writeGenerateRays, writeGeneratePathStates};
  vkUpdateDescriptorSets(_engine._device, 3, writeGenerateSets, 0, nullptr);

  // Extend 0
  VkDescriptorBufferInfo descriptorExtendRaysBufferInfo0{
      _wavefrontRayBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendRays0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[0],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendRaysBufferInfo0};

  VkDescriptorBufferInfo descriptorExtendHitRecordsBufferInfo0{
      _wavefrontHitRecordBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendHitRecords0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[0],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendHitRecordsBufferInfo0};

  VkDescriptorBufferInfo descriptorExtendSceneBufferInfo0{_sceneBuffer, 0,
                                                          VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendScene0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[0],
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendSceneBufferInfo0};

  VkDescriptorBufferInfo descriptorExtendLastRayCountInfo0
    { _wavefrontNextRayCountBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendLastRayCount0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[0],
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendLastRayCountInfo0 };

  VkWriteDescriptorSet writeExtendSets0[4] = {
      writeExtendRays0, writeExtendHitRecords0, 
      writeExtendScene0, writeExtendLastRayCount0 };
  vkUpdateDescriptorSets(_engine._device, 4, writeExtendSets0, 0, nullptr);

  // Extend 1
  VkDescriptorBufferInfo descriptorExtendRaysBufferInfo1{
      _wavefrontRayBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendRays1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[1],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendRaysBufferInfo1};

  VkDescriptorBufferInfo descriptorExtendHitRecordsBufferInfo1{
      _wavefrontHitRecordBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendHitRecords1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[1],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendHitRecordsBufferInfo1};

  VkDescriptorBufferInfo descriptorExtendSceneBufferInfo1{_sceneBuffer, 0,
                                                          VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeExtendScene1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[1],
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendSceneBufferInfo1};

  VkDescriptorBufferInfo descriptorExtendLastRayCountInfo1
    { _wavefrontNextRayCountBuffers[1], 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeExtendLastRayCount1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontExtendDescriptorSets[1],
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorExtendLastRayCountInfo1 };

  VkWriteDescriptorSet writeExtendSets1[4] = {
      writeExtendRays1, writeExtendHitRecords1, 
      writeExtendScene1, writeExtendLastRayCount1 };
  vkUpdateDescriptorSets(_engine._device, 4, writeExtendSets1, 0, nullptr);

  // Shade 0
  VkDescriptorBufferInfo descriptorShadeRaysInBufferInfo0{
      _wavefrontRayBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeRaysIn0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeRaysInBufferInfo0};

  VkDescriptorBufferInfo descriptorShadePathStatesInBufferInfo0{
      _wavefrontPathStateBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadePathStatesIn0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadePathStatesInBufferInfo0};

  VkDescriptorBufferInfo descriptorShadeHitRecordBufferInfo0{
      _wavefrontHitRecordBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeHitRecord0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeHitRecordBufferInfo0};

  VkDescriptorBufferInfo descriptorShadeLastRayCountBufferInfo0{
      _wavefrontNextRayCountBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeLastRayCount0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeLastRayCountBufferInfo0 };

  VkDescriptorBufferInfo descriptorShadeRaysOutBufferInfo0{
      _wavefrontRayBuffers[1], 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeShadeRaysOut0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 4,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeRaysOutBufferInfo0};

  VkDescriptorBufferInfo descriptorShadePathStatesOutBufferInfo0{
      _wavefrontPathStateBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadePathStatesOut0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 5,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadePathStatesOutBufferInfo0};

  VkDescriptorBufferInfo descriptorShadeFinalRadianceBufferInfo0{
      _wavefrontFinalRadianceBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeFinalRadiance0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 6,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeFinalRadianceBufferInfo0};

  VkDescriptorBufferInfo descriptorShadeNextRayCountBufferInfo0{
      _wavefrontNextRayCountBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeNextRayCount0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[0],
      .dstBinding = 7,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeNextRayCountBufferInfo0};

  VkWriteDescriptorSet writeShadeSets0[8] = {
      writeShadeRaysIn0,        writeShadePathStatesIn0,
      writeShadeHitRecord0,     writeShadeLastRayCount0,
      writeShadeRaysOut0,       writeShadePathStatesOut0,
      writeShadeFinalRadiance0, writeShadeNextRayCount0};
  vkUpdateDescriptorSets(_engine._device, 8, writeShadeSets0, 0, nullptr);

  // Shade 1
  VkDescriptorBufferInfo descriptorShadeRaysInBufferInfo1{
      _wavefrontRayBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeRaysIn1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeRaysInBufferInfo1};

  VkDescriptorBufferInfo descriptorShadePathStatesInBufferInfo1{
      _wavefrontPathStateBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadePathStatesIn1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadePathStatesInBufferInfo1};

  VkDescriptorBufferInfo descriptorShadeHitRecordBufferInfo1{
      _wavefrontHitRecordBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeHitRecord1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeHitRecordBufferInfo1};

  VkDescriptorBufferInfo descriptorShadeLastRayCountBufferInfo1{
      _wavefrontNextRayCountBuffers[1], 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeShadeLastRayCount1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeLastRayCountBufferInfo1 };

  VkDescriptorBufferInfo descriptorShadeRaysOutBufferInfo1{
      _wavefrontRayBuffers[0], 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeShadeRaysOut1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 4,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeRaysOutBufferInfo1};

  VkDescriptorBufferInfo descriptorShadePathStatesOutBufferInfo1{
      _wavefrontPathStateBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadePathStatesOut1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 5,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadePathStatesOutBufferInfo1};

  VkDescriptorBufferInfo descriptorShadeFinalRadianceBufferInfo1{
      _wavefrontFinalRadianceBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeFinalRadiance1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 6,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeFinalRadianceBufferInfo1};

  VkDescriptorBufferInfo descriptorShadeNextRayCountBufferInfo1{
      _wavefrontNextRayCountBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeShadeNextRayCount1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontShadeDescriptorSets[1],
      .dstBinding = 7,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorShadeNextRayCountBufferInfo1};

  VkWriteDescriptorSet writeShadeSets1[8] = {
      writeShadeRaysIn1,        writeShadePathStatesIn1,
      writeShadeHitRecord1,     writeShadeLastRayCount1,
      writeShadeRaysOut1,       writeShadePathStatesOut1, 
      writeShadeFinalRadiance1, writeShadeNextRayCount1};
  vkUpdateDescriptorSets(_engine._device, 8, writeShadeSets1, 0, nullptr);

  // Dispatch 0
  VkDescriptorBufferInfo descriptorDispatchNextRayCountBufferInfo0{
      _wavefrontNextRayCountBuffers[1], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeDispatchNextRayCount0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontDispatchDescriptorSets[0],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorDispatchNextRayCountBufferInfo0 };

  VkDescriptorBufferInfo descriptorDispatchDispatchBufferInfo0{
      _wavefrontDispatchBuffer, 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeDispatchDispatch0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontDispatchDescriptorSets[0],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorDispatchDispatchBufferInfo0 };

  VkWriteDescriptorSet writeDispatchSets0[2] = {
      writeDispatchNextRayCount0, writeDispatchDispatch0 };
  vkUpdateDescriptorSets(_engine._device, 2, writeDispatchSets0, 0, nullptr);

  // Dispatch 1
  VkDescriptorBufferInfo descriptorDispatchNextRayCountBufferInfo1{
      _wavefrontNextRayCountBuffers[0], 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeDispatchNextRayCount1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontDispatchDescriptorSets[1],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorDispatchNextRayCountBufferInfo1 };

  VkDescriptorBufferInfo descriptorDispatchDispatchBufferInfo1{
      _wavefrontDispatchBuffer, 0, VK_WHOLE_SIZE };

  VkWriteDescriptorSet writeDispatchDispatch1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontDispatchDescriptorSets[1],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorDispatchDispatchBufferInfo1 };

  VkWriteDescriptorSet writeDispatchSets1[2] = {
      writeDispatchNextRayCount1, writeDispatchDispatch1 };
  vkUpdateDescriptorSets(_engine._device, 2, writeDispatchSets1, 0, nullptr);

  // Finalize
  VkDescriptorImageInfo descriptorFinalizeImageInfo{
      .imageView = _renderTargetView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

  VkWriteDescriptorSet writeFinalizeRender{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontFinalizeDescriptorSet,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &descriptorFinalizeImageInfo};

  VkDescriptorBufferInfo descriptorFinalizeFinalRadianceBufferInfo{
      _wavefrontFinalRadianceBuffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writeFinalizeFinalRadiance{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _wavefrontFinalizeDescriptorSet,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorFinalizeFinalRadianceBufferInfo};

  VkWriteDescriptorSet writeFinalizeSets[2] = {writeFinalizeRender,
                                               writeFinalizeFinalRadiance};
  vkUpdateDescriptorSets(_engine._device, 2, writeFinalizeSets, 0, nullptr);
}

void Raytracer::createWavefrontPipelines() {
  // Generate
  VkPushConstantRange generatePushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = 64}; // Currently hard-coded 64 Bytes

  VkPipelineLayoutCreateInfo generateLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_wavefrontGenerateDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &generatePushConstantRange};

  VK_CHECK(vkCreatePipelineLayout(_engine._device, &generateLayoutInfo, nullptr,
                                  &_wavefrontGenerateLayout));

  VkShaderModule generateShader =
      load_shader("shaders/wavefront_generate.spv", _engine._device);

  VkPipelineShaderStageCreateInfo generateShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = generateShader,
      .pName = "main"};
  VkComputePipelineCreateInfo generatePipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = generateShaderStageInfo,
      .layout = _wavefrontGenerateLayout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
                                    &generatePipelineInfo, nullptr,
                                    &_wavefrontGeneratePipeline));

  vkDestroyShaderModule(_engine._device, generateShader, nullptr);

  // Extend
  VkPipelineLayoutCreateInfo extendLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_wavefrontExtendDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = VK_NULL_HANDLE};

  VK_CHECK(vkCreatePipelineLayout(_engine._device, &extendLayoutInfo, nullptr,
                                  &_wavefrontExtendLayout));

  VkShaderModule extendShader =
      load_shader("shaders/wavefront_extend.spv", _engine._device);

  VkPipelineShaderStageCreateInfo extendShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = extendShader,
      .pName = "main"};
  VkComputePipelineCreateInfo extendPipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = extendShaderStageInfo,
      .layout = _wavefrontExtendLayout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
                                    &extendPipelineInfo, nullptr,
                                    &_wavefrontExtendPipeline));

  vkDestroyShaderModule(_engine._device, extendShader, nullptr);

  // Shade
  VkPipelineLayoutCreateInfo shadeLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_wavefrontShadeDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = VK_NULL_HANDLE};

  VK_CHECK(vkCreatePipelineLayout(_engine._device, &shadeLayoutInfo, nullptr,
                                  &_wavefrontShadeLayout));

  VkShaderModule shadeShader =
      load_shader("shaders/wavefront_shade.spv", _engine._device);

  VkPipelineShaderStageCreateInfo shadeShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shadeShader,
      .pName = "main"};
  VkComputePipelineCreateInfo shadePipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = shadeShaderStageInfo,
      .layout = _wavefrontShadeLayout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
                                    &shadePipelineInfo, nullptr,
                                    &_wavefrontShadePipeline));

  vkDestroyShaderModule(_engine._device, shadeShader, nullptr);

  // Dispatch
  VkPipelineLayoutCreateInfo dispatchLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_wavefrontDispatchDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = VK_NULL_HANDLE };

  VK_CHECK(vkCreatePipelineLayout(_engine._device, &dispatchLayoutInfo, nullptr,
      &_wavefrontDispatchLayout));

  VkShaderModule dispatchShader =
      load_shader("shaders/wavefront_dispatch.spv", _engine._device);

  VkPipelineShaderStageCreateInfo dispatchShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = dispatchShader,
      .pName = "main" };
  VkComputePipelineCreateInfo dispatchPipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = dispatchShaderStageInfo,
      .layout = _wavefrontDispatchLayout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
      &dispatchPipelineInfo, nullptr,
      &_wavefrontDispatchPipeline));

  vkDestroyShaderModule(_engine._device, dispatchShader, nullptr);

  // Finalize
  VkPipelineLayoutCreateInfo finalizeLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_wavefrontFinalizeDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = VK_NULL_HANDLE};

  VK_CHECK(vkCreatePipelineLayout(_engine._device, &finalizeLayoutInfo, nullptr,
                                  &_wavefrontFinalizeLayout));

  VkShaderModule finalizeShader =
      load_shader("shaders/wavefront_finalize.spv", _engine._device);

  VkPipelineShaderStageCreateInfo finalizeShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = finalizeShader,
      .pName = "main"};
  VkComputePipelineCreateInfo finalizePipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = finalizeShaderStageInfo,
      .layout = _wavefrontFinalizeLayout,
  };

  VK_CHECK(vkCreateComputePipelines(_engine._device, VK_NULL_HANDLE, 1,
                                    &finalizePipelineInfo, nullptr,
                                    &_wavefrontFinalizePipeline));

  vkDestroyShaderModule(_engine._device, finalizeShader, nullptr);
}

void Raytracer::createWavefrontMemoryBarrier() {
  _wavefrontMemoryBarrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask =
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT | 
                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
      .dstAccessMask =
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
          VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT };

  _wavefrontMemoryDependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .memoryBarrierCount = 1,
                                .pMemoryBarriers = &_wavefrontMemoryBarrier};
}

void Raytracer::recordWavefrontBuffer(uint32_t image_index) {
  VkCommandBuffer computeCmd =
      _engine._computeCommandBuffers[_engine.frame_index];
  VkCommandBuffer graphicsCmd = _engine._commandBuffers[_engine.frame_index];

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  VK_CHECK(vkBeginCommandBuffer(computeCmd, &beginInfo));

  vkCmdResetQueryPool(computeCmd, _timestampPool, 0, 2);

  vkCmdWriteTimestamp2(computeCmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                       _timestampPool, 0);

  // Generate
  vkCmdPushConstants(computeCmd, _wavefrontGenerateLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 64,
                     _cameraConstants); // TODO ALSO HARDCODE 64 bytes here

  _engine.transition_image_layout(
      computeCmd, _renderTarget, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL, {}, VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

  vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _wavefrontGeneratePipeline);

  vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _wavefrontGenerateLayout, 0, 1,
                          &_wavefrontGenerateDescriptorSet, 0, nullptr);

  vkCmdDispatch(computeCmd, (_engine._windowExtent.width + 15) / 16,
                (_engine._windowExtent.height + 15) / 16, 1);

  vkCmdFillBuffer(computeCmd, _wavefrontFinalRadianceBuffer, 0, VK_WHOLE_SIZE,
                  0);

  // Fill correct value into dispatch buffer for first extend
  vkCmdFillBuffer(computeCmd, _wavefrontNextRayCountBuffers[0], 0,
                  VK_WHOLE_SIZE,
                  _engine._windowExtent.width * _engine._windowExtent.height *
                      1); // Samples = 1

  vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

  vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _wavefrontDispatchPipeline);

  vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _wavefrontDispatchLayout, 0, 1,
                          &_wavefrontDispatchDescriptorSets[1], 0, nullptr);

  vkCmdDispatch(computeCmd, 1, 1, 1);

  // Extend and Shade loop
  // MAX BOUNCES = 20
  for (int i = 0; i < 20; i += 2) {
    // Memory barrier either between generate or shade and extend
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Extend 1
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontExtendPipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontExtendLayout, 0, 1,
                            &_wavefrontExtendDescriptorSets[0], 0, nullptr);

    vkCmdDispatchIndirect(computeCmd, _wavefrontDispatchBuffer, 0);

    // Reset ray count
    vkCmdFillBuffer(computeCmd, _wavefrontNextRayCountBuffers[1], 0,
                    VK_WHOLE_SIZE, 0);

    // Memory barrier between extend 1 and shade 1
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Shade 1
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontShadePipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontShadeLayout, 0, 1,
                            &_wavefrontShadeDescriptorSets[0], 0, nullptr);

    vkCmdDispatchIndirect(computeCmd, _wavefrontDispatchBuffer, 0);

    // Memory barrier between shade 1 and dispatch 1
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Dispatch 1
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontDispatchPipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontDispatchLayout, 0, 1,
                            &_wavefrontDispatchDescriptorSets[0], 0, nullptr);

    vkCmdDispatch(computeCmd, 1, 1, 1);

    // Memory barrier between dispatch 1 and extend 2
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Extend 2
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontExtendPipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontExtendLayout, 0, 1,
                            &_wavefrontExtendDescriptorSets[1], 0, nullptr);

    vkCmdDispatchIndirect(computeCmd, _wavefrontDispatchBuffer, 0);

    // Reset ray count
    vkCmdFillBuffer(computeCmd, _wavefrontNextRayCountBuffers[0], 0,
                    VK_WHOLE_SIZE, 0);

    // Memory barrier between extend 1 and shade 1
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Shade 2
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontShadePipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontShadeLayout, 0, 1,
                            &_wavefrontShadeDescriptorSets[1], 0, nullptr);

    vkCmdDispatchIndirect(computeCmd, _wavefrontDispatchBuffer, 0);

    // Memory barrier between shade 2 and dispatch 2
    vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

    // Dispatch 2
    vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _wavefrontDispatchPipeline);

    vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _wavefrontDispatchLayout, 0, 1,
                            &_wavefrontDispatchDescriptorSets[1], 0, nullptr);

    vkCmdDispatch(computeCmd, 1, 1, 1);
  }

  // Memory barrier between shade and finalize
  vkCmdPipelineBarrier2(computeCmd, &_wavefrontMemoryDependency);

  // Finalize
  _engine.transition_image_layout(
      computeCmd, _renderTarget, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

  vkCmdBindPipeline(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _wavefrontFinalizePipeline);

  vkCmdBindDescriptorSets(computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _wavefrontFinalizeLayout, 0, 1,
                          &_wavefrontFinalizeDescriptorSet, 0, nullptr);

  vkCmdDispatch(computeCmd, (_engine._windowExtent.width + 15) / 16,
                (_engine._windowExtent.height + 15) / 16, 1);

  vkCmdWriteTimestamp2(computeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       _timestampPool, 1);

  VK_CHECK(vkEndCommandBuffer(computeCmd));

  VK_CHECK(vkBeginCommandBuffer(graphicsCmd, &beginInfo));

  // Rest
  _engine.transition_image_layout(
      graphicsCmd, _renderTarget, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, {}, VK_ACCESS_2_TRANSFER_READ_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  _engine.transition_image_layout(
      graphicsCmd, _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {},
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  VkImageBlit blitRegion{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0},
                     {(int32_t)_engine._windowExtent.width,
                      (int32_t)_engine._windowExtent.height, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0},
                     {(int32_t)_engine._swapchainExtent.width,
                      (int32_t)_engine._swapchainExtent.height, 1}}};
  vkCmdBlitImage(graphicsCmd, _renderTarget,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 _engine._swapchainImages[image_index],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion,
                 VK_FILTER_LINEAR);

  _engine.transition_image_layout(
      graphicsCmd, _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, {}, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  VK_CHECK(vkEndCommandBuffer(graphicsCmd));
}
