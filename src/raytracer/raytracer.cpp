#include "raytracer.h"
#include "vulkan_engine.h"
#include "vulkan_types.h"
#include "vulkan_utils.h"
#include <cstdint>

#include <vulkan/vulkan_core.h>

Raytracer::Raytracer(VulkanEngine &engine) : _engine(engine) {}

void Raytracer::initRaytracer(const void *data, size_t size,
                              const void *cameraData) {
  _cameraConstants = cameraData;

  createRenderTarget();
  createSceneBuffer(data, size);
  createDescriptors();
  createPipeline();
}

void Raytracer::createRenderTarget() {
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
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
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

void Raytracer::createSceneBuffer(const void *data, size_t size) {
  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VK_CHECK(
      vkCreateBuffer(_engine._device, &bufferInfo, nullptr, &_sceneBuffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(_engine._device, _sceneBuffer,
                                &memRequirements);

  uint32_t mem_index =
      find_memory_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       memRequirements.memoryTypeBits);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = memRequirements.size,
                                 .memoryTypeIndex = mem_index};
  VK_CHECK(
      vkAllocateMemory(_engine._device, &allocInfo, nullptr, &_sceneMemory));
  VK_CHECK(vkBindBufferMemory(_engine._device, _sceneBuffer, _sceneMemory, 0));

  void *mapped;
  VK_CHECK(vkMapMemory(_engine._device, _sceneMemory, 0, size, {}, &mapped));
  memcpy(mapped, data, size);
  vkUnmapMemory(_engine._device, _sceneMemory);
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

  VkDescriptorSetLayoutBinding layoutBindings[2] = {renderBinding,
                                                    sceneBinding};

  VkDescriptorSetLayoutCreateInfo setInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = layoutBindings};
  VK_CHECK(vkCreateDescriptorSetLayout(_engine._device, &setInfo, nullptr,
                                       &_setLayout));

  // Pool
  VkDescriptorPoolSize renderPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .descriptorCount = 1};
  VkDescriptorPoolSize scenePoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     .descriptorCount = 1};

  VkDescriptorPoolSize poolSizes[2] = {renderPoolSize, scenePoolSize};
  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
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

  VkWriteDescriptorSet writeSets[2] = {writeRender, writeScene};
  vkUpdateDescriptorSets(_engine._device, 2, writeSets, 0, nullptr);
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
  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  VK_CHECK(vkBeginCommandBuffer(_engine._commandBuffers[_engine.frame_index],
                                &beginInfo));

  vkCmdPushConstants(_engine._commandBuffers[_engine.frame_index], _layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 64,
                     _cameraConstants); // TODO ALSO HARDCODE 64 bytes here

  _engine.transition_image_layout(
      _renderTarget, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, {},
      VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  vkCmdBindPipeline(_engine._commandBuffers[_engine.frame_index],
                    VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

  vkCmdBindDescriptorSets(_engine._commandBuffers[_engine.frame_index],
                          VK_PIPELINE_BIND_POINT_COMPUTE, _layout, 0, 1,
                          &_descriptorSet, 0, nullptr);

  vkCmdDispatch(_engine._commandBuffers[_engine.frame_index],
                (_engine._windowExtent.width + 15) / 16,
                (_engine._windowExtent.height + 15) / 16, 1);

  _engine.transition_image_layout(
      _renderTarget, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  _engine.transition_image_layout(
      _engine._swapchainImages[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {}, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  VkImageBlit blitRegion{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0},
                     {(int32_t)_engine._windowExtent.width,
                      (int32_t)_engine._windowExtent.height, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0},
                     {(int32_t)_engine._swapchainExtent.width,
                      (int32_t)_engine._swapchainExtent.height, 1}}};
  vkCmdBlitImage(_engine._commandBuffers[_engine.frame_index], _renderTarget,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 _engine._swapchainImages[image_index],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion,
                 VK_FILTER_LINEAR);

  _engine.transition_image_layout(
      _engine._swapchainImages[image_index],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, {}, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  VK_CHECK(vkEndCommandBuffer(_engine._commandBuffers[_engine.frame_index]));
}
