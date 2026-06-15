#include "bvh.h"
#include "vulkan_engine.h"
#include "vulkan_types.h"
#include "vulkan_utils.h"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>

Bvh::Bvh(VulkanEngine &engine) : _engine(engine) {}

void Bvh::init(size_t N, VkBuffer &primBuffer) {
  createBvhBuffer(N);
  createDescriptor(primBuffer);
  createPipelines();
}

void Bvh::build() {}

void Bvh::createBvhBuffer(size_t N) {
  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = (2 * N - 1) * BVH_BYTES,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VK_CHECK(vkCreateBuffer(DEVICE, &bufferInfo, nullptr, &_bvhBuffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(DEVICE, _bvhBuffer, &memRequirements);

  uint32_t mem_index = find_memory_type(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        memRequirements.memoryTypeBits);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = memRequirements.size,
                                 .memoryTypeIndex = mem_index};

  VK_CHECK(vkAllocateMemory(DEVICE, &allocInfo, nullptr, &_bvhMemory));
  VK_CHECK(vkBindBufferMemory(DEVICE, _bvhBuffer, _bvhMemory, 0));
}

void Bvh::createDescriptor(VkBuffer &primBuffer) {

  VkDescriptorSetLayoutBinding primitiveBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutBinding bvhBinding{
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

  VkDescriptorSetLayoutBinding bindings[2] = {primitiveBinding, bvhBinding};

  VkDescriptorSetLayoutCreateInfo setLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings};

  VK_CHECK(vkCreateDescriptorSetLayout(DEVICE, &setLayoutInfo, nullptr,
                                       &_bvhDescriptorLayout));

  VkDescriptorPoolSize primPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .descriptorCount = 1};
  VkDescriptorPoolSize bvhPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   .descriptorCount = 1};
  VkDescriptorPoolSize poolSizes[2] = {primPoolSize, bvhPoolSize};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = poolSizes};
  VK_CHECK(
      vkCreateDescriptorPool(DEVICE, &poolInfo, nullptr, &_bvhDescriptorPool));

  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _bvhDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_bvhDescriptorLayout};

  VK_CHECK(vkAllocateDescriptorSets(DEVICE, &allocInfo, &_bvhDescriptor));

  VkDescriptorBufferInfo primBufferInfo{
      .buffer = primBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet primWriteSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _bvhDescriptor,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &primBufferInfo};

  VkDescriptorBufferInfo bvhBufferInfo{
      .buffer = _bvhBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet bvhWriteSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _bvhDescriptor,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bvhBufferInfo};

  VkWriteDescriptorSet writeSets[2] = {primWriteSet, bvhWriteSet};
  vkUpdateDescriptorSets(DEVICE, 2, writeSets, 0, nullptr);
}
void Bvh::createPipelines() {
  _bvhPipelines.resize(BVH_KERNELS);

  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_bvhDescriptorLayout};

  VK_CHECK(vkCreatePipelineLayout(DEVICE, &layoutInfo, nullptr,
                                  &_bvhPipelineLayout));

  std::vector<std::string> names = {
      "1", "2", "3", "4", "5"}; // TODO MAKE SURE THESE ARE CORRECT FILE NAMES

  std::vector<VkComputePipelineCreateInfo> createInfos;
  std::vector<VkShaderModule> shaderModules;
  for (uint32_t idx = 0; idx < BVH_KERNELS; idx++) {

    std::string name = names[idx];
    shaderModules.push_back(load_shader("shaders/" + name, DEVICE));

    VkPipelineShaderStageCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shaderModules[idx],
        .pName = "main"};

    createInfos.push_back(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shaderInfo,
        .layout = _bvhPipelineLayout});
  }

  VK_CHECK(vkCreateComputePipelines(DEVICE, VK_NULL_HANDLE, BVH_KERNELS,
                                    createInfos.data(), nullptr,
                                    _bvhPipelines.data()));

  for (uint32_t idx = 0; idx < BVH_KERNELS; idx++) {
    vkDestroyShaderModule(DEVICE, shaderModules[idx], nullptr);
  }
}
