#include "bvh.h"
#include "vulkan_engine.h"
#include "vulkan_utils.h"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>

Bvh::Bvh(VulkanEngine &engine) : _engine(engine) {}

void Bvh::init(size_t primitiveCount, VkBuffer &primBuffer) {
  createBvhBuffers(primitiveCount);
  createDescriptor(primBuffer);
  createPipelines();
}

void Bvh::build() {}

void Bvh::createBvhBuffers(
    size_t primitiveCount) { // XXX: Currently doing 1:1 buffer/deviceMemory.
                             // Change later if needed

  // BvhBuffer: 2N-1 nodes (N leaves + N-1 internal)
  bvhBuffer.size = (2 * primitiveCount - 1) * NODE_STRUCT_BYTES;
  createStorageBuffer(DEVICE, bvhBuffer.buffer, bvhBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bvhBuffer.memory);

  // primBboxBuffer: N aabbs (2 vec4 = 32 Bytes)
  primBboxBuffer.size = primitiveCount * 32;
  createStorageBuffer(DEVICE, primBboxBuffer.buffer, primBboxBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      primBboxBuffer.memory);

  // mortonCodesBuffer: N uints
  mortonCodesBuffer.size = primitiveCount * 4;
  createStorageBuffer(DEVICE, mortonCodesBuffer.buffer, mortonCodesBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      mortonCodesBuffer.memory);

  // primIndicesBuffer: N uints
  primIndicesBuffer.size = primitiveCount * 4;
  createStorageBuffer(DEVICE, primIndicesBuffer.buffer, primIndicesBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      primIndicesBuffer.memory);

  // worldBboxBuffer: 1 aabb (2 vec4 = 32 Bytes)
  worldBboxBuffer.size = 32;
  createStorageBuffer(DEVICE, worldBboxBuffer.buffer, worldBboxBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      worldBboxBuffer.memory);

  size_t groups = (primitiveCount + 1023) /
                  1024; // create_histogram groups (1024 elems/group)
  size_t histogramElems = 16 * groups * 256; // 16 digits * groups * 256 threads
  size_t numBlocks =
      histogramElems / 512; // prefix_sum blocks (512 elems each) = 8*groups
  // histogramBuffer: 16 * groups * 256 uints
  histogramBuffer.size = histogramElems * 4;
  createStorageBuffer(DEVICE, histogramBuffer.buffer, histogramBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      histogramBuffer.memory);

  // scannedGramBuffer: same size as histogram
  scannedGramBuffer.size = histogramElems * 4;
  createStorageBuffer(DEVICE, scannedGramBuffer.buffer, scannedGramBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      scannedGramBuffer.memory);

  // globSumBuffer: numBlocks uints
  globSumBuffer.size = numBlocks * 4;
  createStorageBuffer(DEVICE, globSumBuffer.buffer, globSumBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      globSumBuffer.memory);

  // outputMortonCodeBuffer: N uints
  outputMortonCodeBuffer.size = primitiveCount * 4;
  createStorageBuffer(
      DEVICE, outputMortonCodeBuffer.buffer, outputMortonCodeBuffer.size,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outputMortonCodeBuffer.memory);

  // outputPrimIndicesBuffer: N uints
  outputPrimIndicesBuffer.size = primitiveCount * 4;
  createStorageBuffer(
      DEVICE, outputPrimIndicesBuffer.buffer, outputPrimIndicesBuffer.size,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outputPrimIndicesBuffer.memory);
}

void Bvh::createDescriptor(VkBuffer &primBuffer) {

  uint32_t bindingCount = 11; // XXX : BUFFER COUNT HARDCODED TO 11

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (uint32_t i = 0; i < bindingCount; i++) {
    bindings.push_back(VkDescriptorSetLayoutBinding{
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
  }

  VkDescriptorSetLayoutCreateInfo setLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = bindingCount,
      .pBindings = bindings.data()};

  VK_CHECK(vkCreateDescriptorSetLayout(DEVICE, &setLayoutInfo, nullptr,
                                       &_bvhDescriptorLayout));

  std::vector<VkDescriptorPoolSize> poolSizes;

  for (uint32_t i = 0; i < bindingCount; i++) {
    poolSizes.push_back(VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1});
  }

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = bindingCount,
      .pPoolSizes = poolSizes.data()};
  VK_CHECK(
      vkCreateDescriptorPool(DEVICE, &poolInfo, nullptr, &_bvhDescriptorPool));

  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _bvhDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &_bvhDescriptorLayout};

  VK_CHECK(vkAllocateDescriptorSets(DEVICE, &allocInfo, &_bvhDescriptor));

  std::vector<VkDescriptorBufferInfo> bufferInfos;
  std::vector<VkWriteDescriptorSet> writeSets;

  for (uint32_t i = 0; i < bindingCount; i++) {
    bufferInfos.push_back(
        VkDescriptorBufferInfo{.offset = 0, .range = VK_WHOLE_SIZE});
  }

  bufferInfos[0].buffer = primBuffer;
  bufferInfos[1].buffer = bvhBuffer.buffer;
  bufferInfos[2].buffer = primBboxBuffer.buffer;
  bufferInfos[3].buffer = mortonCodesBuffer.buffer;
  bufferInfos[4].buffer = primIndicesBuffer.buffer;
  bufferInfos[5].buffer = worldBboxBuffer.buffer;
  bufferInfos[6].buffer = histogramBuffer.buffer;
  bufferInfos[7].buffer = scannedGramBuffer.buffer;
  bufferInfos[8].buffer = globSumBuffer.buffer;
  bufferInfos[9].buffer = outputMortonCodeBuffer.buffer;
  bufferInfos[10].buffer = outputPrimIndicesBuffer.buffer;

  for (uint32_t i = 0; i < bindingCount; i++) {
    writeSets.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _bvhDescriptor,
        .dstBinding = i,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos[i]});
  }

  vkUpdateDescriptorSets(DEVICE, bindingCount, writeSets.data(), 0, nullptr);
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
      "init_prim_bboxes.spv",     "create_world_bbox.spv",
      "compute_morton_codes.spv", "create_histogram.spv",
      "prefix_sum.spv",           "scan_global_sum.spv",
      "add_global_sums.spv",      "scatter.spv",
      "init_prim_nodes.spv",      "create_bvh_hierarchy.spv",
      "build_bboxes.spv"};

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
