#include "bvh.h"
#include "metrics_macros.h"
#include "vulkan_engine.h"
#include "vulkan_utils.h"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

Bvh::Bvh(VulkanEngine &engine, size_t primitiveCount)
    : _engine(engine), _primitiveCount(primitiveCount) {}

void Bvh::init(VkBuffer &primBuffer) {
  groups = (_primitiveCount + 1023) / 1024;
  histogramElems = 16 * groups * 256;
  numBlocks = histogramElems / 512;

  createBvhBuffers();
  createDescriptor(primBuffer);
  createPipelines();
}

void Bvh::build() {
  constexpr uint32_t maxDispatches = 3 + 8 * 5 + 3;
  VkQueryPoolCreateInfo queryInfo{.sType =
                                      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = maxDispatches * 2};
  VkQueryPool timestampPool;
  VK_CHECK(vkCreateQueryPool(DEVICE, &queryInfo, nullptr, &timestampPool));

  std::vector<uint32_t> kernelOfDispatch;
  uint32_t queryCount = 0;

  VkCommandPool buildPool;
  VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = _engine._computeQueueFamilyIndex};
  VK_CHECK(vkCreateCommandPool(DEVICE, &poolInfo, nullptr, &buildPool));

  // Command Buffer
  VkCommandBuffer buildCommandBuffer;
  VkCommandBufferAllocateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = buildPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VK_CHECK(vkAllocateCommandBuffers(DEVICE, &bufferInfo, &buildCommandBuffer));

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

  VK_CHECK(vkBeginCommandBuffer(buildCommandBuffer, &beginInfo));

  vkCmdResetQueryPool(buildCommandBuffer, timestampPool, 0, maxDispatches * 2);

  struct PushConstants {
    int32_t primitiveCount;
    int32_t pass_idx;
    int32_t num_blocks;
    int32_t histogram_size;
  } pc{(int32_t)_primitiveCount, -1, (int32_t)numBlocks,
       (int32_t)histogramElems};

  auto dispatch = [&](uint32_t pipelineIdx, VkDescriptorSet set,
                      uint32_t groupCount) {
    vkCmdBindPipeline(buildCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      _bvhPipelines[pipelineIdx]);
    vkCmdBindDescriptorSets(buildCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _bvhPipelineLayout, 0, 1, &set, 0, nullptr);

    vkCmdWriteTimestamp2(buildCommandBuffer,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, timestampPool,
                         queryCount);
    vkCmdDispatch(buildCommandBuffer, groupCount, 1, 1);
    vkCmdWriteTimestamp2(buildCommandBuffer,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool,
                         queryCount + 1);
    kernelOfDispatch.push_back(pipelineIdx);
    queryCount += 2;

    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
    VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                         .memoryBarrierCount = 1,
                         .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(buildCommandBuffer, &dep);
  };

  uint32_t primGroups = (_primitiveCount + 255) / 256;
  uint32_t internalGroups = (_primitiveCount - 1 + 255) / 256;

  vkCmdPushConstants(buildCommandBuffer, _bvhPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

  dispatch(0, _bvhDescriptors[0], primGroups); // init_prim_bboxes
  dispatch(1, _bvhDescriptors[0], 1);          // create_world_bbox
  dispatch(2, _bvhDescriptors[0], primGroups); // compute_morton_codes

  for (int pass = 0; pass < 8; pass++) {
    pc.pass_idx = pass;
    vkCmdPushConstants(buildCommandBuffer, _bvhPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkDescriptorSet set = _bvhDescriptors[pass % 2];
    dispatch(3, set, (uint32_t)groups);    // create_histogram
    dispatch(4, set, (uint32_t)numBlocks); // prefix_sum
    dispatch(5, set, 1);                   // scan_global_sum
    dispatch(6, set, (uint32_t)numBlocks); // add_global_sums
    dispatch(7, set, (uint32_t)groups);    // scatter
  }

  dispatch(8, _bvhDescriptors[0], primGroups);     // init_prim_nodes
  dispatch(9, _bvhDescriptors[0], internalGroups); // create_bvh_hierarchy
  dispatch(10, _bvhDescriptors[0], primGroups);    // build_bboxes

  VK_CHECK(vkEndCommandBuffer(buildCommandBuffer));

  VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(DEVICE, &fenceInfo, nullptr, &fence));

  VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1,
                      .pCommandBuffers = &buildCommandBuffer};
  VK_CHECK(vkQueueSubmit(_engine._computeQueue, 1, &submit, fence));
  VK_CHECK(vkWaitForFences(DEVICE, 1, &fence, VK_TRUE, UINT64_MAX));

#ifdef EVALUATE
  std::vector<uint64_t> timestamps(queryCount);
  VK_CHECK(vkGetQueryPoolResults(
      DEVICE, timestampPool, 0, queryCount,
      timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

  float perKernel[BVH_KERNELS] = {};
  for (uint32_t d = 0; d < kernelOfDispatch.size(); d++) {
    float ms = static_cast<float>(timestamps[2 * d + 1] - timestamps[2 * d]) *
               _engine._timestampPeriod * 1e-6f;
    perKernel[kernelOfDispatch[d]] += ms;
  }
  float bboxSetup = perKernel[0] + perKernel[1];
  float morton = perKernel[2];
  float radix =
      perKernel[3] + perKernel[4] + perKernel[5] + perKernel[6] + perKernel[7];
  float hierarchy = perKernel[8] + perKernel[9];
  float bbox = perKernel[10];

  METRIC_SET_VALUE("Primitive+World BBox Kernel", bboxSetup);
  METRIC_SET_VALUE("Morton Code Kernel", morton);
  METRIC_SET_VALUE("Radix Sort Kernel", radix);
  METRIC_SET_VALUE("Hierarchy Creation Kernel", hierarchy);
  METRIC_SET_VALUE("Bounding Box Kernel", bbox);
  METRIC_SET_VALUE("Total BVH Kernel Time",
                   bboxSetup + morton + radix + hierarchy + bbox);
#endif

  vkDestroyQueryPool(DEVICE, timestampPool, nullptr);
  vkDestroyFence(DEVICE, fence, nullptr);
  vkDestroyCommandPool(DEVICE, buildPool, nullptr);
}

void Bvh::createBvhBuffers() { // XXX: Currently doing 1:1 buffer/deviceMemory.
                               // Change later if needed

  // BvhBuffer: 2N-1 nodes (N leaves + N-1 internal)
  bvhBuffer.size = (2 * _primitiveCount - 1) * NODE_STRUCT_BYTES;
  createStorageBuffer(DEVICE, bvhBuffer.buffer, bvhBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bvhBuffer.memory);

  // primBboxBuffer: N aabbs (2 vec4 = 32 Bytes)
  primBboxBuffer.size = _primitiveCount * 32;
  createStorageBuffer(DEVICE, primBboxBuffer.buffer, primBboxBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      primBboxBuffer.memory);

  // mortonCodesBuffer: N uints
  mortonCodesBuffer.size = _primitiveCount * 4;
  createStorageBuffer(DEVICE, mortonCodesBuffer.buffer, mortonCodesBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      mortonCodesBuffer.memory);

  // primIndicesBuffer: N uints
  primIndicesBuffer.size = _primitiveCount * 4;
  createStorageBuffer(DEVICE, primIndicesBuffer.buffer, primIndicesBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      primIndicesBuffer.memory);

  // worldBboxBuffer: 1 aabb (2 vec4 = 32 Bytes)
  worldBboxBuffer.size = 32;
  createStorageBuffer(DEVICE, worldBboxBuffer.buffer, worldBboxBuffer.size,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      worldBboxBuffer.memory);

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
  outputMortonCodeBuffer.size = _primitiveCount * 4;
  createStorageBuffer(
      DEVICE, outputMortonCodeBuffer.buffer, outputMortonCodeBuffer.size,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outputMortonCodeBuffer.memory);

  // outputPrimIndicesBuffer: N uints
  outputPrimIndicesBuffer.size = _primitiveCount * 4;
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
    poolSizes.push_back(
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 2}); // 2 sets share the pool
  }

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = bindingCount,
      .pPoolSizes = poolSizes.data()};
  VK_CHECK(
      vkCreateDescriptorPool(DEVICE, &poolInfo, nullptr, &_bvhDescriptorPool));

  VkDescriptorSetLayout layouts[2] = {_bvhDescriptorLayout,
                                      _bvhDescriptorLayout};
  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _bvhDescriptorPool,
      .descriptorSetCount = 2,
      .pSetLayouts = layouts};

  VK_CHECK(vkAllocateDescriptorSets(DEVICE, &allocInfo, _bvhDescriptors));

  std::vector<VkDescriptorBufferInfo> bufferInfos(
      bindingCount,
      VkDescriptorBufferInfo{.offset = 0, .range = VK_WHOLE_SIZE});

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

  for (uint32_t set = 0; set < 2; set++) {
    if (set == 1) {
      std::swap(bufferInfos[3].buffer, bufferInfos[9].buffer);
      std::swap(bufferInfos[4].buffer, bufferInfos[10].buffer);
    }

    std::vector<VkWriteDescriptorSet> writeSets;
    for (uint32_t i = 0; i < bindingCount; i++) {
      writeSets.push_back(VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _bvhDescriptors[set],
          .dstBinding = i,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufferInfos[i]});
    }
    vkUpdateDescriptorSets(DEVICE, bindingCount, writeSets.data(), 0, nullptr);
  }
}

void Bvh::createPipelines() {
  _bvhPipelines.resize(BVH_KERNELS);

  VkPushConstantRange pcRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                              .offset = 0,
                              .size = 16}; // 4 x int32 push block

  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_bvhDescriptorLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcRange};

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
