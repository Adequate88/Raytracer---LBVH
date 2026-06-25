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

void Bvh::init(const void *data, size_t size) {
  groups = (_primitiveCount + 1023) / 1024;
  histogramElems = 16 * groups * 256;
  numBlocks = histogramElems / 512;

  createSceneBuffer(data, size);
  createBvhBuffers();
  createDescriptor();
  createPipelines();
}

void Bvh::build() {
  constexpr uint32_t maxDispatches = 3 + 8 * 5 + 4;
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

  auto pfnBeginLabel = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      _engine._instance, "vkCmdBeginDebugUtilsLabelEXT");
  auto pfnEndLabel = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      _engine._instance, "vkCmdEndDebugUtilsLabelEXT");

  auto dispatch = [&](uint32_t pipelineIdx, VkDescriptorSet set,
                      uint32_t groupCount, const char *name) {
    VkDebugUtilsLabelEXT label{.sType =
                                   VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                               .pLabelName = name};
    if (pfnBeginLabel)
      pfnBeginLabel(buildCommandBuffer, &label);
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
    if (pfnEndLabel)
      pfnEndLabel(buildCommandBuffer);
  };

  uint32_t primGroups = (_primitiveCount + 255) / 256;
  uint32_t internalGroups = (_primitiveCount - 1 + 255) / 256;

  vkCmdPushConstants(buildCommandBuffer, _bvhPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

  dispatch(0, _bvhDescriptors[0], primGroups, "init_prim_bboxes");
  dispatch(1, _bvhDescriptors[0], 1, "create_world_bbox");
  dispatch(2, _bvhDescriptors[0], primGroups, "compute_morton_codes");

  for (int pass = 0; pass < 8; pass++) {
    pc.pass_idx = pass;
    vkCmdPushConstants(buildCommandBuffer, _bvhPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkDescriptorSet set = _bvhDescriptors[pass % 2];
    dispatch(3, set, (uint32_t)groups, "create_histogram");
    dispatch(4, set, (uint32_t)numBlocks, "prefix_sum");
    dispatch(5, set, 1, "scan_global_sum");
    dispatch(6, set, (uint32_t)numBlocks, "add_global_sums");
    dispatch(7, set, (uint32_t)groups, "scatter");
  }

  dispatch(8, _bvhDescriptors[0], primGroups, "init_prim_nodes");

  dispatch(11, _bvhDescriptors[0], primGroups, "reorder_leaf_bboxes");

  VkMemoryBarrier2 toCopy{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
  VkDependencyInfo depToCopy{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                             .memoryBarrierCount = 1,
                             .pMemoryBarriers = &toCopy};
  vkCmdPipelineBarrier2(buildCommandBuffer, &depToCopy);

  size_t leafOffset = (_primitiveCount - 1) * 4;
  size_t leafBytes = _primitiveCount * 4;
  for (int i = 0; i < 6; i++) {
    VkBufferCopy region{
        .srcOffset = leafOffset, .dstOffset = leafOffset, .size = leafBytes};
    vkCmdCopyBuffer(buildCommandBuffer, cornerOutBuffers[i].buffer,
                    cornerBuffers[i].buffer, 1, &region);
  }

  VkMemoryBarrier2 fromCopy{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
  VkDependencyInfo depFromCopy{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .memoryBarrierCount = 1,
                               .pMemoryBarriers = &fromCopy};
  vkCmdPipelineBarrier2(buildCommandBuffer, &depFromCopy);

  dispatch(9, _bvhDescriptors[0], internalGroups, "create_bvh_hierarchy");
  dispatch(10, _bvhDescriptors[0], primGroups, "build_bboxes");

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
  float bbox = perKernel[10] + perKernel[11]; // build_bboxes + reorder

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

void Bvh::createSceneBuffer(const void *data, size_t size) {
  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VK_CHECK(vkCreateBuffer(_engine._device, &bufferInfo, nullptr,
                          &sceneBuffer.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(_engine._device, sceneBuffer.buffer,
                                &memRequirements);

  uint32_t mem_index =
      find_memory_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       memRequirements.memoryTypeBits);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = memRequirements.size,
                                 .memoryTypeIndex = mem_index};
  VK_CHECK(vkAllocateMemory(_engine._device, &allocInfo, nullptr,
                            &sceneBuffer.memory));
  VK_CHECK(vkBindBufferMemory(_engine._device, sceneBuffer.buffer,
                              sceneBuffer.memory, 0));

  void *mapped;
  VK_CHECK(
      vkMapMemory(_engine._device, sceneBuffer.memory, 0, size, {}, &mapped));
  memcpy(mapped, data, size);
  vkUnmapMemory(_engine._device, sceneBuffer.memory);
}

void Bvh::createBvhBuffers() { // XXX: Currently doing 1:1 buffer/deviceMemory.
                               // Change later if needed

  size_t nodeArrayBytes = (2 * _primitiveCount - 1) * 4;

  for (int i = 0; i < 6; i++) {
    cornerBuffers[i].size = nodeArrayBytes;
    createStorageBuffer(DEVICE, cornerBuffers[i].buffer, cornerBuffers[i].size,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        cornerBuffers[i].memory,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  }

  for (int i = 0; i < 6; i++) {
    cornerOutBuffers[i].size = nodeArrayBytes;
    createStorageBuffer(DEVICE, cornerOutBuffers[i].buffer,
                        cornerOutBuffers[i].size,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        cornerOutBuffers[i].memory,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  }

  bufferMemory *intArrays[4] = {&leftChildsBuffer, &rightChildsBuffer,
                                &parentsBuffer, &atomicCountersBuffer};
  for (bufferMemory *b : intArrays) {
    b->size = nodeArrayBytes;
    createStorageBuffer(DEVICE, b->buffer, b->size,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b->memory);
  }

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

  // worldBboxBuffer: 6 floats
  worldBboxBuffer.size = 24;
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

void Bvh::createDescriptor() {

  std::vector<uint32_t> usedBindings = {0,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                        22, 23, 24, 25, 26};
  uint32_t bindingCount = usedBindings.size();

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (uint32_t b : usedBindings) {
    bindings.push_back(VkDescriptorSetLayoutBinding{
        .binding = b,
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

  VkBuffer bufByBinding[27] = {};
  bufByBinding[0] = sceneBuffer.buffer;
  bufByBinding[3] = mortonCodesBuffer.buffer;
  bufByBinding[4] = primIndicesBuffer.buffer;
  bufByBinding[5] = worldBboxBuffer.buffer;
  bufByBinding[6] = histogramBuffer.buffer;
  bufByBinding[7] = scannedGramBuffer.buffer;
  bufByBinding[8] = globSumBuffer.buffer;
  bufByBinding[9] = outputMortonCodeBuffer.buffer;
  bufByBinding[10] = outputPrimIndicesBuffer.buffer;
  for (int i = 0; i < 6; i++)
    bufByBinding[11 + i] = cornerBuffers[i].buffer;
  bufByBinding[17] = leftChildsBuffer.buffer;
  bufByBinding[18] = rightChildsBuffer.buffer;
  bufByBinding[19] = parentsBuffer.buffer;
  bufByBinding[20] = atomicCountersBuffer.buffer;
  for (int i = 0; i < 6; i++)
    bufByBinding[21 + i] = cornerOutBuffers[i].buffer;

  for (uint32_t set = 0; set < 2; set++) {
    if (set == 1) {
      std::swap(bufByBinding[3], bufByBinding[9]);
      std::swap(bufByBinding[4], bufByBinding[10]);
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos(bindingCount);
    std::vector<VkWriteDescriptorSet> writeSets;
    for (uint32_t i = 0; i < bindingCount; i++) {
      uint32_t b = usedBindings[i];
      bufferInfos[i] = {bufByBinding[b], 0, VK_WHOLE_SIZE};
      writeSets.push_back(VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _bvhDescriptors[set],
          .dstBinding = b,
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
      "build_bboxes.spv",         "reorder_leaf_bboxes.spv"};

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
