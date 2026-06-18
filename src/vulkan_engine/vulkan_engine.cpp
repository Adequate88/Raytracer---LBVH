#include "vulkan_engine.h"
#include "vulkan_types.h"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fmt/base.h>
#include <map>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>

// Validation Layers
#ifdef EVALUATE
constexpr bool bUseValidationLayers = false;
#else
constexpr bool bUseValidationLayers = true;
#endif
std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

//
VulkanEngine *loadedEngine = nullptr;
VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init() {
  assert(loadedEngine == nullptr);
  loadedEngine = this;

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
    return;
  }

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow("Renderer", _windowExtent.width,
                             _windowExtent.height, window_flags);

  if (!_window) {
    SDL_Log("Couldn't open SDL window: %s", SDL_GetError());
    return;
  }

  init_vulkan();
  init_swapchain();
  init_commands();
  create_sync_objects();
}

void VulkanEngine::cleanup() {}

uint32_t VulkanEngine::begin_frame() {
  VK_CHECK(vkWaitForFences(_device, 1, &_inFlightFences[frame_index], VK_TRUE,
                           UINT64_MAX));
  VK_CHECK(vkResetFences(_device, 1, &_inFlightFences[frame_index]));

  uint32_t imageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX,
                                 _presentCompleteSemaphores[frame_index],
                                 VK_NULL_HANDLE, &imageIndex));

  vkResetCommandBuffer(_commandBuffers[frame_index], {});
  return imageIndex;
}

void VulkanEngine::end_frame(uint32_t imageIndex) {
  VkCommandBufferSubmitInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = _commandBuffers[frame_index],
      .deviceMask = 0};
  VkSemaphoreSubmitInfo presentSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = _presentCompleteSemaphores[frame_index],
      .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
  VkSemaphoreSubmitInfo signalSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = _renderFinishedSemaphores[imageIndex]};
  VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .waitSemaphoreInfoCount = 1,
                           .pWaitSemaphoreInfos = &presentSemaphoreInfo,
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &bufferInfo,
                           .signalSemaphoreInfoCount = 1,
                           .pSignalSemaphoreInfos = &signalSemaphoreInfo};

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo,
                          _inFlightFences[frame_index]));

  VkPresentInfoKHR presentInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                               .waitSemaphoreCount = 1,
                               .pWaitSemaphores =
                                   &_renderFinishedSemaphores[imageIndex],
                               .swapchainCount = 1,
                               .pSwapchains = &_swapchain,
                               .pImageIndices = &imageIndex};

  VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

  frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanEngine::init_vulkan() { // NOTE : Functions written in big chunks
                                   // cause my aim is to learn rn - later might
                                   // cleanup
  // Set-up Layers
  std::vector<char const *> requiredLayers;
  if (bUseValidationLayers)
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());

  // Get SDL extensions
  uint32_t instanceExtensionCount = 0;
  auto instanceExtensions =
      SDL_Vulkan_GetInstanceExtensions(&instanceExtensionCount);
  // Instance
  VkApplicationInfo appInfo{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Raytracer",
                            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                            .pEngineName = "LEngine",
                            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                            .apiVersion = VK_API_VERSION_1_4};

  VkInstanceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames = requiredLayers.data(),
      .enabledExtensionCount = instanceExtensionCount,
      .ppEnabledExtensionNames = instanceExtensions};

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &_instance));

  // Create surface

  if (!SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &_surface)) {
    SDL_Log("Couldn't create SDL Vulkan surface: %s", SDL_GetError());
    return;
  }

  // Device and GPU

  uint32_t deviceCount;
  VK_CHECK(vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr));
  if (deviceCount == 0) {
    fmt::println("No suitable GPUS\n");
    return;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data()));
  std::multimap<int, VkPhysicalDevice> candidates;

  std::vector<const char *> requiredDeviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  for (const VkPhysicalDevice &device : devices) {

    VkPhysicalDeviceProperties2 deviceProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(device, &deviceProperties);

    // Feautres
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamicFeatures{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        .pNext = nullptr};
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &dynamicFeatures};
    VkPhysicalDeviceFeatures2 deviceFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features13};

    vkGetPhysicalDeviceFeatures2(device, &deviceFeatures);

    if (!features13.dynamicRendering)
      continue;
    if (!dynamicFeatures.extendedDynamicState)
      continue;
    if (!deviceFeatures.features.geometryShader)
      continue;

    // Is discrete GPU?
    uint32_t score = 0;
    if (deviceProperties.properties.deviceType ==
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score += 1000;

    score += deviceProperties.properties.limits.maxImageDimension2D;

    // Check extension requirement
    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr,
                                                  &extensionCount, nullptr));
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        device, nullptr, &extensionCount, availableExtensions.data()));
    bool swapchainSupport = std::ranges::all_of(
        requiredDeviceExtensions,
        [&availableExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableExtensions,
              [&requiredDeviceExtension](auto const &availableExtension) {
                return strcmp(requiredDeviceExtension,
                              availableExtension.extensionName) == 0;
              });
        });

    if (!swapchainSupport)
      continue;

    // Check graphics queue family support
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> familyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             familyProperties.data());
    bool graphicsQueueSupport =
        std::ranges::any_of(familyProperties, [](auto const &familyProperty) {
          return !!(familyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT);
        });

    if (!graphicsQueueSupport)
      continue;

    candidates.insert(std::make_pair(score, device));
  }

  // Determine best candidate and set
  if (!candidates.empty() && candidates.rbegin()->first > 0) {
    _gpu = candidates.rbegin()->second;

    VkPhysicalDeviceProperties2 deviceProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(_gpu, &deviceProperties);
    _timestampPeriod = deviceProperties.properties.limits.timestampPeriod;
    fmt::print("GPU found: {}\n", deviceProperties.properties.deviceName);
  } else {
    fmt::print("No valid GPUs found\n");
    return;
  }

  // Queue Info
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(_gpu, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> familyProperties(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(_gpu, &queueFamilyCount,
                                           familyProperties.data());

  VkBool32 surfaceSupported;

  for (uint32_t queueInd = 0; queueInd < familyProperties.size(); queueInd++) {
    if ((familyProperties[queueInd].queueFlags & VK_QUEUE_GRAPHICS_BIT) !=
        static_cast<VkQueueFlags>(0)) {
      VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(_gpu, queueInd, _surface,
                                                    &surfaceSupported));
      if (surfaceSupported) {
        _queueFamilyIndex = queueInd;
        break;
      }
    }
  }

  float queuePriority = 0.5f;
  VkDeviceQueueCreateInfo queueInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = nullptr,
      .queueFamilyIndex = _queueFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority};

  // Logical device

  VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamicFeatures{
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
      .pNext = nullptr,
      .extendedDynamicState = VK_TRUE};
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &dynamicFeatures,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE};
  VkPhysicalDeviceFeatures2 deviceFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features13};

  VkDeviceCreateInfo deviceInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &deviceFeatures,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueInfo,
      .enabledExtensionCount =
          static_cast<uint32_t>(requiredDeviceExtensions.size()),
      .ppEnabledExtensionNames = requiredDeviceExtensions.data(),
  };

  VK_CHECK(vkCreateDevice(_gpu, &deviceInfo, nullptr, &_device));
  vkGetDeviceQueue(_device, _queueFamilyIndex, 0, &_graphicsQueue);
}

void VulkanEngine::init_swapchain() {
  // Retrieve Info

  VkSurfaceCapabilitiesKHR surfaceCapabilities{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_gpu, _surface,
                                                     &surfaceCapabilities));

  uint32_t surfaceFormatCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(_gpu, _surface,
                                                &surfaceFormatCount, nullptr));
  std::vector<VkSurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
      _gpu, _surface, &surfaceFormatCount, surfaceFormats.data()));

  uint32_t surfacePresentModeCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      _gpu, _surface, &surfacePresentModeCount, nullptr));
  std::vector<VkPresentModeKHR> surfacePresentModes(surfacePresentModeCount);
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      _gpu, _surface, &surfacePresentModeCount, surfacePresentModes.data()));

  // Picking format
  const auto formatIter =
      std::ranges::find_if(surfaceFormats, [](const auto &format) {
        return format.format == VK_FORMAT_B8G8R8A8_SRGB &&
               format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      });
  _swapchainSurfaceFormat =
      formatIter != surfaceFormats.end() ? *formatIter : surfaceFormats[0];

  // Present mode
  assert(std::ranges::any_of(surfacePresentModes, [](const auto &presentMode) {
    return presentMode == VK_PRESENT_MODE_FIFO_KHR;
  }));
  VkPresentModeKHR surfacePresentMode =
      std::ranges::any_of(surfacePresentModes,
                          [](const auto &presentMode) {
                            return presentMode == VK_PRESENT_MODE_MAILBOX_KHR;
                          })
          ? VK_PRESENT_MODE_MAILBOX_KHR
          : VK_PRESENT_MODE_FIFO_KHR;

  // Extent
  _swapchainExtent = {
      std::clamp<uint32_t>(_windowExtent.width,
                           surfaceCapabilities.minImageExtent.width,
                           surfaceCapabilities.maxImageExtent.width),
      std::clamp<uint32_t>(_windowExtent.height,
                           surfaceCapabilities.minImageExtent.height,
                           surfaceCapabilities.maxImageExtent.height)};

  uint32_t minImgCount = std::max(3u, surfaceCapabilities.minImageCount);
  uint32_t _swapchainImageCount =
      (minImgCount > surfaceCapabilities.maxImageCount &&
       surfaceCapabilities.maxImageCount > 0)
          ? surfaceCapabilities.maxImageCount
          : minImgCount;

  // Create swapchain
  VkSwapchainCreateInfoKHR swapchainInfo{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .surface = _surface,
      .minImageCount = _swapchainImageCount,
      .imageFormat = _swapchainSurfaceFormat.format,
      .imageColorSpace = _swapchainSurfaceFormat.colorSpace,
      .imageExtent = _swapchainExtent,
      .imageArrayLayers = 1,
      .imageUsage =
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
          VK_IMAGE_USAGE_TRANSFER_DST_BIT, // NOTE : Maybe change this to
                                           // destination bit since we will
                                           // render with CPU first?
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = surfaceCapabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = surfacePresentMode,
      .clipped = true,
      .oldSwapchain = nullptr};

  VK_CHECK(vkCreateSwapchainKHR(_device, &swapchainInfo, nullptr, &_swapchain));

  uint32_t actualImgCount = 0;
  VK_CHECK(
      vkGetSwapchainImagesKHR(_device, _swapchain, &actualImgCount, nullptr));
  _swapchainImages.resize(actualImgCount);
  VK_CHECK(vkGetSwapchainImagesKHR(_device, _swapchain, &actualImgCount,
                                   _swapchainImages.data()));

  assert(!_swapchainImages.empty());

  VkImageViewCreateInfo viewInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = _swapchainSurfaceFormat.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
  viewInfo.components = {
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

  for (auto &image : _swapchainImages) {
    viewInfo.image = image;
    VkImageView imageView;
    VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &imageView));
    _swapchainImageViews.push_back(imageView);
  }
}

void VulkanEngine::init_commands() {
  // Command Pool

  VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = _queueFamilyIndex};
  VK_CHECK(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool));

  // Command Buffer

  _commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

  VkCommandBufferAllocateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = _commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
  VK_CHECK(
      vkAllocateCommandBuffers(_device, &bufferInfo, _commandBuffers.data()));
}

void VulkanEngine::record_buffer(uint32_t image_index) { // record buffer for
  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  VK_CHECK(vkBeginCommandBuffer(_commandBuffers[frame_index], &beginInfo));

  transition_image_layout(
      _renderImage, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {}, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  VkBufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {_renderImgExt.width, _renderImgExt.height, 1}};
  vkCmdCopyBufferToImage(_commandBuffers[frame_index], _stagingBuffer,
                         _renderImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &region);

  transition_image_layout(
      _renderImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  transition_image_layout(
      _swapchainImages[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {}, VK_ACCESS_2_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  VkImageBlit blitRegion{.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                         .srcOffsets = {{0, 0, 0},
                                        {(int32_t)_renderImgExt.width,
                                         (int32_t)_renderImgExt.height, 1}},
                         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                         .dstOffsets = {{0, 0, 0},
                                        {(int32_t)_swapchainExtent.width,
                                         (int32_t)_swapchainExtent.height, 1}}};
  vkCmdBlitImage(
      _commandBuffers[frame_index], _renderImage,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _swapchainImages[image_index],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

  transition_image_layout(
      _swapchainImages[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_2_TRANSFER_WRITE_BIT, {},
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  VK_CHECK(vkEndCommandBuffer(_commandBuffers[frame_index]));
}

void VulkanEngine::transition_image_layout(
    VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
    VkAccessFlags2 src_access_flags, VkAccessFlags2 dst_access_flags,
    VkPipelineStageFlags2 src_stage_flags,
    VkPipelineStageFlags2 dst_stage_flags) {

  VkImageMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage_flags,
      .srcAccessMask = src_access_flags,
      .dstStageMask = dst_stage_flags,
      .dstAccessMask = dst_access_flags,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
  VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                  .dependencyFlags = {},
                                  .imageMemoryBarrierCount = 1,
                                  .pImageMemoryBarriers = &barrier};
  vkCmdPipelineBarrier2(_commandBuffers[frame_index], &dependencyInfo);
}

void VulkanEngine::create_sync_objects() {

  _presentCompleteSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  _renderFinishedSemaphores.resize(_swapchainImages.size());
  _inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

  assert(!_presentCompleteSemaphores.empty() &&
         !_renderFinishedSemaphores.empty() && !_inFlightFences.empty());

  VkFenceCreateInfo drawInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                             .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  VkSemaphoreCreateInfo presentSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    VK_CHECK(vkCreateSemaphore(_device, &presentSemaphoreInfo, nullptr,
                               &_presentCompleteSemaphores[i]));

    VK_CHECK(vkCreateFence(_device, &drawInfo, nullptr, &_inFlightFences[i]));
  }

  VkSemaphoreCreateInfo renderFinishedSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (size_t i = 0; i < _swapchainImages.size(); i++) {
    VK_CHECK(vkCreateSemaphore(_device, &renderFinishedSemaphoreInfo, nullptr,
                               &_renderFinishedSemaphores[i]));
  }
}

void VulkanEngine::allocate_buffer(VkBuffer &buffer,
                                   VkDeviceMemory &deviceMemory,
                                   VkDeviceSize size,
                                   VkBufferUsageFlags usage_flags,
                                   VkMemoryPropertyFlags properties) {
  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = usage_flags,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VK_CHECK(vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

  // Get memory type index
  uint32_t mem_index =
      find_memory_type(memRequirements.memoryTypeBits, properties);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = memRequirements.size,
                                 .memoryTypeIndex = mem_index};

  VK_CHECK(vkAllocateMemory(_device, &allocInfo, nullptr, &deviceMemory));
  VK_CHECK(vkBindBufferMemory(_device, buffer, deviceMemory, 0));
}

uint32_t VulkanEngine::find_memory_type(uint32_t type,
                                        VkMemoryPropertyFlags properties) {

  // Get memory type index
  VkPhysicalDeviceMemoryProperties deviceProperties;
  vkGetPhysicalDeviceMemoryProperties(_gpu, &deviceProperties);
  for (uint32_t mem_index = 0; mem_index < deviceProperties.memoryTypeCount;
       mem_index++) {
    if ((deviceProperties.memoryTypes[mem_index].propertyFlags & properties) ==
            properties &&
        (1u << mem_index) & type)
      return mem_index;
  }

  fmt::println("No valid memory found");
  return 0;
}

void VulkanEngine::write_image(const std::vector<uint8_t> &image_data,
                               uint32_t width, uint32_t height) {
  VkDeviceSize imageSize = image_data.size();

  allocate_buffer(_stagingBuffer, _stagingMemory, imageSize,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  void *mapped;
  vkMapMemory(_device, _stagingMemory, 0, imageSize, {}, &mapped);

  memcpy(mapped, image_data.data(), imageSize);

  vkUnmapMemory(_device, _stagingMemory);

  _renderImgExt = {width, height};

  VkImageCreateInfo imgInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_B8G8R8A8_SRGB,
                            .extent = {width, height, 1},
                            .mipLevels = 1,
                            .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  VK_CHECK(vkCreateImage(_device, &imgInfo, nullptr, &_renderImage));

  VkMemoryRequirements imgMemReq;
  vkGetImageMemoryRequirements(_device, _renderImage, &imgMemReq);
  uint32_t imgMemType = find_memory_type(imgMemReq.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkMemoryAllocateInfo allocInfo{.sType =
                                     VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = imgMemReq.size,
                                 .memoryTypeIndex = imgMemType};

  VK_CHECK(vkAllocateMemory(_device, &allocInfo, nullptr, &_renderImageMemory));
  VK_CHECK(vkBindImageMemory(_device, _renderImage, _renderImageMemory, 0));
}
