#include "vulkan_engine.h"
#include "SDL3/SDL_opengl.h"
#include "vulkan_types.h"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>

// Validation Layers
constexpr bool bUseValidationLayers = true;
std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

//
VulkanEngine *loadedEngine = nullptr;
VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::run() {
  bool bQuit = false;
  SDL_Event e;

  while (!bQuit) {
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_EVENT_QUIT)
        bQuit = true;
    }
  }
}

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
}

void VulkanEngine::cleanup() {}

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
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, // NOTE : Maybe change this to
                                               // destination bit since we will
                                               // render with CPU first?
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = surfaceCapabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = surfacePresentMode,
      .clipped = true,
      .oldSwapchain = nullptr};

  VK_CHECK(vkCreateSwapchainKHR(_device, &swapchainInfo, nullptr, &_swapchain));
  VK_CHECK(vkGetSwapchainImagesKHR(_device, _swapchain, &_swapchainImageCount,
                                   _swapchainImages.data()));

  // Image Views
  assert(_swapchainImages.empty());
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
  VkCommandBufferAllocateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = _commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VK_CHECK(vkAllocateCommandBuffers(_device, &bufferInfo, &_commandBuffer));
}

void VulkanEngine::record_buffer(uint32_t image_index) {
  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
}

void VulkanEngine::transition_image() {}

void VulkanEngine::init_placeholder() {
  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = _swapchainSurfaceFormat.format,
      .extent = {_windowExtent.width, _windowExtent.height, 0},
      .mipLevels = 0,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,

  };
}
