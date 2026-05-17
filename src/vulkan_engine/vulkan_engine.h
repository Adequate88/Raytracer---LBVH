#pragma once

#include <cstdint>
#include <vulkan/vulkan_core.h>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

#include "vulkan_types.h"

class VulkanEngine {
public:
  static VulkanEngine &Get();

  void init(); // Initalize everything in the engine

  void run();

  void cleanup();

  VkQueue _graphicsQueue;

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debugger;
  VkDevice _device;
  VkPhysicalDevice _gpu;
  VkSurfaceKHR _surface;

  VkSwapchainKHR _swapchain;
  uint32_t _swapchainImageCount;
  VkSurfaceFormatKHR _swapchainSurfaceFormat;
  VkExtent2D _swapchainExtent;
  std::vector<VkImage> _swapchainImages;
  std::vector<VkImageView> _swapchainImageViews;

  VkExtent2D _windowExtent{1600, 1080};
  struct SDL_Window *_window{nullptr};

private:
  void init_vulkan(); // Init basics
  void init_swapchain();
  void init_surface();
};
