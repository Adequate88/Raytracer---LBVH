#pragma once

#include <cstdint>
#include <sys/types.h>
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

  // Queue
  VkQueue _graphicsQueue;
  uint32_t _queueFamilyIndex = 0;

  // Basics
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debugger;
  VkDevice _device;
  VkPhysicalDevice _gpu;
  VkSurfaceKHR _surface;

  // Swapchain
  VkSwapchainKHR _swapchain;
  uint32_t _swapchainImageCount;
  VkSurfaceFormatKHR _swapchainSurfaceFormat;
  VkExtent2D _swapchainExtent;
  std::vector<VkImage> _swapchainImages;
  std::vector<VkImageView> _swapchainImageViews;

  // Commands
  VkCommandPool _commandPool;
  VkCommandBuffer _commandBuffer;

  // Window
  VkExtent2D _windowExtent{1600, 1080};
  struct SDL_Window *_window{nullptr};

  // Placeholder Image TODO :: REMOVE ONCE RAYTRACER IS BUILT IN

  VkImage placeholderImage;
  void write_image();

private:
  void init_vulkan(); // Init basics
  void init_swapchain();
  void init_commands();
  void init_placeholder(); // TODO :: DELETE ONCE RAYTRACER

  void transition_image();
  void record_buffer(uint32_t image_index);
};
