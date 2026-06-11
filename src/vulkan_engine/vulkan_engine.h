#pragma once

#include <cstdint>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

#include "vulkan_types.h"

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

class VulkanEngine {
public:
  static VulkanEngine &Get();

  void init(); // Initalize everything in the engine

  uint32_t begin_frame();
  void end_frame(uint32_t imageIndex);

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

  VkBuffer _stagingBuffer;
  VkDeviceMemory _stagingMemory;
  VkExtent2D _renderImgExt;
  VkImage _renderImage;
  VkDeviceMemory _renderImageMemory;

  // Commands
  VkCommandPool _commandPool;
  std::vector<VkCommandBuffer> _commandBuffers;

  // Window
  VkExtent2D _windowExtent{1600, 1080};
  struct SDL_Window *_window{nullptr};

  // Placeholder Image TODO :: REMOVE ONCE RAYTRACER IS BUILT IN

  // Sync Objects
  std::vector<VkSemaphore> _presentCompleteSemaphores;
  std::vector<VkSemaphore> _renderFinishedSemaphores;
  std::vector<VkFence> _inFlightFences;

  VkBuffer placeholderData;
  void write_image(const std::vector<uint8_t> &image_data, uint32_t width,
                   uint32_t height);

  void transition_image_layout(VkImage image, VkImageLayout old_layout,
                               VkImageLayout new_layout,
                               VkAccessFlags2 src_access_flags,
                               VkAccessFlags2 dst_access_flags,
                               VkPipelineStageFlags2 src_stage_flags,
                               VkPipelineStageFlags2 dst_stage_flags);

  uint32_t frame_index = 0;

private:
  void init_vulkan(); // Init basics
  void init_swapchain();
  void init_commands();
  void init_placeholder(); // TODO :: DELETE ONCE RAYTRACER

  void create_sync_objects();

  void record_buffer(uint32_t image_index);

  void allocate_buffer(VkBuffer &buffer, VkDeviceMemory &deviceMemory,
                       VkDeviceSize size, VkBufferUsageFlags usage_flags,
                       VkMemoryPropertyFlags properties);

  uint32_t find_memory_type(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);
};
