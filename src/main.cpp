#include "bvh.h"
#include "lbvh_gpu.h"
#include "metrics_macros.h"
#include "raytracer.h"
#include "scene_generator.h"
#include "scenes.h"
#include "triangle.h"
#include "types.h"
#include "vulkan_engine.h"
#include <SDL3/SDL.h>
#include <cstdint>

int main() {

  VulkanEngine engine;
  METRIC_START_TIME("Vulkan Device Init");
  engine.init();
  METRIC_END_TIME("Vulkan Device Init");

  Raytracer raytracer(engine);

  // auto config = load_bunny(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  auto config = load_teapot(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  int triangle_target = 8000;
  // auto config =
  //     load_sphere(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES, triangle_target);
  //  LBVH bvh_tree(config.world.objects);
  //   config.cam.render(bvh_tree); // THIS IS OLD CPU RAYTRACER
  //   engine.write_image(config.cam.img.data, config.cam.img.width,
  //   config.cam.img.height); // OLD WRITE CPU IMAGE TO DISPLAy
  //

  config.cam.initialize();

  Bvh bvh(engine, config.world.size());

  METRIC_START_TIME("Total BVH Construction Time");
  METRIC_START_TIME("BVH Initialization");
  bvh.init(config.world.data(), config.world.size() * sizeof(triangle_new));
  METRIC_END_TIME("BVH Initialization");
  bvh.build();
  METRIC_END_TIME("Total BVH Construction Time");

  raytracer.initRaytracer(&config.cam.gpu_constants, bvh.sceneBufferHandle(),
                          bvh.bvhBufferHandle());

  METRIC_BENCHMARK(100, 10, bvh.build());

  METRIC_SET_VALUE("Ray Count",
                   static_cast<float>(IMAGE_WIDTH * IMAGE_HEIGHT *
                                      config.cam.samples_per_pixel));

  SDL_Event e;

#ifdef EVALUATE
  METRIC_BENCHMARK(15, 3, {
    while (SDL_PollEvent(&e) != 0) {
    }
    raytracer.forceRerender();
    uint32_t idx = engine.begin_frame();
    raytracer.recordBuffer(idx);
    engine.end_frame(idx);
    raytracer.recordRenderTime();
    METRIC_SET_VALUE("Rays traced per second",
                     METRIC_READ("Ray Count") /
                         (METRIC_READ("Total Rendering Time") * ms_scale));
  });
#else
  bool bQuit = false;
  while (!bQuit) {
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_EVENT_QUIT)
        bQuit = true;
    }
    uint32_t idx = engine.begin_frame();
    raytracer.recordBuffer(idx);
    engine.end_frame(idx);
  }
#endif

  VK_CHECK(vkDeviceWaitIdle(engine._device));

#ifdef EVALUATE
  METRIC_SET_VALUE("Total Run Time",
                   METRIC_READ("Vulkan Device Init") +
                       METRIC_READ("Total BVH Construction Time") +
                       METRIC_READ("Total Rendering Time"));
#endif

  METRIC_EXPORT("data/raytracer_with_accel_traversal.csv");

  return 0;
}
