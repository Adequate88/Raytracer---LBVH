#include "lbvh_gpu.h"
#include "metrics_macros.h"
#include "raytracer.h"
#include "scenes.h"
#include "types.h"
#include "vulkan_engine.h"
#include <SDL3/SDL.h>
#include <cstdint>

// Run experiment with a specific BVH type
void run_experiment(scene_config conf, const std::string &output_prefix) {
  // Build BVH
  LBVH bvh_tree(conf.world.objects);

  // Render
  METRIC_START_TIME("RENDER_TIME");
  conf.cam.render(bvh_tree);
  METRIC_END_TIME("RENDER_TIME");

  // Get stats from camera (populated by GPU renderer)
  // TODO GET STATS FROM CAMERA

  // Export stats
  // TODO EXPORT STATS
  std::string filename = "data/" + output_prefix + ".csv";
}

int main() {

  VulkanEngine engine;
  engine.init();

  Raytracer raytracer(engine);
  raytracer.initRaytracer();

  auto config = load_bunny(IMAGE_WIDTH, SAMPLES);
  LBVH bvh_tree(config.world.objects);
  // config.cam.render(bvh_tree); // THIS IS OLD CPU RAYTRACER
  // engine.write_image(config.cam.img.data, config.cam.img.width,
  // config.cam.img.height); // OLD WRITE CPU IMAGE TO DISPLAy

  bool bQuit = false;
  SDL_Event e;

  while (!bQuit) {
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_EVENT_QUIT)
        bQuit = true;
    }
    uint32_t idx = engine.begin_frame();
    raytracer.recordBuffer(idx);
    engine.end_frame(idx);
  }
  VK_CHECK(vkDeviceWaitIdle(engine._device));

  METRIC_EXPORT("test.csv");

  return 0;
}
