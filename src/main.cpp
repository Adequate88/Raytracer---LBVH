#include "bvh.h"
#include "lbvh_gpu.h"
#include "metrics_macros.h"
#include "raytracer.h"
#include "scenes.h"
#include "triangle.h"
#include "types.h"
#include "vulkan_engine.h"
#include <SDL3/SDL.h>
#include <cstdint>

int main() {

  VulkanEngine engine;
  engine.init();

  Raytracer raytracer(engine);

  //auto config = load_bunny(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  auto config = load_teapot(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  // LBVH bvh_tree(config.world.objects);
  //  config.cam.render(bvh_tree); // THIS IS OLD CPU RAYTRACER
  //  engine.write_image(config.cam.img.data, config.cam.img.width,
  //  config.cam.img.height); // OLD WRITE CPU IMAGE TO DISPLAy
  //

  config.cam.initialize();

  raytracer.initRaytracer(config.world.data(),
                          config.world.size() * sizeof(triangle_new),
                          &config.cam.gpu_constants);

  Bvh bvh(engine, config.world.size());
  bvh.init(raytracer.sceneBuffer());
  bvh.build();

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

  // METRIC_EXPORT("data/test.csv");

  return 0;
}
