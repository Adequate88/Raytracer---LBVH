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
#include <algorithm>
#include <cmath>
#include <cstdint>

int main() {

  VulkanEngine engine;
  METRIC_START_TIME("Vulkan Device Init");
  engine.init();
  METRIC_END_TIME("Vulkan Device Init");

  Raytracer raytracer(engine);

  auto config = load_bunny(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  // auto config = load_teapot(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  //  auto config = load_conference(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES);
  //     LBVH bvh_tree(config.world.objects);
  //      config.cam.render(bvh_tree); // THIS IS OLD CPU RAYTRACER
  //      engine.write_image(config.cam.img.data, config.cam.img.width,
  //      config.cam.img.height); // OLD WRITE CPU IMAGE TO DISPLAy
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
                          bvh.traversalHandles());

#ifdef RGP_CAPTURE
  auto present_once = [&]() {
    uint32_t idx = engine.begin_frame();
    raytracer.recordWavefrontBuffer(idx);
    engine.end_frame(idx);
  };

  present_once();

  bvh.build();
  present_once();

  raytracer.forceRerender();
  present_once();

  VK_CHECK(vkDeviceWaitIdle(engine._device));
#endif

  METRIC_BENCHMARK(100, 10, bvh.build());

  METRIC_SET_VALUE("Ray Count",
                   static_cast<float>(IMAGE_WIDTH * IMAGE_HEIGHT *
                                      config.cam.samples_per_pixel));

  SDL_Event e;

#ifdef EVALUATE
  METRIC_BENCHMARK(15, 3, {
    while (SDL_PollEvent(&e) != 0) {
    }
    uint32_t idx = engine.begin_frame();
    raytracer.recordWavefrontBuffer(idx);
    engine.end_frame(idx);
    raytracer.recordRenderTime();
    METRIC_SET_VALUE("Rays traced per second",
                     METRIC_READ("Ray Count") /
                         (METRIC_READ("Total Rendering Time") * ms_scale));
  });
#else
  const double move_speed = 0.25; // per frame; tune to scene scale
  const double mouse_sens = 0.0025;

  point3 cam_pos = config.cam.lookfrom;
  vec3 dir = unit_vector(config.cam.lookat - config.cam.lookfrom);
  double pitch = std::asin(std::clamp(dir.y(), -1.0, 1.0));
  double yaw = std::atan2(dir.z(), dir.x());

  SDL_SetWindowRelativeMouseMode(engine._window, true);

  bool bQuit = false;
  while (!bQuit) {
    double dx = 0, dy = 0;
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_EVENT_QUIT)
        bQuit = true;
      else if (e.type == SDL_EVENT_MOUSE_MOTION) {
        dx += e.motion.xrel;
        dy += e.motion.yrel;
      }
    }

    yaw += dx * mouse_sens;
    pitch -= dy * mouse_sens;
    const double pitch_limit = degrees_to_radians(89.0);
    pitch = std::clamp(pitch, -pitch_limit, pitch_limit);

    vec3 forward =
        unit_vector(vec3(std::cos(pitch) * std::cos(yaw), std::sin(pitch),
                         std::cos(pitch) * std::sin(yaw)));
    vec3 right = unit_vector(cross(forward, config.cam.vup));

    const bool *keys = SDL_GetKeyboardState(nullptr);
    if (keys[SDL_SCANCODE_ESCAPE])
      bQuit = true;
    double speed = move_speed * 1.0;
    if (keys[SDL_SCANCODE_W])
      cam_pos = cam_pos + speed * forward;
    if (keys[SDL_SCANCODE_S])
      cam_pos = cam_pos - speed * forward;
    if (keys[SDL_SCANCODE_D])
      cam_pos = cam_pos + speed * right;
    if (keys[SDL_SCANCODE_A])
      cam_pos = cam_pos - speed * right;

    config.cam.lookfrom = cam_pos;
    config.cam.lookat = cam_pos + forward;
    config.cam.initialize();

    uint32_t idx = engine.begin_frame();
    // raytracer.recordBuffer(idx);
    raytracer.recordWavefrontBuffer(idx);
    engine.end_frame(idx);
  }
#endif

  VK_CHECK(vkDeviceWaitIdle(engine._device));

#ifdef EVALUATE
  METRIC_SET_VALUE("Total Run Time",
                   METRIC_READ("Vulkan Device Init") +
                       METRIC_READ("Total BVH Construction Time") +
                       METRIC_READ("Total Rendering Time"));
  METRIC_EXPORT("data/wavefront_with_accel_traversal.csv");
#endif

  return 0;
}
