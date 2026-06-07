#include "lbvh_gpu.h"
#include "metrics_macros.h"
#include "scenes.h"
#include "types.h"
#include "vulkan_engine.h"

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

  auto config = load_bunny(IMAGE_WIDTH, SAMPLES);
  LBVH bvh_tree(config.world.objects);
  config.cam.render(bvh_tree);

  engine.write_image(config.cam.img.data, config.cam.img.width,
                     config.cam.img.height);

  engine.run();

  METRIC_EXPORT("test.csv");

  return 0;
}
