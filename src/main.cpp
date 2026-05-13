#include "lbvh_gpu.h"
#include "camera.h"
#include "metrics_macros.h"
#include "mesh.h"
#include "types.h"

const int IMAGE_WIDTH = 720;
const int SAMPLES = 20;



// Model paths
const std::string MODELS_DIR = "models/";

// Load a mesh and set up camera for it
struct scene_config {
    hittable_list world;
    camera cam;
    std::string name;
    int triangle_count;
};

scene_config load_bunny(int image_width, int samples) {
    scene_config config;
    config.name = "bunny";

    auto mat = make_shared<lambertian>(color(0.4, 0.6, 0.4));
    auto mesh_triangles = mesh::load_obj(MODELS_DIR + "bunny.obj", mat, true, 10.0);

    for (auto& tri : mesh_triangles->objects) {
        config.world.add(tri);
    }
    config.triangle_count = config.world.objects.size();

    // Camera setup for bunny
    config.cam.aspect_ratio      = 16.0 / 9.0;
    config.cam.image_width       = image_width;
    config.cam.samples_per_pixel = samples;
    config.cam.max_depth         = 10;
    config.cam.background        = color(0.7, 0.8, 1.00);
    config.cam.vfov              = 30;
    config.cam.lookfrom          = point3(-2, 3, 6);
    config.cam.lookat            = point3(-0.3, 1.0, 0);
    config.cam.vup               = vec3(0, 1, 0);
    config.cam.defocus_angle     = 0;
    config.cam.focus_dist        = 3.0;

    return config;
}


// Run experiment with a specific BVH type
template<typename BVH>
void run_experiment(scene_config conf, const std::string& output_prefix)
{
    // Build BVH
    BVH bvh_tree(conf.world.objects);

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

    auto config = load_bunny(IMAGE_WIDTH, SAMPLES);
    run_experiment<LBVH>(config, "BASELINE");

    return 0;
}
