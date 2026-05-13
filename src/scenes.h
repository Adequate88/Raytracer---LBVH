#include "types.h"
#include "camera.h"
#include "metrics_macros.h"
#include "quad.h"
#include "sphere.h"
#include "mesh.h"

const int IMAGE_WIDTH = 360;
const int SAMPLES = 20;
const std::string MODELS_DIR = "models/";

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

scene_config cornell_box() {
    scene_config config;
    config.name = "cornell_box";

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    config.world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    config.world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    config.world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light));
    config.world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    config.world.add(make_shared<quad>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white));
    config.world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    config.world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));
    config.world.add(box2);

    config.triangle_count = config.world.objects.size();

    config.cam.aspect_ratio      = 1.0;
    config.cam.image_width       = 100;
    config.cam.samples_per_pixel = 200;
    config.cam.max_depth         = 50;
    config.cam.background        = color(0,0,0);
    config.cam.vfov     = 40;
    config.cam.lookfrom = point3(278, 278, -800);
    config.cam.lookat   = point3(278, 278, 0);
    config.cam.vup      = vec3(0,1,0);
    config.cam.defocus_angle = 0;

    return config;
}
