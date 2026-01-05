#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H
//==============================================================================================
// GPU Renderer for Ray Tracing
// Separated from BVH construction for better code organization
//==============================================================================================

#include "../bvh/lbvh_gpu.h"
#include "color.h"

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// Forward declarations
class camera;
class material;
class diffuse_light;
class lambertian;
class metal;
class sphere;
class triangle;
class hittable;

// GPU-friendly data structures
struct gpu_sphere {
    float radius;
    struct { float x, y, z; } center;
    int mat_idx;
};

struct gpu_triangle {
    struct { float x, y, z; } v0, v1, v2;     // Vertices
    struct { float x, y, z; } n0, n1, n2;     // Vertex normals
    int mat_idx;
    int smooth_shading;  // 0=flat, 1=smooth
    float _pad0, _pad1;  // Alignment padding
};

// Material type enum (matches OpenCL kernel definitions)
enum {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_LIGHT = 2
};

struct gpu_material {
    int type;
    struct { float x, y, z; } albedo;
    float fuzz;
    float _pad0;
};

class gpu_renderer {
  public:
    // Constructor: Takes a built BVH and prepares for GPU rendering
    gpu_renderer(const lbvh_gpu& bvh);

    // Destructor
    ~gpu_renderer();

    // Render scene using GPU
    void render(const camera& cam, std::vector<color>& output_pixels);

    // Get statistics
    int get_num_spheres() const { return num_spheres; }
    int get_num_triangles() const { return num_triangles; }
    int get_num_materials() const { return num_materials; }

  private:
    // Reference to BVH (does not own)
    const lbvh_gpu& bvh;

    // OpenCL resources (shares context/device/queue from BVH)
    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    // Rendering kernels
    cl_kernel kernel_render_primary = nullptr;
    cl_kernel kernel_render_sampling = nullptr;

    // Rendering-specific buffers
    cl_mem buf_spheres = nullptr;
    cl_mem buf_triangles = nullptr;
    cl_mem buf_leaf_prim_indices = nullptr;
    cl_mem buf_prim_types = nullptr;
    cl_mem buf_materials = nullptr;

    // Scene data
    std::vector<gpu_material> materials;
    int num_spheres = 0;
    int num_triangles = 0;
    int num_materials = 0;

    // Initialization and cleanup
    void init_opencl_rendering();
    void prepare_scene_buffers();
    void cleanup_opencl();

    // Helper methods
    void check_cl_error(cl_int err, const char* operation) const;
    std::string load_kernel_source_with_includes(const std::vector<const char*>& filenames) const;
};

//==============================================================================================
// gpu_renderer Implementation
//==============================================================================================

#include "material.h"
#include "sphere.h"
#include "triangle.h"

inline gpu_renderer::gpu_renderer(const lbvh_gpu& bvh) : bvh(bvh) {
    // Reuse OpenCL resources from BVH
    context = bvh.get_opencl_context();
    device = bvh.get_opencl_device();
    queue = bvh.get_opencl_queue();

    // Initialize rendering-specific resources
    init_opencl_rendering();
    prepare_scene_buffers();
}

inline gpu_renderer::~gpu_renderer() {
    cleanup_opencl();
}

inline void gpu_renderer::init_opencl_rendering() {
    cl_int err;

    // Load and compile rendering kernels
    std::string source = load_kernel_source_with_includes({
        "src/bvh/opencl_shared_types.cl",
        "src/raytracer/gpu_renderer_kernels.cl"
    });
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();

    program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    check_cl_error(err, "clCreateProgramWithSource(rendering)");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL rendering kernel compilation error:\n" << log.data() << std::endl;
        exit(1);
    }

    // Create rendering kernels
    kernel_render_sampling = clCreateKernel(program, "render_with_sampling", &err);
    check_cl_error(err, "clCreateKernel(render_with_sampling)");

    kernel_render_primary = clCreateKernel(program, "render_primary_rays", &err);
    check_cl_error(err, "clCreateKernel(render_primary_rays)");
}

inline void gpu_renderer::prepare_scene_buffers() {
    cl_int err;

    const std::vector<shared_ptr<hittable>>& objects = bvh.get_objects();
    int N = bvh.get_primitive_count();

    // Extract spheres, triangles, and materials from objects vector
    std::vector<gpu_sphere> spheres;
    std::vector<gpu_triangle> triangles;
    std::vector<int> leaf_prim_indices(N);
    std::vector<int> prim_types(N);  // 0=sphere, 1=triangle
    std::map<const material*, int> mat_to_idx;

    for (int i = 0; i < N; i++) {
        // Use dynamic_cast to identify sphere primitives
        if (auto* sph = dynamic_cast<const sphere*>(objects[i].get())) {
            prim_types[i] = 0;  // Sphere
            leaf_prim_indices[i] = spheres.size();

            gpu_sphere gs;
            point3 sph_center = sph->center_at(0);  // Stationary sphere at t=0
            gs.center.x = sph_center.x();
            gs.center.y = sph_center.y();
            gs.center.z = sph_center.z();
            gs.radius = sph->get_radius();

            // Extract material
            const material* mat_ptr = sph->get_material_ptr();

            // Check if we've seen this material before
            if (mat_to_idx.find(mat_ptr) == mat_to_idx.end()) {
                // New material - add to materials vector
                int idx = materials.size();
                mat_to_idx[mat_ptr] = idx;

                gpu_material gm;
                gm.fuzz = 0.0f;
                gm._pad0 = 0.0f;

                // Identify material type and extract properties
                if (auto* lamb = dynamic_cast<const lambertian*>(mat_ptr)) {
                    gm.type = MAT_LAMBERTIAN;
                    color alb = lamb->get_albedo();
                    gm.albedo.x = alb.x();
                    gm.albedo.y = alb.y();
                    gm.albedo.z = alb.z();
                }
                else if (auto* met = dynamic_cast<const metal*>(mat_ptr)) {
                    gm.type = MAT_METAL;
                    color alb = met->get_albedo();
                    gm.albedo.x = alb.x();
                    gm.albedo.y = alb.y();
                    gm.albedo.z = alb.z();
                    gm.fuzz = met->get_fuzz();
                }
                else if (auto* light = dynamic_cast<const diffuse_light*>(mat_ptr)) {
                    gm.type = MAT_LIGHT;
                    color emit = light->get_emit();
                    gm.albedo.x = emit.x();
                    gm.albedo.y = emit.y();
                    gm.albedo.z = emit.z();
                }
                else {
                    // Unknown material type - default to white lambertian
                    std::cerr << "Warning: Unknown material type, defaulting to white lambertian\n";
                    gm.type = MAT_LAMBERTIAN;
                    gm.albedo.x = 1.0f;
                    gm.albedo.y = 1.0f;
                    gm.albedo.z = 1.0f;
                }

                materials.push_back(gm);
            }

            // Set sphere's material index
            gs.mat_idx = mat_to_idx[mat_ptr];

            spheres.push_back(gs);
        } else if (auto* tri = dynamic_cast<const triangle*>(objects[i].get())) {
            prim_types[i] = 1;  // Triangle
            leaf_prim_indices[i] = triangles.size();

            gpu_triangle gt;

            // Extract vertices
            point3 v0 = tri->get_v0();
            gt.v0.x = v0.x(); gt.v0.y = v0.y(); gt.v0.z = v0.z();

            point3 v1 = tri->get_v1();
            gt.v1.x = v1.x(); gt.v1.y = v1.y(); gt.v1.z = v1.z();

            point3 v2 = tri->get_v2();
            gt.v2.x = v2.x(); gt.v2.y = v2.y(); gt.v2.z = v2.z();

            // Extract normals
            vec3 n0 = tri->get_n0();
            gt.n0.x = n0.x(); gt.n0.y = n0.y(); gt.n0.z = n0.z();

            vec3 n1 = tri->get_n1();
            gt.n1.x = n1.x(); gt.n1.y = n1.y(); gt.n1.z = n1.z();

            vec3 n2 = tri->get_n2();
            gt.n2.x = n2.x(); gt.n2.y = n2.y(); gt.n2.z = n2.z();

            gt.smooth_shading = tri->uses_smooth_shading() ? 1 : 0;

            // Extract material
            const material* mat_ptr = tri->get_material_ptr();

            // Check if we've seen this material before
            if (mat_to_idx.find(mat_ptr) == mat_to_idx.end()) {
                // New material - add to materials vector
                int idx = materials.size();
                mat_to_idx[mat_ptr] = idx;

                gpu_material gm;
                gm.fuzz = 0.0f;
                gm._pad0 = 0.0f;

                // Identify material type and extract properties
                if (auto* lamb = dynamic_cast<const lambertian*>(mat_ptr)) {
                    gm.type = MAT_LAMBERTIAN;
                    color alb = lamb->get_albedo();
                    gm.albedo.x = alb.x();
                    gm.albedo.y = alb.y();
                    gm.albedo.z = alb.z();
                }
                else if (auto* met = dynamic_cast<const metal*>(mat_ptr)) {
                    gm.type = MAT_METAL;
                    color alb = met->get_albedo();
                    gm.albedo.x = alb.x();
                    gm.albedo.y = alb.y();
                    gm.albedo.z = alb.z();
                    gm.fuzz = met->get_fuzz();
                }
                else if (auto* light = dynamic_cast<const diffuse_light*>(mat_ptr)) {
                    gm.type = MAT_LIGHT;
                    color emit = light->get_emit();
                    gm.albedo.x = emit.x();
                    gm.albedo.y = emit.y();
                    gm.albedo.z = emit.z();
                }
                else {
                    // Unknown material type - default to white lambertian
                    std::cerr << "Warning: Unknown material type, defaulting to white lambertian\n";
                    gm.type = MAT_LAMBERTIAN;
                    gm.albedo.x = 1.0f;
                    gm.albedo.y = 1.0f;
                    gm.albedo.z = 1.0f;
                }

                materials.push_back(gm);
            }

            gt.mat_idx = mat_to_idx[mat_ptr];
            gt._pad0 = 0.0f;
            gt._pad1 = 0.0f;

            triangles.push_back(gt);
        } else {
            std::cerr << "Warning: Unsupported primitive type skipped in GPU rendering\n";
            prim_types[i] = -1;
            leaf_prim_indices[i] = -1;
        }
    }

    num_spheres = spheres.size();
    num_triangles = triangles.size();
    num_materials = materials.size();
    std::clog << "Prepared " << num_spheres << " spheres, " << num_triangles << " triangles for GPU rendering\n";
    std::clog << "Extracted " << num_materials << " unique materials\n";

    // Create GPU buffers (use max(size,1) to avoid zero-sized buffers)
    buf_spheres = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        std::max(spheres.size(), size_t(1)) * sizeof(gpu_sphere),
        spheres.empty() ? nullptr : spheres.data(), &err
    );
    check_cl_error(err, "clCreateBuffer(buf_spheres)");

    buf_triangles = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        std::max(triangles.size(), size_t(1)) * sizeof(gpu_triangle),
        triangles.empty() ? nullptr : triangles.data(), &err
    );
    check_cl_error(err, "clCreateBuffer(buf_triangles)");

    buf_leaf_prim_indices = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(int),
        leaf_prim_indices.data(), &err
    );
    check_cl_error(err, "clCreateBuffer(buf_leaf_prim_indices)");

    buf_prim_types = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(int),
        prim_types.data(), &err
    );
    check_cl_error(err, "clCreateBuffer(buf_prim_types)");

    buf_materials = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        materials.size() * sizeof(gpu_material),
        materials.data(), &err
    );
    check_cl_error(err, "clCreateBuffer(buf_materials)");
}

inline void gpu_renderer::cleanup_opencl() {
    // Release rendering kernels
    if (kernel_render_primary) clReleaseKernel(kernel_render_primary);
    if (kernel_render_sampling) clReleaseKernel(kernel_render_sampling);

    // Release rendering buffers
    if (buf_spheres) clReleaseMemObject(buf_spheres);
    if (buf_triangles) clReleaseMemObject(buf_triangles);
    if (buf_leaf_prim_indices) clReleaseMemObject(buf_leaf_prim_indices);
    if (buf_prim_types) clReleaseMemObject(buf_prim_types);
    if (buf_materials) clReleaseMemObject(buf_materials);

    // Release rendering program
    if (program) clReleaseProgram(program);

    // Note: context, device, queue are borrowed from BVH, so we don't release them
}

inline void gpu_renderer::check_cl_error(cl_int err, const char* operation) const {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error during " << operation << ": " << err << std::endl;
        exit(1);
    }
}

inline std::string gpu_renderer::load_kernel_source_with_includes(const std::vector<const char*>& filenames) const {
    std::stringstream combined;

    for (const char* filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open kernel file: " << filename << std::endl;
            exit(1);
        }
        combined << "// ============ BEGIN: " << filename << " ============\n";
        combined << file.rdbuf();
        combined << "\n// ============ END: " << filename << " ============\n\n";
    }

    return combined.str();
}

inline void gpu_renderer::render(const camera& cam, std::vector<color>& output_pixels) {
    cl_int err;

    int width = cam.image_width;
    int height = cam.get_image_height();
    int samples = cam.get_samples_per_pixel();
    int depth = cam.get_max_depth();

    // POD structs for camera parameters
    struct { float x, y, z; } cam_center, pixel00, delta_u, delta_v, defocus_u, defocus_v, bg;

    point3 tmp_center = cam.get_center();
    cam_center = {(float)tmp_center.x(), (float)tmp_center.y(), (float)tmp_center.z()};

    point3 tmp_pixel00 = cam.get_pixel00_loc();
    pixel00 = {(float)tmp_pixel00.x(), (float)tmp_pixel00.y(), (float)tmp_pixel00.z()};

    vec3 tmp_delta_u = cam.get_pixel_delta_u();
    delta_u = {(float)tmp_delta_u.x(), (float)tmp_delta_u.y(), (float)tmp_delta_u.z()};

    vec3 tmp_delta_v = cam.get_pixel_delta_v();
    delta_v = {(float)tmp_delta_v.x(), (float)tmp_delta_v.y(), (float)tmp_delta_v.z()};

    vec3 tmp_defocus_u = cam.get_defocus_disk_u();
    defocus_u = {(float)tmp_defocus_u.x(), (float)tmp_defocus_u.y(), (float)tmp_defocus_u.z()};

    vec3 tmp_defocus_v = cam.get_defocus_disk_v();
    defocus_v = {(float)tmp_defocus_v.x(), (float)tmp_defocus_v.y(), (float)tmp_defocus_v.z()};

    float defocus_angle_f = (float)cam.get_defocus_angle();

    color tmp_bg = cam.background;
    bg = {(float)tmp_bg.x(), (float)tmp_bg.y(), (float)tmp_bg.z()};

    size_t num_pixels = width * height;
    cl_mem buf_output = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        num_pixels * sizeof(float) * 3,
        nullptr, &err
    );
    check_cl_error(err, "clCreateBuffer(output)");

    // Get BVH resources
    cl_mem buf_nodes = bvh.get_nodes_buffer();
    int N = bvh.get_primitive_count();

    // Set kernel arguments (20 total - was 18)
    int arg = 0;
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cam_center), &cam_center);
    check_cl_error(err, "clSetKernelArg(camera_center)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(pixel00), &pixel00);
    check_cl_error(err, "clSetKernelArg(pixel00)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(delta_u), &delta_u);
    check_cl_error(err, "clSetKernelArg(delta_u)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(delta_v), &delta_v);
    check_cl_error(err, "clSetKernelArg(delta_v)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(defocus_u), &defocus_u);
    check_cl_error(err, "clSetKernelArg(defocus_disk_u)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(defocus_v), &defocus_v);
    check_cl_error(err, "clSetKernelArg(defocus_disk_v)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(float), &defocus_angle_f);
    check_cl_error(err, "clSetKernelArg(defocus_angle)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &width);
    check_cl_error(err, "clSetKernelArg(width)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &height);
    check_cl_error(err, "clSetKernelArg(height)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &samples);
    check_cl_error(err, "clSetKernelArg(samples_per_pixel)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &depth);
    check_cl_error(err, "clSetKernelArg(max_depth)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(bg), &bg);
    check_cl_error(err, "clSetKernelArg(background)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_nodes);
    check_cl_error(err, "clSetKernelArg(nodes)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_spheres);
    check_cl_error(err, "clSetKernelArg(spheres)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_triangles);
    check_cl_error(err, "clSetKernelArg(triangles)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_leaf_prim_indices);
    check_cl_error(err, "clSetKernelArg(leaf_prim_indices)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_prim_types);
    check_cl_error(err, "clSetKernelArg(prim_types)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_materials);
    check_cl_error(err, "clSetKernelArg(materials)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &N);
    check_cl_error(err, "clSetKernelArg(N)");
    err = clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_output);
    check_cl_error(err, "clSetKernelArg(output)");

    size_t global[2] = {(size_t)width, (size_t)height};

    std::clog << "Launching GPU render kernel (" << width << "×" << height
              << " pixels, " << samples << " samples, depth=" << depth << ")...\n";

    err = clEnqueueNDRangeKernel(
        queue, kernel_render_sampling, 2, nullptr,
        global, nullptr, 0, nullptr, nullptr
    );
    check_cl_error(err, "clEnqueueNDRangeKernel(render)");

    clFinish(queue);

    std::clog << "GPU rendering complete. Reading back pixels...\n";

    std::vector<float> pixel_data(num_pixels * 3);
    err = clEnqueueReadBuffer(
        queue, buf_output, CL_TRUE, 0,
        pixel_data.size() * sizeof(float),
        pixel_data.data(), 0, nullptr, nullptr
    );
    check_cl_error(err, "clEnqueueReadBuffer(output)");

    output_pixels.resize(num_pixels);
    for (size_t i = 0; i < num_pixels; i++) {
        output_pixels[i] = color(
            pixel_data[i*3],
            pixel_data[i*3+1],
            pixel_data[i*3+2]
        );
    }

    clReleaseMemObject(buf_output);

    std::clog << "Pixel readback complete.\n";
}

#endif
