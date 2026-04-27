#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include "color.h"
#include "material.h"
#include "sphere.h"
#include "triangle.h"

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct camera_params {
    int width, height, samples, max_depth;
    float center_x, center_y, center_z;
    float pixel00_x, pixel00_y, pixel00_z;
    float delta_u_x, delta_u_y, delta_u_z;
    float delta_v_x, delta_v_y, delta_v_z;
    float defocus_u_x, defocus_u_y, defocus_u_z;
    float defocus_v_x, defocus_v_y, defocus_v_z;
    float defocus_angle;
    float bg_r, bg_g, bg_b;
};

struct gpu_bvh_node {
    float bbox_min_x, bbox_min_y, bbox_min_z;
    float bbox_max_x, bbox_max_y, bbox_max_z;
    int left, right, parent, primitive_id;
    int atomic_counter;
    int _pad;
};

struct gpu_sphere {
    float radius;
    struct { float x, y, z; } center;
    int mat_idx;
};

struct gpu_triangle {
    struct { float x, y, z; } v0, v1, v2;
    struct { float x, y, z; } n0, n1, n2;
    int mat_idx, smooth_shading;
    float _pad0, _pad1;
};

enum { MAT_LAMBERTIAN = 0, MAT_METAL = 1, MAT_LIGHT = 2 };

struct gpu_material {
    int type;
    struct { float x, y, z; } albedo;
    float fuzz, _pad0;
};

template<typename BVH>
class gpu_renderer {
  public:
    gpu_renderer(BVH& bvh) : bvh_ref(bvh) {
        init_opencl();
        init_rendering_kernels();
        upload_bvh_nodes();
        prepare_scene_buffers();
    }

    ~gpu_renderer() { cleanup_opencl(); }

    void render(const camera_params& cam, std::vector<color>& output_pixels);

    void export_statistics(const std::string& filename) const {
        bvh_ref.export_statistics(filename, last_total_rays, last_avg_box_tests, last_avg_prim_tests);
    }

    long long get_total_rays() const { return last_total_rays; }
    double get_avg_box_tests() const { return last_avg_box_tests; }
    double get_avg_prim_tests() const { return last_avg_prim_tests; }

  private:
    BVH& bvh_ref;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel kernel_render_sampling = nullptr;
    cl_kernel kernel_render_primary = nullptr;

    cl_mem buf_nodes = nullptr;
    cl_mem buf_spheres = nullptr;
    cl_mem buf_triangles = nullptr;
    cl_mem buf_leaf_prim_indices = nullptr;
    cl_mem buf_prim_types = nullptr;
    cl_mem buf_materials = nullptr;

    std::vector<gpu_material> materials;
    int N = 0;

    long long last_total_rays = 0;
    double last_avg_box_tests = 0.0;
    double last_avg_prim_tests = 0.0;

    void init_opencl() {
        cl_int err;
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
    }

    void init_rendering_kernels() {
        cl_int err;
        std::string source = load_kernel_source({
            "src/bvh/opencl_shared_types.cl",
            "src/raytracer/gpu_renderer_kernels.cl"
        });
        const char* src_ptr = source.c_str();
        size_t src_len = source.length();

        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL kernel error:\n" << log.data() << std::endl;
            exit(1);
        }

        kernel_render_sampling = clCreateKernel(program, "render_with_sampling", &err);
        kernel_render_primary = clCreateKernel(program, "render_primary_rays", &err);
    }

    void upload_bvh_nodes() {
        cl_int err;
        N = bvh_ref.get_primitive_count();
        const auto& nodes = bvh_ref.get_nodes();
        int num_nodes = 2 * N - 1;

        std::vector<gpu_bvh_node> gpu_nodes(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            const auto& node = nodes[i];
            gpu_nodes[i].bbox_min_x = node.bbox.min_x;
            gpu_nodes[i].bbox_min_y = node.bbox.min_y;
            gpu_nodes[i].bbox_min_z = node.bbox.min_z;
            gpu_nodes[i].bbox_max_x = node.bbox.max_x;
            gpu_nodes[i].bbox_max_y = node.bbox.max_y;
            gpu_nodes[i].bbox_max_z = node.bbox.max_z;
            gpu_nodes[i].left = node.left;
            gpu_nodes[i].right = node.right;
            gpu_nodes[i].parent = node.parent;
            gpu_nodes[i].primitive_id = node.primitive_id;
            gpu_nodes[i].atomic_counter = 0;
        }

        buf_nodes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   num_nodes * sizeof(gpu_bvh_node), gpu_nodes.data(), &err);
    }

    void prepare_scene_buffers() {
        cl_int err;
        const auto& objects = bvh_ref.get_objects();

        std::vector<gpu_sphere> spheres;
        std::vector<gpu_triangle> triangles;
        std::vector<int> leaf_prim_indices(N);
        std::vector<int> prim_types(N);
        std::map<const material*, int> mat_to_idx;

        for (int i = 0; i < N; i++) {
            if (auto* sph = dynamic_cast<const sphere*>(objects[i].get())) {
                prim_types[i] = 0;
                leaf_prim_indices[i] = spheres.size();

                gpu_sphere gs;
                point3 c = sph->center_at(0);
                gs.center = {(float)c.x(), (float)c.y(), (float)c.z()};
                gs.radius = sph->get_radius();
                gs.mat_idx = get_or_add_material(sph->get_material_ptr(), mat_to_idx);
                spheres.push_back(gs);
            } else if (auto* tri = dynamic_cast<const triangle*>(objects[i].get())) {
                prim_types[i] = 1;
                leaf_prim_indices[i] = triangles.size();

                gpu_triangle gt;
                point3 v0 = tri->get_v0(), v1 = tri->get_v1(), v2 = tri->get_v2();
                gt.v0 = {(float)v0.x(), (float)v0.y(), (float)v0.z()};
                gt.v1 = {(float)v1.x(), (float)v1.y(), (float)v1.z()};
                gt.v2 = {(float)v2.x(), (float)v2.y(), (float)v2.z()};

                vec3 n0 = tri->get_n0(), n1 = tri->get_n1(), n2 = tri->get_n2();
                gt.n0 = {(float)n0.x(), (float)n0.y(), (float)n0.z()};
                gt.n1 = {(float)n1.x(), (float)n1.y(), (float)n1.z()};
                gt.n2 = {(float)n2.x(), (float)n2.y(), (float)n2.z()};

                gt.smooth_shading = tri->uses_smooth_shading() ? 1 : 0;
                gt.mat_idx = get_or_add_material(tri->get_material_ptr(), mat_to_idx);
                gt._pad0 = gt._pad1 = 0;
                triangles.push_back(gt);
            } else {
                prim_types[i] = -1;
                leaf_prim_indices[i] = -1;
            }
        }

        buf_spheres = spheres.empty()
            ? clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(gpu_sphere), nullptr, &err)
            : clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             spheres.size() * sizeof(gpu_sphere), spheres.data(), &err);

        buf_triangles = triangles.empty()
            ? clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(gpu_triangle), nullptr, &err)
            : clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             triangles.size() * sizeof(gpu_triangle), triangles.data(), &err);

        buf_leaf_prim_indices = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               N * sizeof(int), leaf_prim_indices.data(), &err);

        buf_prim_types = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        N * sizeof(int), prim_types.data(), &err);

        buf_materials = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       materials.size() * sizeof(gpu_material), materials.data(), &err);
    }

    int get_or_add_material(const material* mat_ptr, std::map<const material*, int>& mat_to_idx) {
        if (mat_to_idx.count(mat_ptr)) return mat_to_idx[mat_ptr];

        int idx = materials.size();
        mat_to_idx[mat_ptr] = idx;

        gpu_material gm = {0, {1,1,1}, 0, 0};

        if (auto* lamb = dynamic_cast<const lambertian*>(mat_ptr)) {
            gm.type = MAT_LAMBERTIAN;
            color alb = lamb->get_albedo();
            gm.albedo = {(float)alb.x(), (float)alb.y(), (float)alb.z()};
        } else if (auto* met = dynamic_cast<const metal*>(mat_ptr)) {
            gm.type = MAT_METAL;
            color alb = met->get_albedo();
            gm.albedo = {(float)alb.x(), (float)alb.y(), (float)alb.z()};
            gm.fuzz = met->get_fuzz();
        } else if (auto* light = dynamic_cast<const diffuse_light*>(mat_ptr)) {
            gm.type = MAT_LIGHT;
            color emit = light->get_emit();
            gm.albedo = {(float)emit.x(), (float)emit.y(), (float)emit.z()};
        }

        materials.push_back(gm);
        return idx;
    }

    void cleanup_opencl() {
        if (kernel_render_primary) clReleaseKernel(kernel_render_primary);
        if (kernel_render_sampling) clReleaseKernel(kernel_render_sampling);
        if (buf_nodes) clReleaseMemObject(buf_nodes);
        if (buf_spheres) clReleaseMemObject(buf_spheres);
        if (buf_triangles) clReleaseMemObject(buf_triangles);
        if (buf_leaf_prim_indices) clReleaseMemObject(buf_leaf_prim_indices);
        if (buf_prim_types) clReleaseMemObject(buf_prim_types);
        if (buf_materials) clReleaseMemObject(buf_materials);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    std::string load_kernel_source(const std::vector<const char*>& files) const {
        std::stringstream ss;
        for (const char* f : files) {
            std::ifstream file(f);
            if (!file.is_open()) { std::cerr << "Failed to open: " << f << std::endl; exit(1); }
            ss << file.rdbuf() << "\n";
        }
        return ss.str();
    }
};

template<typename BVH>
void gpu_renderer<BVH>::render(const camera_params& cam, std::vector<color>& output_pixels) {
    cl_int err;

    struct { float x, y, z; } cam_center = {cam.center_x, cam.center_y, cam.center_z};
    struct { float x, y, z; } pixel00 = {cam.pixel00_x, cam.pixel00_y, cam.pixel00_z};
    struct { float x, y, z; } delta_u = {cam.delta_u_x, cam.delta_u_y, cam.delta_u_z};
    struct { float x, y, z; } delta_v = {cam.delta_v_x, cam.delta_v_y, cam.delta_v_z};
    struct { float x, y, z; } defocus_u = {cam.defocus_u_x, cam.defocus_u_y, cam.defocus_u_z};
    struct { float x, y, z; } defocus_v = {cam.defocus_v_x, cam.defocus_v_y, cam.defocus_v_z};
    struct { float x, y, z; } bg = {cam.bg_r, cam.bg_g, cam.bg_b};

    size_t num_pixels = cam.width * cam.height;

    cl_mem buf_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_pixels * 3 * sizeof(float), nullptr, &err);
    cl_mem buf_box_tests = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_pixels * sizeof(int), nullptr, &err);
    cl_mem buf_prim_tests = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_pixels * sizeof(int), nullptr, &err);

    int arg = 0;
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cam_center), &cam_center);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(pixel00), &pixel00);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(delta_u), &delta_u);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(delta_v), &delta_v);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(defocus_u), &defocus_u);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(defocus_v), &defocus_v);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(float), &cam.defocus_angle);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &cam.width);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &cam.height);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &cam.samples);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &cam.max_depth);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(bg), &bg);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_nodes);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_spheres);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_triangles);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_leaf_prim_indices);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_prim_types);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_materials);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(int), &N);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_output);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_box_tests);
    clSetKernelArg(kernel_render_sampling, arg++, sizeof(cl_mem), &buf_prim_tests);

    size_t global[2] = {(size_t)cam.width, (size_t)cam.height};

    clEnqueueNDRangeKernel(queue, kernel_render_sampling, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    std::vector<float> pixel_data(num_pixels * 3);
    clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, pixel_data.size() * sizeof(float), pixel_data.data(), 0, nullptr, nullptr);

    output_pixels.resize(num_pixels);
    for (size_t i = 0; i < num_pixels; i++) {
        output_pixels[i] = color(pixel_data[i*3], pixel_data[i*3+1], pixel_data[i*3+2]);
    }

    std::vector<int> box_tests_data(num_pixels), prim_tests_data(num_pixels);
    clEnqueueReadBuffer(queue, buf_box_tests, CL_TRUE, 0, num_pixels * sizeof(int), box_tests_data.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, buf_prim_tests, CL_TRUE, 0, num_pixels * sizeof(int), prim_tests_data.data(), 0, nullptr, nullptr);

    long long total_box = 0, total_prim = 0;
    for (size_t i = 0; i < num_pixels; i++) {
        total_box += box_tests_data[i];
        total_prim += prim_tests_data[i];
    }

    last_total_rays = (long long)num_pixels * cam.samples;
    last_avg_box_tests = (double)total_box / last_total_rays;
    last_avg_prim_tests = (double)total_prim / last_total_rays;

    clReleaseMemObject(buf_output);
    clReleaseMemObject(buf_box_tests);
    clReleaseMemObject(buf_prim_tests);
}

#endif
