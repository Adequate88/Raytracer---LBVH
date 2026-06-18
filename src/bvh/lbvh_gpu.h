#pragma once

#include "aabb.h"
#include "bvh_utils.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "types.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct int2 {
  int x, y;
  int2() : x(0), y(0) {}
  int2(int x, int y) : x(x), y(y) {}
};

struct gpu_float3 {
  float x, y, z;
  gpu_float3() : x(0), y(0), z(0) {}
  gpu_float3(float x, float y, float z) : x(x), y(y), z(z) {}
  gpu_float3(const point3 &p)
      : x((float)p.x()), y((float)p.y()), z((float)p.z()) {}
};

struct lbvh_node {
  AABB bbox;
  int left;
  int right;
  int parent;
  int primitive_id;
  int atomic_counter;
  int _pad; // Padding to ensure 48-byte alignment matches OpenCL

  lbvh_node()
      : left(-1), right(-1), parent(-1), primitive_id(-1), atomic_counter(0),
        _pad(0) {}
  lbvh_node(const AABB &bbox, int prim_id)
      : bbox(bbox), left(-1), right(-1), parent(-1), primitive_id(prim_id),
        atomic_counter(0), _pad(0) {}
  lbvh_node(int l, int r)
      : left(l), right(r), parent(-1), primitive_id(-1), atomic_counter(0),
        _pad(0) {}
  bool is_leaf() const { return primitive_id >= 0; }
};

class LBVH {
public:
  ~LBVH();
  LBVH(std::vector<shared_ptr<hittable>> &objects);

  bool hit(const ray &r, interval ray_t, hit_record &rec) const;

  int get_tree_depth() const;

  cl_context get_opencl_context() const;
  cl_device_id get_opencl_device() const;
  cl_command_queue get_opencl_queue() const;
  cl_mem get_nodes_buffer() const;
  const std::vector<lbvh_node> &get_nodes() const;
  int get_primitive_count() const;
  const std::vector<shared_ptr<hittable>> &get_objects() const;

  // Host-side phase timings (ms), to mirror the Vulkan metric split:
  //   device init  -> "Vulkan Device Init"  (analog of engine.init())
  //   bvh init     -> "BVH Initialization"  (program build + kernels + buffers)
  //   construction -> "Total BVH Construction Time" (bvh init + build, excl dev init)
  double get_device_init_ms() const;
  double get_bvh_init_ms() const;
  double get_construction_ms() const;

private:
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;

  cl_kernel kernel_morton = nullptr;
  cl_kernel kernel_init_leaves = nullptr;
  cl_kernel kernel_hierarchy = nullptr;
  cl_kernel kernel_build_bboxes = nullptr;
  cl_kernel kernel_create_histogram = nullptr;
  cl_kernel kernel_prefix_sum = nullptr;
  cl_kernel kernel_scatter = nullptr;
  cl_kernel kernel_scan_glob_sum = nullptr;
  cl_kernel kernel_add_glob_sums = nullptr;

  cl_mem buf_centroids = nullptr;
  cl_mem buf_morton_list = nullptr;
  cl_mem buf_bboxes = nullptr;
  cl_mem buf_nodes = nullptr;
  cl_mem buf_histogram = nullptr;
  cl_mem buf_scanned_gram = nullptr;
  cl_mem buf_glob_sum = nullptr;
  cl_mem buf_morton_list_out = nullptr;

  std::vector<shared_ptr<hittable>> &objects;
  int N = 0;
  std::vector<lbvh_node> nodes;
  std::vector<AABB> primitive_bboxes;
  std::vector<point3> centroids;
  AABB temp_scene_bbox;
  std::vector<morton_primitive> morton_list;
  uint32_t k = 10;
  uint32_t cell_count = 0;

  double t_device_init_ms = 0.0;
  double t_bvh_init_ms = 0.0;
  double t_construction_ms = 0.0;

  bool hit_recursive(const ray &r, interval ray_t, hit_record &rec,
                     const int node_idx) const;

  // Returns total device time (ms) of the radix-sort kernels.
  float radix_sort();

  int compute_depth(int node_idx) const;

  void check_cl_error(cl_int err, const char *operation) const;

  // Return pure device execution time (ms) from OpenCL profiling events.
  float record_kernel_time(cl_event event) const;
  float record_kernel_time(const std::vector<cl_event> &events) const;

  void init_device();                // platform/device/context/queue
  void build_program_and_kernels();  // program build + clCreateKernel
  void cleanup_opencl();

  std::string load_kernel_source_with_includes(
      const std::vector<const char *> &filenames) const;
};
