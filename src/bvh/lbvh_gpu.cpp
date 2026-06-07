#include "lbvh_gpu.h"
#include "metrics_macros.h"

LBVH::LBVH(std::vector<shared_ptr<hittable>> &objects)
    : objects(objects), N(objects.size()) {
  nodes.resize(2 * N - 1);
  METRIC_START_TIME("HOST_BVH_TOTAL");

  if (N == 0)
    return;

  METRIC_START_TIME("HOST_OPENCL_INIT");
  init_opencl();
  METRIC_END_TIME("HOST_OPENCL_INIT");

  METRIC_START_TIME("HOST_DATA_PREP");
  morton_list.resize(N);
  cell_count = 1 << k;

  temp_scene_bbox = AABB::empty;
  for (int i = 0; i < N; i++) {
    temp_scene_bbox = AABB(temp_scene_bbox, objects[i]->bounding_box());
    primitive_bboxes.push_back(objects[i]->bounding_box());
    centroids.push_back(objects[i]->get_centroid());
  }

  std::vector<gpu_float3> gpu_centroids(N);
  for (int i = 0; i < N; i++) {
    gpu_centroids[i] = gpu_float3(centroids[i]);
  }

  cl_int err;
  buf_centroids =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     N * sizeof(gpu_float3), gpu_centroids.data(), &err);
  check_cl_error(err, "clCreateBuffer(buf_centroids)");

  buf_bboxes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              N * sizeof(AABB), primitive_bboxes.data(), &err);
  check_cl_error(err, "clCreateBuffer(buf_bboxes)");

  buf_morton_list = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   N * sizeof(morton_primitive), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_morton_list)");

  buf_nodes = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             (2 * N - 1) * sizeof(lbvh_node), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_nodes)");
  METRIC_END_TIME("HOST_DATA_PREP");

  METRIC_START_TIME("HOST_MORTON");
  err = clSetKernelArg(kernel_morton, 0, sizeof(cl_mem), &buf_centroids);
  err |= clSetKernelArg(kernel_morton, 1, sizeof(AABB), &temp_scene_bbox);
  err |= clSetKernelArg(kernel_morton, 2, sizeof(int), &N);
  err |= clSetKernelArg(kernel_morton, 3, sizeof(uint32_t), &cell_count);
  err |= clSetKernelArg(kernel_morton, 4, sizeof(uint32_t), &k);
  err |= clSetKernelArg(kernel_morton, 5, sizeof(cl_mem), &buf_morton_list);
  check_cl_error(err, "clSetKernelArg(kernel_morton)");

  size_t global_work_size = N;
  cl_event ev_morton;
  err = clEnqueueNDRangeKernel(queue, kernel_morton, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr,
                               &ev_morton);
  check_cl_error(err, "clEnqueueNDRangeKernel(kernel_morton)");
  clFinish(queue);
  record_kernel_time(ev_morton, "GPU_MORTON");

  err = clEnqueueReadBuffer(queue, buf_morton_list, CL_TRUE, 0,
                            N * sizeof(morton_primitive), morton_list.data(), 0,
                            nullptr, nullptr);
  check_cl_error(err, "clEnqueueReadBuffer(buf_morton_list)");
  METRIC_END_TIME("HOST_MORTON");

  METRIC_START_TIME("HOST_RADIX_SORT");
  radix_sort();
  err = clEnqueueReadBuffer(queue, buf_morton_list, CL_TRUE, 0,
                            N * sizeof(morton_primitive), morton_list.data(), 0,
                            nullptr, nullptr);
  check_cl_error(err, "clEnqueueReadBuffer(sorted morton_list)");
  METRIC_END_TIME("HOST_RADIX_SORT");

  METRIC_START_TIME("HOST_HIERARCHY");
  err = clSetKernelArg(kernel_init_leaves, 0, sizeof(cl_mem), &buf_morton_list);
  err |= clSetKernelArg(kernel_init_leaves, 1, sizeof(cl_mem), &buf_bboxes);
  err |= clSetKernelArg(kernel_init_leaves, 2, sizeof(int), &N);
  err |= clSetKernelArg(kernel_init_leaves, 3, sizeof(cl_mem), &buf_nodes);
  check_cl_error(err, "clSetKernelArg(kernel_init_leaves)");

  cl_event ev_init_leaves;
  err = clEnqueueNDRangeKernel(queue, kernel_init_leaves, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr,
                               &ev_init_leaves);
  check_cl_error(err, "clEnqueueNDRangeKernel(kernel_init_leaves)");
  clFinish(queue);
  record_kernel_time(ev_init_leaves, "GPU_INIT_LEAVES");

  err = clSetKernelArg(kernel_hierarchy, 0, sizeof(cl_mem), &buf_morton_list);
  err |= clSetKernelArg(kernel_hierarchy, 1, sizeof(int), &N);
  err |= clSetKernelArg(kernel_hierarchy, 2, sizeof(cl_mem), &buf_nodes);
  check_cl_error(err, "clSetKernelArg(kernel_hierarchy)");

  size_t hierarchy_work_size = N - 1;
  cl_event ev_hierarchy;
  err = clEnqueueNDRangeKernel(queue, kernel_hierarchy, 1, nullptr,
                               &hierarchy_work_size, nullptr, 0, nullptr,
                               &ev_hierarchy);
  check_cl_error(err, "clEnqueueNDRangeKernel(kernel_hierarchy)");
  clFinish(queue);
  record_kernel_time(ev_hierarchy, "GPU_HIERARCHY");

  err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                            (2 * N - 1) * sizeof(lbvh_node), nodes.data(), 0,
                            nullptr, nullptr);
  check_cl_error(err, "clEnqueueReadBuffer(buf_nodes)");
  METRIC_END_TIME("HOST_HIERARCHY");

  METRIC_START_TIME("HOST_BBOX");
  err = clSetKernelArg(kernel_build_bboxes, 0, sizeof(cl_mem), &buf_nodes);
  err |= clSetKernelArg(kernel_build_bboxes, 1, sizeof(int), &N);
  check_cl_error(err, "clSetKernelArg(kernel_build_bboxes)");

  cl_event ev_build_bboxes;
  err = clEnqueueNDRangeKernel(queue, kernel_build_bboxes, 1, NULL,
                               &global_work_size, NULL, 0, NULL,
                               &ev_build_bboxes);
  check_cl_error(err, "clEnqueueNDRangeKernel(kernel_build_bboxes)");
  clFinish(queue);
  record_kernel_time(ev_build_bboxes, "GPU_BBOX");

  err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                            (2 * N - 1) * sizeof(lbvh_node), nodes.data(), 0,
                            nullptr, nullptr);
  check_cl_error(err, "clEnqueueReadBuffer(buf_nodes after bbox)");
  METRIC_END_TIME("HOST_BBOX");

  METRIC_END_TIME("HOST_BVH_TOTAL");
}

LBVH::~LBVH() { cleanup_opencl(); }

bool LBVH::hit(const ray &r, interval ray_t, hit_record &rec) const {
  return hit_recursive(r, ray_t, rec, 0);
}

int LBVH::get_tree_depth() const { return compute_depth(0); }

cl_context LBVH::get_opencl_context() const { return context; }
cl_device_id LBVH::get_opencl_device() const { return device; }
cl_command_queue LBVH::get_opencl_queue() const { return queue; }
cl_mem LBVH::get_nodes_buffer() const { return buf_nodes; }
const std::vector<lbvh_node> &LBVH::get_nodes() const { return nodes; }
int LBVH::get_primitive_count() const { return N; }
const std::vector<shared_ptr<hittable>> &LBVH::get_objects() const {
  return objects;
}

bool LBVH::hit_recursive(const ray &r, interval ray_t, hit_record &rec,
                         const int node_idx) const {
  if (node_idx >= 2 * N - 1 || node_idx < 0)
    return false;
  METRIC_INCREMENT("NODES_VISITED");

  const lbvh_node &node = nodes[node_idx];
  if (!node.bbox.hit(r, ray_t))
    return false;

  if (node.is_leaf()) {
    METRIC_INCREMENT("PRIMITIVE_TESTS");
    return objects[node.primitive_id]->hit(r, ray_t, rec);
  }

  bool hit_left = hit_recursive(r, ray_t, rec, node.left);
  bool hit_right = hit_recursive(
      r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, node.right);
  return hit_left || hit_right;
}

void LBVH::radix_sort() {
  cl_int err;
  const int num_buckets = 16;
  const int num_passes = 8;
  const int histo_block_size = 1024;
  const int histo_threads = 256;
  const int num_groups = (N + histo_block_size - 1) / histo_block_size;
  const int histogram_size = num_buckets * num_groups * histo_threads;
  const int prefix_threads = 256;
  const int prefix_elements_per_wg = 512;
  const int num_prefix_wgs =
      (histogram_size + prefix_elements_per_wg - 1) / prefix_elements_per_wg;

  buf_histogram =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     histogram_size * sizeof(uint32_t), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_histogram)");
  buf_scanned_gram =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     histogram_size * sizeof(uint32_t), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_scanned_gram)");
  buf_glob_sum =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     num_prefix_wgs * sizeof(uint32_t), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_glob_sum)");
  buf_morton_list_out = clCreateBuffer(
      context, CL_MEM_READ_WRITE, N * sizeof(morton_primitive), nullptr, &err);
  check_cl_error(err, "clCreateBuffer(buf_morton_list_out)");

  cl_mem buf_in = buf_morton_list;
  cl_mem buf_out = buf_morton_list_out;

  std::vector<cl_event> ev_histogram, ev_prefix, ev_scan, ev_add, ev_scatter;

  for (int pass = 0; pass < num_passes; pass++) {
    size_t histo_local = histo_threads;
    size_t histo_global = num_groups * histo_threads;

    err = clSetKernelArg(kernel_create_histogram, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel_create_histogram, 1, sizeof(int), &N);
    err |= clSetKernelArg(kernel_create_histogram, 2, sizeof(int), &pass);
    err |= clSetKernelArg(kernel_create_histogram, 3, sizeof(cl_mem),
                          &buf_histogram);
    check_cl_error(err, "clSetKernelArg(create_histogram)");
    cl_event ev_h;
    err = clEnqueueNDRangeKernel(queue, kernel_create_histogram, 1, nullptr,
                                 &histo_global, &histo_local, 0, nullptr,
                                 &ev_h);
    check_cl_error(err, "clEnqueueNDRangeKernel(create_histogram)");
    ev_histogram.push_back(ev_h);

    size_t prefix_local = prefix_threads;
    size_t prefix_global = num_prefix_wgs * prefix_threads;

    err = clSetKernelArg(kernel_prefix_sum, 0, sizeof(cl_mem), &buf_histogram);
    err |=
        clSetKernelArg(kernel_prefix_sum, 1, sizeof(cl_mem), &buf_scanned_gram);
    err |= clSetKernelArg(kernel_prefix_sum, 2, sizeof(cl_mem), &buf_glob_sum);
    err |= clSetKernelArg(kernel_prefix_sum, 3, sizeof(int), &num_groups);
    err |= clSetKernelArg(kernel_prefix_sum, 4, sizeof(int), &histogram_size);
    check_cl_error(err, "clSetKernelArg(prefix_sum)");
    cl_event ev_p;
    err = clEnqueueNDRangeKernel(queue, kernel_prefix_sum, 1, nullptr,
                                 &prefix_global, &prefix_local, 0, nullptr,
                                 &ev_p);
    check_cl_error(err, "clEnqueueNDRangeKernel(prefix_sum)");
    ev_prefix.push_back(ev_p);

    if (num_prefix_wgs > 1) {
      size_t scan_local = 1;
      while (scan_local < (size_t)num_prefix_wgs)
        scan_local <<= 1;
      size_t scan_global = scan_local;
      const size_t max_local_size = 256;

      if (scan_local <= max_local_size) {
        err = clSetKernelArg(kernel_scan_glob_sum, 0, sizeof(cl_mem),
                             &buf_glob_sum);
        err |= clSetKernelArg(kernel_scan_glob_sum, 1,
                              scan_local * sizeof(cl_uint), nullptr);
        err |= clSetKernelArg(kernel_scan_glob_sum, 2, sizeof(int),
                              &num_prefix_wgs);
        check_cl_error(err, "clSetKernelArg(scan_glob_sum)");
        cl_event ev_s;
        err = clEnqueueNDRangeKernel(queue, kernel_scan_glob_sum, 1, nullptr,
                                     &scan_global, &scan_local, 0, nullptr,
                                     &ev_s);
        check_cl_error(err, "clEnqueueNDRangeKernel(scan_glob_sum)");
        ev_scan.push_back(ev_s);
      } else {
        clFinish(queue);
        std::vector<uint32_t> glob_sum_host(num_prefix_wgs);
        err = clEnqueueReadBuffer(queue, buf_glob_sum, CL_TRUE, 0,
                                  num_prefix_wgs * sizeof(uint32_t),
                                  glob_sum_host.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(glob_sum fallback)");
        uint32_t running_sum = 0;
        for (int i = 0; i < num_prefix_wgs; i++) {
          uint32_t temp = glob_sum_host[i];
          glob_sum_host[i] = running_sum;
          running_sum += temp;
        }
        err = clEnqueueWriteBuffer(queue, buf_glob_sum, CL_TRUE, 0,
                                   num_prefix_wgs * sizeof(uint32_t),
                                   glob_sum_host.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueWriteBuffer(glob_sum fallback)");
      }

      err = clSetKernelArg(kernel_add_glob_sums, 0, sizeof(cl_mem),
                           &buf_glob_sum);
      err |= clSetKernelArg(kernel_add_glob_sums, 1, sizeof(cl_mem),
                            &buf_scanned_gram);
      err |=
          clSetKernelArg(kernel_add_glob_sums, 2, sizeof(int), &histogram_size);
      check_cl_error(err, "clSetKernelArg(add_glob_sums)");
      cl_event ev_a;
      err = clEnqueueNDRangeKernel(queue, kernel_add_glob_sums, 1, nullptr,
                                   &prefix_global, &prefix_local, 0, nullptr,
                                   &ev_a);
      check_cl_error(err, "clEnqueueNDRangeKernel(add_glob_sums)");
      ev_add.push_back(ev_a);
    }

    err = clSetKernelArg(kernel_scatter, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel_scatter, 1, sizeof(cl_mem), &buf_scanned_gram);
    err |= clSetKernelArg(kernel_scatter, 2, sizeof(int), &N);
    err |= clSetKernelArg(kernel_scatter, 3, sizeof(int), &pass);
    err |= clSetKernelArg(kernel_scatter, 4, sizeof(cl_mem), &buf_out);
    check_cl_error(err, "clSetKernelArg(scatter)");
    cl_event ev_sc;
    err =
        clEnqueueNDRangeKernel(queue, kernel_scatter, 1, nullptr, &histo_global,
                               &histo_local, 0, nullptr, &ev_sc);
    check_cl_error(err, "clEnqueueNDRangeKernel(scatter)");
    ev_scatter.push_back(ev_sc);

    std::swap(buf_in, buf_out);
  }
  clFinish(queue);

  record_kernel_time(ev_histogram, "GPU_RADIX_HISTOGRAM");
  record_kernel_time(ev_prefix, "GPU_RADIX_PREFIX");
  record_kernel_time(ev_scan, "GPU_RADIX_SCAN");
  record_kernel_time(ev_add, "GPU_RADIX_ADD");
  record_kernel_time(ev_scatter, "GPU_RADIX_SCATTER");
}

int LBVH::compute_depth(int node_idx) const {
  if (node_idx < 0)
    return 0;
  const lbvh_node &node = nodes[node_idx];
  if (node.is_leaf())
    return 1;
  return 1 + std::max(compute_depth(node.left), compute_depth(node.right));
}

void LBVH::check_cl_error(cl_int err, const char *operation) const {
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL error during " << operation << ": " << err
              << std::endl;
    exit(1);
  }
}

void LBVH::record_kernel_time(cl_event event, const char *name) const {
  clWaitForEvents(1, &event);

  cl_ulong start_ns = 0;
  cl_ulong end_ns = 0;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &start_ns, nullptr);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end_ns, nullptr);

  // Convert from nano to ms
  METRIC_SET_VALUE(name, static_cast<float>(end_ns - start_ns) * 1e-6f);

  clReleaseEvent(event);
}

void LBVH::record_kernel_time(const std::vector<cl_event> &events,
                              const char *name) const {
  if (events.empty())
    return;

  cl_ulong total_ns = 0;
  for (cl_event event : events) {
    clWaitForEvents(1, &event);
    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                            &start_ns, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &end_ns, nullptr);
    total_ns += end_ns - start_ns;
    clReleaseEvent(event);
  }

  // Total device time for this kernel summed over all radix passes, in ms.
  METRIC_SET_VALUE(name, static_cast<float>(total_ns) * 1e-6f);
}

void LBVH::init_opencl() {
  cl_int err;
  err = clGetPlatformIDs(1, &platform, nullptr);
  check_cl_error(err, "clGetPlatformIDs");
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
  check_cl_error(err, "clGetDeviceIDs");
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  check_cl_error(err, "clCreateContext");
  queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  check_cl_error(err, "clCreateCommandQueue");

  std::string source = load_kernel_source_with_includes(
      {"src/bvh/opencl_shared_types.cl",
       "src/bvh/lbvh_construction_kernels.cl"});
  const char *src_ptr = source.c_str();
  size_t src_len = source.length();

  program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
  check_cl_error(err, "clCreateProgramWithSource");
  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
    std::cerr << "OpenCL kernel compilation error:\n"
              << log.data() << std::endl;
    exit(1);
  }

  kernel_morton = clCreateKernel(program, "compute_morton", &err);
  check_cl_error(err, "clCreateKernel(compute_morton)");
  kernel_init_leaves = clCreateKernel(program, "init_leaf_nodes", &err);
  check_cl_error(err, "clCreateKernel(init_leaf_nodes)");
  kernel_hierarchy = clCreateKernel(program, "create_hierarchy", &err);
  check_cl_error(err, "clCreateKernel(create_hierarchy)");
  kernel_build_bboxes = clCreateKernel(program, "build_bboxes", &err);
  check_cl_error(err, "clCreateKernel(build_bboxes)");
  kernel_create_histogram = clCreateKernel(program, "create_histogram", &err);
  check_cl_error(err, "clCreateKernel(create_histogram)");
  kernel_prefix_sum = clCreateKernel(program, "prefix_sum", &err);
  check_cl_error(err, "clCreateKernel(prefix_sum)");
  kernel_scatter = clCreateKernel(program, "scatter", &err);
  check_cl_error(err, "clCreateKernel(scatter)");
  kernel_scan_glob_sum = clCreateKernel(program, "scan_glob_sum", &err);
  check_cl_error(err, "clCreateKernel(scan_glob_sum)");
  kernel_add_glob_sums = clCreateKernel(program, "add_glob_sums", &err);
  check_cl_error(err, "clCreateKernel(add_glob_sums)");
}

void LBVH::cleanup_opencl() {
  if (kernel_morton)
    clReleaseKernel(kernel_morton);
  if (kernel_init_leaves)
    clReleaseKernel(kernel_init_leaves);
  if (kernel_hierarchy)
    clReleaseKernel(kernel_hierarchy);
  if (kernel_build_bboxes)
    clReleaseKernel(kernel_build_bboxes);
  if (kernel_create_histogram)
    clReleaseKernel(kernel_create_histogram);
  if (kernel_prefix_sum)
    clReleaseKernel(kernel_prefix_sum);
  if (kernel_scatter)
    clReleaseKernel(kernel_scatter);
  if (kernel_scan_glob_sum)
    clReleaseKernel(kernel_scan_glob_sum);
  if (kernel_add_glob_sums)
    clReleaseKernel(kernel_add_glob_sums);
  if (buf_centroids)
    clReleaseMemObject(buf_centroids);
  if (buf_morton_list)
    clReleaseMemObject(buf_morton_list);
  if (buf_bboxes)
    clReleaseMemObject(buf_bboxes);
  if (buf_nodes)
    clReleaseMemObject(buf_nodes);
  if (buf_histogram)
    clReleaseMemObject(buf_histogram);
  if (buf_scanned_gram)
    clReleaseMemObject(buf_scanned_gram);
  if (buf_glob_sum)
    clReleaseMemObject(buf_glob_sum);
  if (buf_morton_list_out)
    clReleaseMemObject(buf_morton_list_out);
  if (program)
    clReleaseProgram(program);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
}

std::string LBVH::load_kernel_source_with_includes(
    const std::vector<const char *> &filenames) const {
  std::stringstream combined;
  for (const char *filename : filenames) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open kernel file: " << filename << std::endl;
      exit(1);
    }
    combined << file.rdbuf() << "\n";
  }
  return combined.str();
}
