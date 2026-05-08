#pragma once

#include <iostream>


struct BVHStats {
  const std::string& fileName;

  // Count metrics
  long long nodesVisited = 0;
  long long primitiveTests = 0;
  long long raysTraced = 0;

  // Time metrics
  double construction_time_ms = 0.0;
  double opencl_init_time_ms = 0.0;
  double data_prep_time_ms = 0.0;
  double morton_time_ms = 0.0;
  double sort_time_ms = 0.0;
  double hierarchy_time_ms = 0.0;
  double bbox_time_ms = 0.0;




  void reset();
  void print() const;
};

void BVHStats::clear() {
  nodesVisited = 0;
  primitiveTests = 0;
  raysTraced = 0;
}
  
void BVHStats::export_stats() const 
{
  std::ofstream out(filename);

  if (!out.is_open())
  {
      std::cerr << "Failed to open: " << filename << std::endl;
      return;
  }

  out << std::fixed << std::setprecision(3);
  out.close();
}

void BVHStats::enable_statistics() const { enable_stats = true; stats.reset(); }
void BVHStats::disable_statistics() const { enable_stats = false; }
void BVHStats::increment_ray_count() const { if (enable_stats) stats.rays_traced++; }

// RENDER STATS

struct RenderStats {

    std::string scene_name_for_stats = "";

    long long last_total_rays = 0;
    double last_avg_box_tests = 0.0;
    double last_avg_prim_tests = 0.0;

    long long get_total_rays() const;
    double get_avg_box_tests() const;
    double get_avg_prim_tests() const; 

    long long camera::get_total_rays() const { return last_total_rays; }
    double camera::get_avg_box_tests() const { return last_avg_box_tests; }
    double camera::get_avg_prim_tests() const { return last_avg_prim_tests; }
}
