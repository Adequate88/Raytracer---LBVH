#include "bvh_utils.h"
#include <iostream>

void BVHStats::reset() {
  node_visits = 0;
  primitive_tests = 0;
  rays_traced = 0;
}
  
void BVHStats::print() const {
  std::clog << "\n=== BVH Statistics ===\n";
  std::clog << "  Total rays traced: " << rays_traced << "\n";
  if (rays_traced > 0) {
      std::clog << "  Avg node visits/ray: " << (double)node_visits / rays_traced << "\n";
      std::clog << "  Avg primitive tests/ray: " << (double)primitive_tests / rays_traced << "\n";
  }
  std::clog << "======================\n";
}

