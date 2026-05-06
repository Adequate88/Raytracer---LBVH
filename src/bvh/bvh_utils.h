#include <cstdint>

struct morton_primitive {
  uint32_t morton_code;
  uint32_t primitive_id;
};

struct BVHStats {
  long long node_visits = 0;
  long long primitive_tests = 0;
  long long rays_traced = 0;
  
  void reset();
  
  void print() const;
};

