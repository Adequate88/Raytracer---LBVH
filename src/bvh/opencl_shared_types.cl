typedef struct {
  float x, y, z;
} point3;

typedef struct {
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
} aabb;

typedef struct {
  uint morton_code;
  int primitive_id;
} morton_primitive;

typedef struct {
  aabb bbox;
  int left;
  int right;
  int parent;
  int primitive_id;
  volatile int atomic_counter;  // For bottom-up bbox construction
  int _pad;  // Padding to ensure 48-byte alignment matches C++
} lbvh_node;

typedef struct {
  point3 origin;
  point3 direction;
} ray;

typedef struct {
  point3 p;
  point3 normal;
  float t;
  int prim_type;
  int prim_idx;
  int mat_idx;
  bool front_face;
} hit_record;

typedef struct {
  float radius;
  point3 center;
  int mat_idx;
} sphere;

typedef struct {
  point3 v0, v1, v2;
  point3 n0, n1, n2;
  int mat_idx;
  int smooth_shading;
  float _pad0, _pad1;
} gpu_triangle;

typedef enum {
  MAT_LAMBERTIAN = 0,
  MAT_METAL = 1,
  MAT_LIGHT = 2
} material_type;

typedef struct {
  material_type type;
  point3 albedo;
  float fuzz;
  float _pad0;
} gpu_material;

inline float length_squared(point3 vec) {
  return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

inline float dot(point3 vec1, point3 vec2) {
  return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

inline point3 ray_at(ray r, float t) {
  point3 new_point;
  new_point.x = r.origin.x + t * r.direction.x;
  new_point.y = r.origin.y + t * r.direction.y;
  new_point.z = r.origin.z + t * r.direction.z;
  return new_point;
}

inline float length(point3 v) {
  return sqrt(length_squared(v));
}

inline point3 normalize(point3 v) {
  float len = length(v);
  point3 result;
  result.x = v.x / len;
  result.y = v.y / len;
  result.z = v.z / len;
  return result;
}

inline point3 vec3_add(point3 a, point3 b) {
  point3 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

inline point3 vec3_sub(point3 a, point3 b) {
  point3 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

inline point3 vec3_scale(point3 v, float s) {
  point3 result;
  result.x = v.x * s;
  result.y = v.y * s;
  result.z = v.z * s;
  return result;
}

inline point3 vec3_mul(point3 a, point3 b) {
  point3 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  result.z = a.z * b.z;
  return result;
}

inline point3 cross(point3 a, point3 b) {
  point3 result;
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;
  return result;
}

inline point3 reflect(point3 v, point3 n) {
  float d = dot(v, n);
  point3 result;
  result.x = v.x - 2*d*n.x;
  result.y = v.y - 2*d*n.y;
  result.z = v.z - 2*d*n.z;
  return result;
}

inline bool near_zero(point3 v) {
  const float s = 1e-8f;
  return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

inline uint seed_rng(int pixel_x, int pixel_y, int sample_idx, int image_width) {
  uint seed = pixel_y * image_width + pixel_x;
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed = seed + (seed << 3u);
  seed = seed ^ (seed >> 4u);
  seed = seed * 0x27d4eb2du;
  seed = seed ^ (seed >> 15u);
  seed ^= sample_idx * 0x9e3779b9u;
  return seed;
}

inline uint lcg_step(uint* state) {
  *state = 1664525u * (*state) + 1013904223u;
  return *state;
}

inline float random_float(uint* state) {
  return (float)lcg_step(state) / 4294967296.0f;
}

inline float random_float_range(uint* state, float min, float max) {
  return min + (max - min) * random_float(state);
}

inline point3 random_vec3(uint* state) {
  point3 result;
  result.x = random_float_range(state, -1.0f, 1.0f);
  result.y = random_float_range(state, -1.0f, 1.0f);
  result.z = random_float_range(state, -1.0f, 1.0f);
  return result;
}

inline point3 random_unit_vector(uint* state) {
  for (int i = 0; i < 64; i++) {
    point3 p = random_vec3(state);
    float lensq = length_squared(p);
    if (lensq > 1e-20f && lensq <= 1.0f) {
      float len = sqrt(lensq);
      p.x /= len;
      p.y /= len;
      p.z /= len;
      return p;
    }
  }
  point3 fallback = {1.0f, 0.0f, 0.0f};
  return fallback;
}

inline point3 random_in_unit_disk(uint* state) {
  for (int i = 0; i < 64; i++) {
    point3 p;
    p.x = random_float_range(state, -1.0f, 1.0f);
    p.y = random_float_range(state, -1.0f, 1.0f);
    p.z = 0.0f;
    if (length_squared(p) < 1.0f)
      return p;
  }
  point3 fallback = {0.0f, 0.0f, 0.0f};
  return fallback;
}
