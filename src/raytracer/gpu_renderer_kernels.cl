//==============================================================================================
// Material Functions
//==============================================================================================

// Get emitted color from material (lights only)
inline point3 material_emitted(int mat_idx, __global const gpu_material* materials) {
  if (mat_idx < 0) {
    point3 black = {0.0f, 0.0f, 0.0f};
    return black;
  }

  gpu_material mat = materials[mat_idx];

  if (mat.type == MAT_LIGHT) {
    return mat.albedo;  // Lights emit their albedo color
  }

  point3 black = {0.0f, 0.0f, 0.0f};
  return black;
}

// Lambertian (diffuse) material scatter
inline bool scatter_lambertian(ray r_in, hit_record rec, gpu_material mat,
                                uint* rng_state, point3* attenuation, ray* scattered) {
  // Random diffuse direction
  point3 scatter_direction = vec3_add(rec.normal, random_unit_vector(rng_state));

  // Catch degenerate scatter direction
  if (near_zero(scatter_direction)) {
    scatter_direction = rec.normal;
  }

  // Scattered ray
  scattered->origin = rec.p;
  scattered->direction = scatter_direction;

  // Attenuation is albedo color
  *attenuation = mat.albedo;

  return true;
}

// Metal (reflective) material scatter
inline bool scatter_metal(ray r_in, hit_record rec, gpu_material mat,
                          uint* rng_state, point3* attenuation, ray* scattered) {
  // Reflect the ray
  point3 reflected = reflect(normalize(r_in.direction), rec.normal);

  // Add fuzz (roughness)
  point3 fuzz_offset = vec3_scale(random_unit_vector(rng_state), mat.fuzz);
  reflected = vec3_add(reflected, fuzz_offset);

  // Scattered ray
  scattered->origin = rec.p;
  scattered->direction = reflected;

  // Attenuation is albedo color
  *attenuation = mat.albedo;

  // Only scatter if reflected ray is above surface
  return (dot(scattered->direction, rec.normal) > 0);
}

// Light material (absorbs rays, doesn't scatter)
inline bool scatter_light(ray r_in, hit_record rec, gpu_material mat,
                          uint* rng_state, point3* attenuation, ray* scattered) {
  // Lights don't scatter - they absorb rays
  return false;
}

// Unified material scatter dispatch
inline bool scatter_material(ray r_in, hit_record rec,
                             __global const gpu_material* materials,
                             uint* rng_state, point3* attenuation, ray* scattered) {
  if (rec.mat_idx < 0) {
    return false;  // No material
  }

  gpu_material mat = materials[rec.mat_idx];

  switch (mat.type) {
    case MAT_LAMBERTIAN:
      return scatter_lambertian(r_in, rec, mat, rng_state, attenuation, scattered);

    case MAT_METAL:
      return scatter_metal(r_in, rec, mat, rng_state, attenuation, scattered);

    case MAT_LIGHT:
      return scatter_light(r_in, rec, mat, rng_state, attenuation, scattered);

    default:
      return false;  // Unknown material type
  }
}

// Ray-sphere intersection (adapted from sphere.h)
inline bool sphere_intersection(ray r, float t_min, float t_max, hit_record* rec, sphere _sphere) {
  point3 oc;
  oc.x = _sphere.center.x - r.origin.x;
  oc.y = _sphere.center.y - r.origin.y;
  oc.z = _sphere.center.z - r.origin.z;

  float a = length_squared(r.direction);
  float h = dot(r.direction, oc);
  float c = length_squared(oc) - _sphere.radius * _sphere.radius;

  float discriminant = h*h - a*c;
  if (discriminant < 0)
    return false;

  float sqrtd = sqrt(discriminant);
  float root = (h - sqrtd) / a;
  if (root >= t_max || root <= t_min) {
    root = (h + sqrtd) / a;
    if (root >= t_max || root <= t_min)
      return false;
  }

  rec->t = root;
  rec->p = ray_at(r, rec->t);

  point3 outward_normal;
  outward_normal.x = (rec->p.x - _sphere.center.x) / _sphere.radius;
  outward_normal.y = (rec->p.y - _sphere.center.y) / _sphere.radius;
  outward_normal.z = (rec->p.z - _sphere.center.z) / _sphere.radius;

  rec->front_face = dot(r.direction, outward_normal) < 0;

  if (rec->front_face) {
    rec->normal = outward_normal;
  } else {
    rec->normal.x = -outward_normal.x;
    rec->normal.y = -outward_normal.y;
    rec->normal.z = -outward_normal.z;
  }

  rec->mat_idx = _sphere.mat_idx;

  return true;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
inline bool triangle_intersection(
    ray r, float t_min, float t_max,
    hit_record* rec, gpu_triangle tri
) {
    const float EPSILON = 1e-8f;

    // Möller-Trumbore algorithm
    point3 edge1 = vec3_sub(tri.v1, tri.v0);
    point3 edge2 = vec3_sub(tri.v2, tri.v0);
    point3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);

    // Check if ray is parallel to triangle
    if (fabs(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    point3 s = vec3_sub(r.origin, tri.v0);
    float u = f * dot(s, h);

    // Check if intersection is outside triangle (barycentric u)
    if (u < 0.0f || u > 1.0f)
        return false;

    point3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);

    // Check if intersection is outside triangle (barycentric v)
    if (v < 0.0f || u + v > 1.0f)
        return false;

    // Compute t to find intersection point
    float t = f * dot(edge2, q);

    // Check if t is within ray interval
    if (t <= t_min || t >= t_max)
        return false;

    // We have a valid intersection - fill hit_record
    rec->t = t;
    rec->p = ray_at(r, t);

    // Compute normal (smooth or flat shading)
    point3 outward_normal;
    if (tri.smooth_shading) {
        // Interpolate vertex normals using barycentric coordinates
        // w = 1 - u - v (barycentric weight for v0)
        float w = 1.0f - u - v;
        outward_normal.x = w * tri.n0.x + u * tri.n1.x + v * tri.n2.x;
        outward_normal.y = w * tri.n0.y + u * tri.n1.y + v * tri.n2.y;
        outward_normal.z = w * tri.n0.z + u * tri.n1.z + v * tri.n2.z;
        outward_normal = normalize(outward_normal);
    } else {
        // Flat shading - use geometric normal stored in n0
        outward_normal = tri.n0;
    }

    // Determine front/back face
    rec->front_face = dot(r.direction, outward_normal) < 0;
    rec->normal = rec->front_face ? outward_normal : vec3_scale(outward_normal, -1.0f);
    rec->mat_idx = tri.mat_idx;

    return true;
}

inline bool aabb_hit(aabb box, ray r, float t_min, float t_max) {
  for (int axis = 0; axis < 3; axis++) {
    float ray_orig, ray_dir, box_min, box_max;

    if (axis == 0) {        
      ray_orig = r.origin.x;
      ray_dir = r.direction.x;
      box_min = box.min_x;
      box_max = box.max_x;
    } else if (axis == 1) { 
      ray_orig = r.origin.y;
      ray_dir = r.direction.y;
      box_min = box.min_y;
      box_max = box.max_y;
    } else {               
      ray_orig = r.origin.z;
      ray_dir = r.direction.z;
      box_min = box.min_z;
      box_max = box.max_z;
    }

    float adinv = 1.0f / ray_dir; 
    float t0 = (box_min - ray_orig) * adinv;
    float t1 = (box_max - ray_orig) * adinv;

    if (t0 < t1) {
      if (t0 > t_min) t_min = t0;
      if (t1 < t_max) t_max = t1;
    } else {
      if (t1 > t_min) t_min = t1;
      if (t0 < t_max) t_max = t0;
    }

    if (t_max <= t_min)
      return false;
  }
  return true;
}

//==============================================================================================
// Forward declaration for ray_color
//==============================================================================================

inline bool bvh_traverse(
  ray r, float t_min, float t_max,
  hit_record* rec,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const gpu_triangle* triangles,
  __global const int* leaf_prim_indices,
  __global const int* prim_types,
  int N
);

//==============================================================================================
// Iterative Ray Tracing (Path Tracing with Multiple Bounces)
//==============================================================================================

inline point3 ray_color(
  ray initial_ray, int max_depth, point3 background,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const gpu_triangle* triangles,
  __global const int* leaf_prim_indices,
  __global const int* prim_types,
  __global const gpu_material* materials,
  int N, uint* rng_state
) {
  point3 accumulated_color = {0.0f, 0.0f, 0.0f};
  point3 accumulated_attenuation = {1.0f, 1.0f, 1.0f};
  ray current_ray = initial_ray;

  for (int depth = 0; depth < max_depth; depth++) {
    hit_record rec;

    // Trace ray through BVH
    if (!bvh_traverse(current_ray, 0.001f, 1e38f, &rec, nodes, spheres, triangles, leaf_prim_indices, prim_types, N)) {
      // No hit - return background color weighted by accumulated attenuation
      point3 bg_contribution = vec3_mul(accumulated_attenuation, background);
      accumulated_color = vec3_add(accumulated_color, bg_contribution);
      break;
    }

    // Add emission from hit surface (lights)
    point3 emission = material_emitted(rec.mat_idx, materials);
    point3 emission_contribution = vec3_mul(accumulated_attenuation, emission);
    accumulated_color = vec3_add(accumulated_color, emission_contribution);

    // Try to scatter ray
    point3 attenuation;
    ray scattered;
    if (!scatter_material(current_ray, rec, materials, rng_state, &attenuation, &scattered)) {
      // Ray absorbed - no more bounces
      break;
    }

    // Update accumulated attenuation and continue with scattered ray
    accumulated_attenuation = vec3_mul(accumulated_attenuation, attenuation);
    current_ray = scattered;
  }

  return accumulated_color;
}

inline bool bvh_traverse(
  ray r, float t_min, float t_max,
  hit_record* rec,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const gpu_triangle* triangles,
  __global const int* leaf_prim_indices,
  __global const int* prim_types,
  int N
  ) {

  int stack[128];  // Increased from 64 for deep trees (depth 44+)
  int stack_ptr = 0;

  stack[stack_ptr++] = 0;

  bool hit_anything = false;
  float closest_t = t_max;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    lbvh_node node = nodes[node_idx];

    if (!aabb_hit(node.bbox, r, t_min, closest_t))
      continue;

    if (node.primitive_id >= 0) {
      hit_record temp_rec;
      int prim_id = node.primitive_id;
      int prim_type = prim_types[prim_id];
      int prim_idx = leaf_prim_indices[prim_id];

      bool hit = false;
      if (prim_type == 0 && prim_idx >= 0) {
        // Sphere primitive
        hit = sphere_intersection(r, t_min, closest_t, &temp_rec, spheres[prim_idx]);
      } else if (prim_type == 1 && prim_idx >= 0) {
        // Triangle primitive
        hit = triangle_intersection(r, t_min, closest_t, &temp_rec, triangles[prim_idx]);
      }

      if (hit && temp_rec.t < closest_t) {
        closest_t = temp_rec.t;
        *rec = temp_rec;
        rec->prim_idx = prim_idx;
        rec->prim_type = prim_type;
        hit_anything = true;
      }
    } else {
      if (node.left >= 0 && node.left < 2*N-1 && stack_ptr < 128) {
        stack[stack_ptr++] = node.left;
      }
      if (node.right >= 0 && node.right < 2*N-1 && stack_ptr < 128) {
        stack[stack_ptr++] = node.right;
      }
    }
  }
  return hit_anything;
}

__kernel void render_primary_rays(
    point3 camera_center,
    point3 pixel00_loc,
    point3 pixel_delta_u,
    point3 pixel_delta_v,
    int image_width,
    int image_height,
    point3 background,
    __global const lbvh_node* nodes,
    __global const sphere* spheres,
    __global const gpu_triangle* triangles,
    __global const int* leaf_prim_indices,
    __global const int* prim_types,
    int N,
    __global point3* output_pixels
) {
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);

    if (pixel_x >= image_width || pixel_y >= image_height)
        return;

    point3 pixel_center;
    pixel_center.x = pixel00_loc.x + pixel_x * pixel_delta_u.x + pixel_y * pixel_delta_v.x;
    pixel_center.y = pixel00_loc.y + pixel_x * pixel_delta_u.y + pixel_y * pixel_delta_v.y;
    pixel_center.z = pixel00_loc.z + pixel_x * pixel_delta_u.z + pixel_y * pixel_delta_v.z;

    ray r;
    r.origin = camera_center;
    r.direction.x = pixel_center.x - camera_center.x;
    r.direction.y = pixel_center.y - camera_center.y;
    r.direction.z = pixel_center.z - camera_center.z;

    hit_record rec;
    point3 color;

    if (bvh_traverse(r, 0.001f, 1e38f, &rec, nodes, spheres, triangles, leaf_prim_indices, prim_types, N)) {
        color.x = 0.5f * (rec.normal.x + 1.0f);
        color.y = 0.5f * (rec.normal.y + 1.0f);
        color.z = 0.5f * (rec.normal.z + 1.0f);
    } else {
        color.x = background.x;
        color.y = background.y;
        color.z = background.z;
    }

    int pixel_idx = pixel_y * image_width + pixel_x;
    output_pixels[pixel_idx] = color;
}

//==============================================================================================
// Phase 2: Full Path Tracing with Multi-Sampling and Defocus Blur
//==============================================================================================

__kernel void render_with_sampling(
    // Camera parameters (7)
    point3 camera_center,
    point3 pixel00_loc,
    point3 pixel_delta_u,
    point3 pixel_delta_v,
    point3 defocus_disk_u,
    point3 defocus_disk_v,
    float defocus_angle,

    // Image parameters (5)
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth,
    point3 background,

    // Scene parameters (7)
    __global const lbvh_node* nodes,
    __global const sphere* spheres,
    __global const gpu_triangle* triangles,
    __global const int* leaf_prim_indices,
    __global const int* prim_types,
    __global const gpu_material* materials,
    int N,

    // Output (1)
    __global point3* output_pixels
) {
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);

    if (pixel_x >= image_width || pixel_y >= image_height)
        return;

    point3 pixel_color = {0.0f, 0.0f, 0.0f};

    // Multi-sampling loop
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Initialize RNG for this sample
        uint rng_state = seed_rng(pixel_x, pixel_y, sample, image_width);

        // Pixel jitter for anti-aliasing
        float jitter_x = random_float(&rng_state) - 0.5f;
        float jitter_y = random_float(&rng_state) - 0.5f;

        // Compute pixel sample location
        point3 pixel_sample;
        pixel_sample.x = pixel00_loc.x + (pixel_x + jitter_x) * pixel_delta_u.x
                                       + (pixel_y + jitter_y) * pixel_delta_v.x;
        pixel_sample.y = pixel00_loc.y + (pixel_x + jitter_x) * pixel_delta_u.y
                                       + (pixel_y + jitter_y) * pixel_delta_v.y;
        pixel_sample.z = pixel00_loc.z + (pixel_x + jitter_x) * pixel_delta_u.z
                                       + (pixel_y + jitter_y) * pixel_delta_v.z;

        // Ray origin (with optional defocus blur)
        ray r;
        if (defocus_angle <= 0.0f) {
            // No defocus blur - ray from camera center
            r.origin = camera_center;
        } else {
            // Defocus blur - randomize origin on defocus disk
            point3 disk_sample = random_in_unit_disk(&rng_state);
            point3 offset;
            offset.x = defocus_disk_u.x * disk_sample.x + defocus_disk_v.x * disk_sample.y;
            offset.y = defocus_disk_u.y * disk_sample.x + defocus_disk_v.y * disk_sample.y;
            offset.z = defocus_disk_u.z * disk_sample.x + defocus_disk_v.z * disk_sample.y;

            r.origin = vec3_add(camera_center, offset);
        }

        // Ray direction points to pixel sample
        r.direction = vec3_sub(pixel_sample, r.origin);

        // Trace ray with bouncing and materials
        point3 sample_color = ray_color(r, max_depth, background,
                                        nodes, spheres, triangles, leaf_prim_indices, prim_types, materials,
                                        N, &rng_state);

        // Accumulate sample
        pixel_color = vec3_add(pixel_color, sample_color);
    }

    // Average samples
    float scale = 1.0f / samples_per_pixel;
    pixel_color = vec3_scale(pixel_color, scale);

    // Write to output
    int pixel_idx = pixel_y * image_width + pixel_x;
    output_pixels[pixel_idx] = pixel_color;
}

