struct triangle {
  vec4 v0;
  vec4 v1;
  vec4 v2;
  vec4 n0;
  vec4 n1;
  vec4 n2;
  vec2 uv0;
  vec2 uv1;
  vec2 uv2;

  int mat_index;
  int _pad;
};

struct hit_record {
  vec3 p;
  vec3 normal;
  int mat_index;
  float t;
  vec2 uv;
  bool front_face;
};

const float EPSILON = 1e-8;

bool intersect_triangle(
  vec3 ray_origin, vec3 ray_dir, float t_closest,
  triangle tri, out hit_record rec)
{
  vec3 edge1 = tri.v1.xyz - tri.v0.xyz;
  vec3 edge2 = tri.v2.xyz - tri.v0.xyz;
  vec3 h = cross(ray_dir, edge2);
  float a = dot(edge1, h);

  if (abs(a) < EPSILON) return false;

  float f = 1.0 / a;
  vec3 s = ray_origin - tri.v0.xyz;
  float u = f * dot(s, h);

  if (u < 0.0 || u > 1.0) return false;

  vec3 q = cross(s, edge1);
  float v = f * dot(ray_dir, q);

  if (v < 0.0 || u + v > 1.0) return false;

  float t = f * dot(edge2, q);

  if (t < 0.001 || t > t_closest) return false;

  rec.t = t;
  rec.p = ray_origin + t * ray_dir;

  float w = 1.0 - u - v;
  vec3 outward_normal = normalize(
      w * tri.n0.xyz +
        u * tri.n1.xyz +
        v * tri.n2.xyz
    );

  rec.front_face = dot(ray_dir, outward_normal) < 0;
  rec.normal = rec.front_face ? outward_normal : -outward_normal;

  rec.uv = w * tri.uv0 + u * tri.uv1 + v * tri.uv2;
  rec.mat_index = tri.mat_index;

  return true;
}
