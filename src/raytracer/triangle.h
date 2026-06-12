#pragma once

#include "aabb.h"
#include "hittable.h"
#include "material.h"
#include <algorithm>
#include <cmath>

struct triangle_new {
  float v0[4]; // 16 Bytes
  float v1[4];
  float v2[4];

  float n0[4];
  float n1[4];
  float n2[4];

  float uv0[2]; // 8 Btyes
  float uv1[2];
  float uv2[2];

  int mat_index; // 4 Bytes
  int _pad;
}; // Total byes = 128 Bytes

static_assert(sizeof(triangle_new) == 128);

class triangle : public hittable {
public:
  triangle(const point3 &v0, const point3 &v1, const point3 &v2, const vec3 &n0,
           const vec3 &n1, const vec3 &n2, const vec2 &uv0, const vec2 &uv1,
           const vec2 &uv2, shared_ptr<material> mat,
           bool smooth_shading = true)
      : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), uv0(uv0), uv1(uv1),
        uv2(uv2), mat(mat), smooth_shading(smooth_shading) {
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    geometric_normal = unit_vector(cross(edge1, edge2));
    centroid = (v0 + v1 + v2) / 3.0;
    bbox = compute_bbox();
  }

  triangle(const point3 &v0, const point3 &v1, const point3 &v2,
           shared_ptr<material> mat)
      : v0(v0), v1(v1), v2(v2), uv0(vec2(0, 0)), uv1(vec2(0, 0)),
        uv2(vec2(0, 0)), mat(mat), smooth_shading(false) {
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    geometric_normal = unit_vector(cross(edge1, edge2));
    n0 = n1 = n2 = geometric_normal;
    centroid = (v0 + v1 + v2) / 3.0;
    bbox = compute_bbox();
  }

  bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
    const double EPSILON = 1e-8;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(r.direction(), edge2);
    double a = dot(edge1, h);

    if (std::fabs(a) < EPSILON)
      return false;

    double f = 1.0 / a;
    vec3 s = r.origin() - v0;
    double u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
      return false;

    vec3 q = cross(s, edge1);
    double v = f * dot(r.direction(), q);

    if (v < 0.0 || u + v > 1.0)
      return false;

    double t = f * dot(edge2, q);

    if (!ray_t.surrounds(t))
      return false;

    rec.t = t;
    rec.p = r.at(t);

    vec3 outward_normal;
    if (smooth_shading) {
      double w = 1.0 - u - v;
      outward_normal = unit_vector(w * n0 + u * n1 + v * n2);
    } else {
      outward_normal = geometric_normal;
    }

    rec.set_face_normal(r, outward_normal);

    double w = 1.0 - u - v;
    rec.u = w * uv0.x() + u * uv1.x() + v * uv2.x();
    rec.v = w * uv0.y() + u * uv1.y() + v * uv2.y();

    rec.mat = mat;

    return true;
  }

  AABB bounding_box() const override { return bbox; }

  const point3 &get_v0() const { return v0; }
  const point3 &get_v1() const { return v1; }
  const point3 &get_v2() const { return v2; }
  const vec3 &get_n0() const { return n0; }
  const vec3 &get_n1() const { return n1; }
  const vec3 &get_n2() const { return n2; }
  const vec2 &get_uv0() const { return uv0; }
  const vec2 &get_uv1() const { return uv1; }
  const vec2 &get_uv2() const { return uv2; }
  bool uses_smooth_shading() const { return smooth_shading; }
  const material *get_material_ptr() const { return mat.get(); }

private:
  point3 v0, v1, v2;
  vec3 n0, n1, n2;
  vec2 uv0, uv1, uv2;
  vec3 geometric_normal;
  shared_ptr<material> mat;
  AABB bbox;
  bool smooth_shading;

  AABB compute_bbox() const {
    point3 min_pt(std::min({v0.x(), v1.x(), v2.x()}),
                  std::min({v0.y(), v1.y(), v2.y()}),
                  std::min({v0.z(), v1.z(), v2.z()}));
    point3 max_pt(std::max({v0.x(), v1.x(), v2.x()}),
                  std::max({v0.y(), v1.y(), v2.y()}),
                  std::max({v0.z(), v1.z(), v2.z()}));
    return AABB(min_pt, max_pt);
  }
};
