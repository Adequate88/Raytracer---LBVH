#ifndef TRIANGLE_H
#define TRIANGLE_H
//==============================================================================================
// Triangle primitive for ray tracing
// Implements Möller-Trumbore ray-triangle intersection algorithm
//==============================================================================================

#include "hittable.h"
#include "material.h"
#include <cmath>
#include <algorithm>

class triangle : public hittable {
  public:
    // Full constructor with all parameters
    triangle(
        const point3& v0, const point3& v1, const point3& v2,
        const vec3& n0, const vec3& n1, const vec3& n2,
        const vec2& uv0, const vec2& uv1, const vec2& uv2,
        shared_ptr<material> mat,
        bool smooth_shading = true
    ) : v0(v0), v1(v1), v2(v2),
        n0(n0), n1(n1), n2(n2),
        uv0(uv0), uv1(uv1), uv2(uv2),
        mat(mat), smooth_shading(smooth_shading)
    {
        // Compute geometric normal
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        geometric_normal = unit_vector(cross(edge1, edge2));

        // Centroid is average of vertices (for BVH)
        centroid = (v0 + v1 + v2) / 3.0;

        // Compute bounding box
        bbox = compute_bbox();
    }

    // Simplified constructor (flat shading, no UVs)
    triangle(
        const point3& v0, const point3& v1, const point3& v2,
        shared_ptr<material> mat
    ) : v0(v0), v1(v1), v2(v2),
        uv0(vec2(0,0)), uv1(vec2(0,0)), uv2(vec2(0,0)),
        mat(mat), smooth_shading(false)
    {
        // Compute geometric normal
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        geometric_normal = unit_vector(cross(edge1, edge2));

        // For flat shading, all vertex normals equal geometric normal
        n0 = n1 = n2 = geometric_normal;

        // Centroid is average of vertices
        centroid = (v0 + v1 + v2) / 3.0;

        // Compute bounding box
        bbox = compute_bbox();
    }

    // Ray-triangle intersection using Möller-Trumbore algorithm
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        const double EPSILON = 1e-8;

        // Edge vectors
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;

        // Begin Möller-Trumbore algorithm
        vec3 h = cross(r.direction(), edge2);
        double a = dot(edge1, h);

        // Ray parallel to triangle
        if (std::fabs(a) < EPSILON)
            return false;

        double f = 1.0 / a;
        vec3 s = r.origin() - v0;
        double u = f * dot(s, h);

        // u outside [0,1]
        if (u < 0.0 || u > 1.0)
            return false;

        vec3 q = cross(s, edge1);
        double v = f * dot(r.direction(), q);

        // v outside [0,1] or u+v > 1
        if (v < 0.0 || u + v > 1.0)
            return false;

        // Compute t
        double t = f * dot(edge2, q);

        if (!ray_t.surrounds(t))
            return false;

        // We have a hit - fill hit_record
        rec.t = t;
        rec.p = r.at(t);

        // Compute normal (smooth or flat)
        vec3 outward_normal;
        if (smooth_shading) {
            // Barycentric interpolation: w = 1-u-v
            double w = 1.0 - u - v;
            outward_normal = unit_vector(w * n0 + u * n1 + v * n2);
        } else {
            outward_normal = geometric_normal;
        }

        rec.set_face_normal(r, outward_normal);

        // Interpolate UVs (barycentric)
        double w = 1.0 - u - v;
        rec.u = w * uv0.x() + u * uv1.x() + v * uv2.x();
        rec.v = w * uv0.y() + u * uv1.y() + v * uv2.y();

        rec.mat = mat;

        return true;
    }

    aabb bounding_box() const override {
        return bbox;
    }

    // Accessors for GPU rendering
    const point3& get_v0() const { return v0; }
    const point3& get_v1() const { return v1; }
    const point3& get_v2() const { return v2; }
    const vec3& get_n0() const { return n0; }
    const vec3& get_n1() const { return n1; }
    const vec3& get_n2() const { return n2; }
    bool uses_smooth_shading() const { return smooth_shading; }
    const material* get_material_ptr() const { return mat.get(); }

  private:
    point3 v0, v1, v2;          // Vertices
    vec3 n0, n1, n2;            // Vertex normals
    vec2 uv0, uv1, uv2;         // Texture coordinates
    vec3 geometric_normal;      // Face normal (always computed)
    shared_ptr<material> mat;
    aabb bbox;
    bool smooth_shading;        // Use interpolated normals vs flat

    aabb compute_bbox() const {
        // Find min/max of all three vertices
        point3 min_pt(
            std::min({v0.x(), v1.x(), v2.x()}),
            std::min({v0.y(), v1.y(), v2.y()}),
            std::min({v0.z(), v1.z(), v2.z()})
        );

        point3 max_pt(
            std::max({v0.x(), v1.x(), v2.x()}),
            std::max({v0.y(), v1.y(), v2.y()}),
            std::max({v0.z(), v1.z(), v2.z()})
        );

        return aabb(min_pt, max_pt);  // aabb constructor handles padding
    }
};


#endif
