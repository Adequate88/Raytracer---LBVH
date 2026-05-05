#pragma once
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"
#include "ray.h"
#include "interval.h"

class AABB {
  public:
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;


    static const AABB empty;
    static const AABB universe;


    AABB(); // The default AABB is empty, since intervals are empty by default.

    AABB
    (
      const float min_x, const float min_y, const float min_z,
      const float max_x, const float max_y, const float max_z
    );

    AABB(const point3& a, const point3& b);
    AABB(const AABB& box0, const AABB& box1);

    float axis_min(int n) const;

    float axis_max(int n) const;

    bool hit(const ray& r, interval ray_t) const;

    int longest_axis() const;

  private:
    void pad_to_minimums();
};

AABB operator+(const AABB& bbox, const vec3& offset);
AABB operator+(const vec3& offset, const AABB& bbox);
