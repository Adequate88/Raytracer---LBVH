#ifndef AABB_H
#define AABB_H
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


class aabb {
  public:
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    aabb() {} // The default AABB is empty, since intervals are empty by default.

    aabb(const float min_x, const float min_y, const float min_z,
          const float max_x, const float max_y, const float max_z)
      : min_x(min_x), min_y(min_y), min_z(min_z), max_x(max_x), max_y(max_y), max_z(max_z)
    {
        pad_to_minimums();
    }

    aabb(const point3& a, const point3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.


        min_x = std::min(a[0], b[0]);
        max_x = std::max(a[0], b[0]);
        min_y = std::min(a[1], b[1]); 
        max_y = std::max(a[1], b[1]); 
        min_z = std::min(a[2], b[2]); 
        max_z = std::max(a[2], b[2]); 

        pad_to_minimums();
    }

    aabb(const aabb& box0, const aabb& box1) {
        min_x = std::min(box0.min_x, box1.min_x);
        max_x = std::max(box0.max_x, box1.max_x);
        min_y = std::min(box0.min_y, box1.min_y); 
        max_y = std::max(box0.max_y, box1.max_y); 
        min_z = std::min(box0.min_z, box1.min_z); 
        max_z = std::max(box0.max_z, box1.max_z); 
    }

    float axis_min(int n) const {
        if (n == 0) return min_x;
        if (n == 1) return min_y;
        return min_z;
    }

    float axis_max(int n) const {
        if (n == 0) return max_x;
        if (n == 1) return max_y;
        return max_z;
    }

    bool hit(const ray& r, interval ray_t) const {
        const point3& ray_orig = r.origin();
        const vec3&   ray_dir  = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const float ax_min = axis_min(axis);
            const float ax_max = axis_max(axis);
            const double adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax_min - ray_orig[axis]) * adinv;
            auto t1 = (ax_max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

    int longest_axis() const {
        // Returns the index of the longest axis of the bounding box.
        float x_size = max_x - min_x;
        float y_size = max_y - min_y;
        float z_size = max_z - min_z;

        if (x_size > y_size)
            return x_size > z_size ? 0 : 2;
        else
            return y_size > z_size ? 1 : 2;
    }

    static const aabb empty, universe;

  private:

    void pad_to_minimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.
        float delta = 0.0001;
        float padding = delta / 2;

        if ((max_x - min_x) < delta) {
            min_x -= padding;
            max_x += padding;
        }
        if ((max_y - min_y) < delta) {
            min_y -= padding;
            max_y += padding;
        }
        if ((max_z - min_z) < delta) {
            min_z -= padding;
            max_z += padding;
        }
    }
};

const aabb aabb::empty    = aabb(+infinity, +infinity, +infinity, -infinity, -infinity, -infinity);
const aabb aabb::universe = aabb(-infinity, -infinity, -infinity, +infinity, +infinity, +infinity);

aabb operator+(const aabb& bbox, const vec3& offset) {
    return aabb(bbox.min_x + offset.x(), bbox.min_y + offset.y(), bbox.min_z + offset.z(),
                bbox.max_x + offset.x(), bbox.max_y + offset.y(), bbox.max_z + offset.z());
}

aabb operator+(const vec3& offset, const aabb& bbox) {
    return bbox + offset;
}


#endif
