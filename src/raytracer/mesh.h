#ifndef MESH_H
#define MESH_H
//==============================================================================================
// Mesh loader for OBJ files using tinyobjloader
//==============================================================================================

#include "triangle.h"
#include "hittable_list.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../external/tinyobjloader.h"

#include <string>
#include <iostream>
#include <algorithm>

class mesh {
  public:
    // Load OBJ file and create triangle list
    static shared_ptr<hittable_list> load_obj(
        const std::string& filename,
        shared_ptr<material> default_material,
        bool smooth_shading = true,
        double scale = 1.0
    ) {
        tinyobj::ObjReader reader;
        tinyobj::ObjReaderConfig config;
        config.triangulate = true;  // Force triangulation of polygons

        if (!reader.ParseFromFile(filename, config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyOBJ Error: " << reader.Error() << std::endl;
            }
            return make_shared<hittable_list>();
        }

        if (!reader.Warning().empty()) {
            std::clog << "TinyOBJ Warning: " << reader.Warning() << std::endl;
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        auto triangle_list = make_shared<hittable_list>();

        // Iterate over shapes (objects in OBJ file)
        for (const auto& shape : shapes) {
            size_t index_offset = 0;

            // Iterate over faces
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                size_t fv = shape.mesh.num_face_vertices[f];

                if (fv != 3) {
                    std::cerr << "Warning: Non-triangle face skipped (vertices: " << fv << ")\n";
                    index_offset += fv;
                    continue;
                }

                // Extract three vertices
                point3 vertices[3];
                vec3 normals[3];
                vec2 uvs[3];

                for (size_t v = 0; v < 3; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    // Vertex position (required)
                    vertices[v] = point3(
                        scale * attrib.vertices[3 * idx.vertex_index + 0],
                        scale * attrib.vertices[3 * idx.vertex_index + 1],
                        scale * attrib.vertices[3 * idx.vertex_index + 2]
                    );

                    // Normal (optional - compute geometric normal if missing)
                    if (idx.normal_index >= 0) {
                        normals[v] = vec3(
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]
                        );
                    } else {
                        // Will compute from geometry later
                        normals[v] = vec3(0,0,0);
                    }

                    // Texture coordinates (optional)
                    if (idx.texcoord_index >= 0) {
                        uvs[v] = vec2(
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]
                        );
                    } else {
                        uvs[v] = vec2(0, 0);
                    }
                }

                // Compute geometric normal if vertex normals missing
                bool has_normals = (shape.mesh.indices[index_offset].normal_index >= 0);
                if (!has_normals) {
                    vec3 edge1 = vertices[1] - vertices[0];
                    vec3 edge2 = vertices[2] - vertices[0];
                    vec3 geo_normal = unit_vector(cross(edge1, edge2));
                    normals[0] = normals[1] = normals[2] = geo_normal;
                }

                // TODO: Material handling (use per-face material from OBJ)
                // For now, use default material
                shared_ptr<material> tri_mat = default_material;

                // Create triangle
                auto tri = make_shared<triangle>(
                    vertices[0], vertices[1], vertices[2],
                    normals[0], normals[1], normals[2],
                    uvs[0], uvs[1], uvs[2],
                    tri_mat,
                    smooth_shading && has_normals
                );

                triangle_list->add(tri);

                index_offset += 3;
            }
        }

        std::clog << "Loaded OBJ: " << filename << " ("
                  << triangle_list->objects.size() << " triangles)\n";

        return triangle_list;
    }

    // Compute mesh center for translation
    static point3 compute_mesh_center(const shared_ptr<hittable_list>& triangles) {
        if (triangles->objects.empty()) {
            return point3(0,0,0);
        }

        // Compute AABB of all triangles
        aabb mesh_bbox = triangles->objects[0]->bounding_box();
        for (size_t i = 1; i < triangles->objects.size(); i++) {
            mesh_bbox = aabb(mesh_bbox, triangles->objects[i]->bounding_box());
        }

        // Center is midpoint of AABB
        point3 center(
            (mesh_bbox.min_x + mesh_bbox.max_x) / 2.0,
            (mesh_bbox.min_y + mesh_bbox.max_y) / 2.0,
            (mesh_bbox.min_z + mesh_bbox.max_z) / 2.0
        );

        return center;
    }
};


#endif
