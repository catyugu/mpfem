/**
 * @file mesh.cpp
 * @brief Mesh implementation
 */

#include "mesh.hpp"
#include <set>
#include <algorithm>

namespace mpfem {

// ============================================================
// Face definition tables for each element type
// Each row defines vertex indices for one face
// ============================================================

namespace {

// Tetrahedron faces (4 triangles)
constexpr Index tet_faces[4][3] = {
    {0, 1, 2},  // Face 0
    {0, 3, 1},  // Face 1
    {0, 2, 3},  // Face 2
    {1, 3, 2}   // Face 3
};

// Hexahedron faces (6 quads)
// Node ordering: 0:(-1,-1,-1), 1:(1,-1,-1), 2:(1,1,-1), 3:(-1,1,-1)
//                4:(-1,-1,1), 5:(1,-1,1), 6:(1,1,1), 7:(-1,1,1)
constexpr Index hex_faces[6][4] = {
    {0, 3, 2, 1},  // Face 0: bottom (z = -1)
    {4, 5, 6, 7},  // Face 1: top (z = 1)
    {0, 1, 5, 4},  // Face 2: front (y = -1)
    {2, 3, 7, 6},  // Face 3: back (y = 1)
    {0, 4, 7, 3},  // Face 4: left (x = -1)
    {1, 2, 6, 5}   // Face 5: right (x = 1)
};

// Wedge faces (2 triangles + 3 quads)
// Node ordering: 0:(0,0,-1), 1:(1,0,-1), 2:(0,1,-1)
//                3:(0,0,1), 4:(1,0,1), 5:(0,1,1)
constexpr Index wedge_faces[5][4] = {
    {0, 2, 1, InvalidIndex},  // Face 0: bottom triangle
    {3, 4, 5, InvalidIndex},  // Face 1: top triangle
    {0, 1, 4, 3},   // Face 2: quad
    {1, 2, 5, 4},   // Face 3: quad
    {2, 0, 3, 5}    // Face 4: quad
};

// Pyramid faces (1 quad + 4 triangles)
// Node ordering: 0:(-1,-1,0), 1:(1,-1,0), 2:(1,1,0), 3:(-1,1,0), 4:(0,0,1)
constexpr Index pyramid_faces[5][4] = {
    {0, 3, 2, 1},   // Face 0: base quad
    {0, 1, 4, InvalidIndex},  // Face 1: triangle
    {1, 2, 4, InvalidIndex},  // Face 2: triangle
    {2, 3, 4, InvalidIndex},  // Face 3: triangle
    {3, 0, 4, InvalidIndex}   // Face 4: triangle
};

// Number of faces for each element type
constexpr int num_faces_per_element[] = {
    0,   // Vertex
    2,   // Segment
    3,   // Triangle
    4,   // Quadrilateral
    4,   // Tetrahedron
    6,   // Hexahedron
    5,   // Wedge
    5,   // Pyramid
    // Quadratic (same number of faces)
    2,   // Segment2
    3,   // Triangle2
    4,   // Quadrilateral2
    4,   // Tetrahedron2
    6,   // Hexahedron2
    5,   // Wedge2
    5    // Pyramid2
};

// Get number of vertices per face for each element type
int get_verts_per_face(ElementType type, int face_idx) {
    switch (type) {
        case ElementType::Tetrahedron:
        case ElementType::Tetrahedron2:
            return 3;
        case ElementType::Hexahedron:
        case ElementType::Hexahedron2:
            return 4;
        case ElementType::Wedge:
        case ElementType::Wedge2:
            if (face_idx < 2) return 3;  // Triangular faces
            return 4;  // Quad faces
        case ElementType::Pyramid:
        case ElementType::Pyramid2:
            if (face_idx == 0) return 4;  // Base quad
            return 3;  // Triangular faces
        default:
            return 0;
    }
}

// Number of vertices per face for each element type
// (triangular face = 3, quad face = 4)
// Returns -1 for mixed (wedge, pyramid)
constexpr int verts_per_face(ElementType type, int face_idx) {
    switch (type) {
        case ElementType::Tetrahedron:
        case ElementType::Tetrahedron2:
            return 3;
        case ElementType::Hexahedron:
        case ElementType::Hexahedron2:
            return 4;
        case ElementType::Wedge:
        case ElementType::Wedge2:
            if (face_idx < 2) return 3;  // Triangular faces
            return 4;  // Quad faces
        case ElementType::Pyramid:
        case ElementType::Pyramid2:
            if (face_idx == 0) return 4;  // Base quad
            return 3;  // Triangular faces
        default:
            return -1;
    }
}

// Get face vertices for a given element type and face index
void get_face_vertices(ElementType type, int face_idx,
                       std::span<const Index> cell_verts,
                       std::vector<Index>& face_verts) {
    face_verts.clear();

    // For quadratic elements, use only corner vertices for face matching
    // This is critical: num_corner_vertices returns geometric corner count (e.g., 4 for Tet2)
    // while num_vertices returns total nodes (e.g., 10 for Tet2)
    if (is_quadratic(type)) {
        int num_corners = num_corner_vertices(type);
        // Safety check: ensure we have enough vertices
        if (static_cast<int>(cell_verts.size()) < num_corners) {
            // This should not happen - indicates a bug in mesh reading
            MPFEM_ERROR("Quadratic element " << element_type_name(type) 
                       << " has only " << cell_verts.size() 
                       << " vertices in connectivity, but needs " << num_corners 
                       << " corners for topology. Check mesh file or reader.");
            return;
        }
        // Map to linear element for face generation
        ElementType linear_type = get_linear_element_type(type);
        get_face_vertices(linear_type, face_idx,
                          std::span<const Index>(cell_verts.data(), num_corners),
                          face_verts);
        return;
    }

    int num_face_verts = get_verts_per_face(type, face_idx);
    if (num_face_verts == 0) return;

    const Index* face_def = nullptr;

    switch (type) {
        case ElementType::Tetrahedron:
            face_def = tet_faces[face_idx];
            break;
        case ElementType::Hexahedron:
            face_def = hex_faces[face_idx];
            break;
        case ElementType::Wedge:
            face_def = wedge_faces[face_idx];
            break;
        case ElementType::Pyramid:
            face_def = pyramid_faces[face_idx];
            break;
        default:
            return;
    }

    for (int i = 0; i < num_face_verts; ++i) {
        if (face_def[i] != InvalidIndex) {
            face_verts.push_back(cell_verts[face_def[i]]);
        }
    }
}

// Face key for hashing (sorted vertex indices)
struct FaceKey {
    std::vector<Index> sorted_verts;

    explicit FaceKey(const std::vector<Index>& verts) : sorted_verts(verts) {
        std::sort(sorted_verts.begin(), sorted_verts.end());
    }

    bool operator==(const FaceKey& other) const {
        return sorted_verts == other.sorted_verts;
    }
};

struct FaceKeyHash {
    size_t operator()(const FaceKey& key) const {
        size_t h = 0;
        for (Index v : key.sorted_verts) {
            h ^= std::hash<Index>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

} // anonymous namespace

// ============================================================
// build_topology implementation
// ============================================================

void Mesh::build_topology() {
    if (topology_built_) {
        MPFEM_WARN("Topology already built, skipping");
        return;
    }

    const SizeType total_cells = num_cells();

    // Map: face key -> face index
    std::unordered_map<FaceKey, Index, FaceKeyHash> face_map;

    // Temporary storage for face -> cells mapping
    struct FaceCellInfo {
        std::vector<Index> cell_ids;
        std::vector<int> local_face_indices;
        std::vector<Index> face_verts;
    };
    std::vector<FaceCellInfo> face_infos;

    // Resize cell topologies
    cell_topologies_.resize(total_cells);

    // Process each cell block
    Index global_cell_id = 0;
    for (const auto& block : cell_blocks_) {
        const SizeType block_size = block.size();
        const ElementType elem_type = block.type();

        // Skip non-3D elements
        if (element_dimension(elem_type) != 3) {
            global_cell_id += block_size;
            continue;
        }

        const int num_faces = num_faces_per_element[static_cast<int>(elem_type)];
        std::vector<Index> face_verts;

        for (SizeType local_cell = 0; local_cell < block_size; ++local_cell) {
            auto cell_vertices = block.element_vertices(local_cell);

            for (int f = 0; f < num_faces; ++f) {
                get_face_vertices(elem_type, f, cell_vertices, face_verts);

                if (face_verts.empty()) continue;

                FaceKey key(face_verts);
                auto it = face_map.find(key);

                if (it == face_map.end()) {
                    // New face
                    Index face_id = static_cast<Index>(face_infos.size());
                    face_map[key] = face_id;

                    FaceCellInfo info;
                    info.cell_ids.push_back(global_cell_id);
                    info.local_face_indices.push_back(f);
                    info.face_verts = face_verts;
                    face_infos.push_back(std::move(info));

                    cell_topologies_[global_cell_id].face_ids.push_back(face_id);
                } else {
                    // Existing face - add this cell
                    Index face_id = it->second;
                    face_infos[face_id].cell_ids.push_back(global_cell_id);
                    face_infos[face_id].local_face_indices.push_back(f);

                    cell_topologies_[global_cell_id].face_ids.push_back(face_id);
                }
            }

            ++global_cell_id;
        }
    }

    // Build face topologies and cell neighbors
    face_topologies_.resize(face_infos.size());

    // Build boundary entity ID lookup from face blocks
    std::map<std::set<Index>, Index> boundary_face_to_entity;
    for (const auto& block : face_blocks_) {
        for (SizeType i = 0; i < block.size(); ++i) {
            auto verts = block.element_vertices(i);
            std::set<Index> vert_set(verts.begin(), verts.end());
            boundary_face_to_entity[vert_set] = block.entity_id(i);
        }
    }

    for (SizeType fid = 0; fid < face_infos.size(); ++fid) {
        const auto& info = face_infos[fid];
        auto& ft = face_topologies_[fid];

        ft.face_id = static_cast<Index>(fid);
        ft.cell_id = info.cell_ids[0];
        ft.local_face_index = info.local_face_indices[0];

        if (info.cell_ids.size() > 1) {
            // Internal face
            ft.neighbor_cell_id = info.cell_ids[1];
            ft.neighbor_local_face = info.local_face_indices[1];
            ft.is_boundary = false;
            ft.boundary_entity_id = InvalidIndex;

            // Add neighbor relationship
            cell_topologies_[ft.cell_id].neighbor_cells.push_back(ft.neighbor_cell_id);
            cell_topologies_[ft.neighbor_cell_id].neighbor_cells.push_back(ft.cell_id);
        } else {
            // Boundary face
            ft.neighbor_cell_id = InvalidIndex;
            ft.neighbor_local_face = -1;
            ft.is_boundary = true;

            // Look up boundary entity ID
            std::set<Index> vert_set(info.face_verts.begin(), info.face_verts.end());
            auto it = boundary_face_to_entity.find(vert_set);
            if (it != boundary_face_to_entity.end()) {
                ft.boundary_entity_id = it->second;
            } else {
                ft.boundary_entity_id = InvalidIndex;
            }
        }
    }

    topology_built_ = true;

    MPFEM_INFO("Topology built: " << face_topologies_.size() << " faces, "
               << num_boundary_faces() << " boundary, "
               << num_internal_faces() << " internal");
}

}  // namespace mpfem