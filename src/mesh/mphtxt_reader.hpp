/**
 * @file mphtxt_reader.hpp
 * @brief COMSOL mphtxt mesh file reader
 */

#ifndef MPFEM_MESH_MPHTXT_READER_HPP
#define MPFEM_MESH_MPHTXT_READER_HPP

#include "mesh.hpp"
#include "core/types.hpp"
#include "core/exception.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace mpfem {

/**
 * @brief Reader for COMSOL Multiphysics .mphtxt mesh files
 */
class MphtxtReader {
public:
    MphtxtReader() = default;

    std::unique_ptr<Mesh> read(const std::string& filename) {
        MPFEM_INFO("Reading mesh from: " << filename);

        std::ifstream file(filename);
        if (!file.is_open()) {
            MPFEM_THROW(FileError, "Cannot open file: " << filename);
        }

        auto mesh = std::make_unique<Mesh>();

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (token == "Mesh") {
                    parse_mesh_object(file, *mesh);
                    break;
                }
            }
        }

        mesh->build_topology();

        MPFEM_INFO("Mesh loaded successfully");

        return mesh;
    }

private:
    std::string skip_to_data(std::ifstream& file) {
        std::string line;
        while (std::getline(file, line)) {
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            if (line[start] == '#') continue;
            return line;
        }
        return "";
    }

    void parse_mesh_object(std::ifstream& file, Mesh& mesh) {
        std::string line;

        // Read version
        line = skip_to_data(file);
        int version = 0;
        std::istringstream(line) >> version;

        // Read spatial dimension
        line = skip_to_data(file);
        int sdim = 0;
        std::istringstream(line) >> sdim;
        mesh.set_dimension(sdim);

        // Read number of vertices
        line = skip_to_data(file);
        SizeType num_vertices = 0;
        std::istringstream(line) >> num_vertices;
        mesh.initialize_vertices(num_vertices);

        // Skip lowest vertex index
        std::getline(file, line);

        // Read vertex coordinates
        line = skip_to_data(file);
        {
            std::istringstream iss(line);
            Scalar x = 0, y = 0, z = 0;
            iss >> x >> y >> z;
            mesh.set_vertex(0, Point<3>(x, y, z));
        }

        for (SizeType i = 1; i < num_vertices; ++i) {
            std::getline(file, line);
            std::istringstream iss(line);
            Scalar x = 0, y = 0, z = 0;
            iss >> x >> y >> z;
            mesh.set_vertex(static_cast<Index>(i), Point<3>(x, y, z));
        }

        // Read number of element types
        line = skip_to_data(file);
        int num_element_types = 0;
        std::istringstream(line) >> num_element_types;
        // Track global cell/face indices across all blocks
        Index global_cell_idx = 0;
        Index global_face_idx = 0;

        for (int t = 0; t < num_element_types; ++t) {
            parse_element_type(file, mesh, global_cell_idx, global_face_idx);
        }
    }

    void parse_element_type(std::ifstream& file, Mesh& mesh,
                            Index& global_cell_idx, Index& global_face_idx) {
        std::string line;

        // Skip to element type definition
        while (std::getline(file, line)) {
            if (line.find("# Type #") != std::string::npos) {
                break;
            }
        }

        // Read type name
        line = skip_to_data(file);
        std::istringstream iss(line);
        int type_code;
        std::string type_name;
        iss >> type_code >> type_name;

        ElementType etype = mpfem::parse_element_type(type_name);
        int elem_dim = element_dimension(etype);

        // Read vertices per element (from file - may be Euler format)
        line = skip_to_data(file);
        int file_vertices_per_element = 0;
        std::istringstream(line) >> file_vertices_per_element;

        // Read number of elements
        line = skip_to_data(file);
        SizeType num_elements = 0;
        std::istringstream(line) >> num_elements;

        // Skip "# Elements" comment
        std::getline(file, line);

        // Determine if we need Euler to Serendipity conversion
        bool needs_filtering = needs_euler_to_serendipity_filter(etype);
        std::vector<int> serendipity_mapping;
        int serendipity_nodes_per_element = num_nodes(etype);
        
        if (needs_filtering) {
            serendipity_mapping = get_euler_to_serendipity_mapping(etype);
            MPFEM_INFO("Quadratic element " << type_name 
                       << ": converting from Euler (" << file_vertices_per_element 
                       << " nodes) to Serendipity (" << serendipity_nodes_per_element 
                       << " nodes)");
        } else {
            // Identity mapping - no filtering needed
            for (int i = 0; i < file_vertices_per_element; ++i) {
                serendipity_mapping.push_back(i);
            }
        }

        // Create element block
        ElementBlock* block = nullptr;
        bool is_volume = (elem_dim == 3);
        bool is_surface = (elem_dim == 2);

        if (is_volume) {
            block = mesh.add_cell_block(etype);
        } else if (is_surface) {
            block = mesh.add_face_block(etype);
        } else if (elem_dim == 1) {
            block = mesh.add_edge_block(etype);
        }

        if (block) {
            block->reserve(num_elements);
        }

        // Track starting global index for this block
        Index block_start_idx = is_volume ? global_cell_idx : 
                                is_surface ? global_face_idx : 0;

        // Read element connectivity
        // Note: COMSOL mphtxt format uses 0-based vertex indices
        std::vector<Index> file_vertices(file_vertices_per_element);
        std::vector<Index> serendipity_vertices(serendipity_nodes_per_element);
        
        for (SizeType i = 0; i < num_elements; ++i) {
            std::getline(file, line);
            std::istringstream iss2(line);
            
            // Read all vertices from file (Euler format)
            for (int j = 0; j < file_vertices_per_element; ++j) {
                iss2 >> file_vertices[j];
                // Vertex indices are already 0-based in mphtxt format
            }
            
            // Apply Serendipity mapping if needed
            for (int j = 0; j < serendipity_nodes_per_element; ++j) {
                serendipity_vertices[j] = file_vertices[serendipity_mapping[j]];
            }
            
            if (block) {
                block->add_element(serendipity_vertices, 0);
            }
        }

        // Read number of geometric entity indices
        line = skip_to_data(file);
        SizeType num_entities = 0;
        std::istringstream(line) >> num_entities;

        // Skip "# Geometric entity indices" comment
        std::getline(file, line);

        // Read geometric entity indices and register with GeometryManager
        // Note: COMSOL mphtxt uses 0-based entity indices, but case.xml uses 1-based
        // Convert to 1-based to match COMSOL's user-facing numbering
        for (SizeType i = 0; i < num_entities; ++i) {
            std::getline(file, line);
            Index entity_id = 0;
            std::istringstream(line) >> entity_id;
            
            // Convert from 0-based to 1-based to match COMSOL GUI numbering
            entity_id += 1;

            if (block) {
                block->set_entity_id(i, entity_id);
            }

            // Register with geometry manager using global index
            Index global_idx = block_start_idx + static_cast<Index>(i);
            if (is_volume) {
                mesh.geometry().add_cell_to_domain(entity_id, global_idx);
            } else if (is_surface) {
                mesh.geometry().add_face_to_boundary(entity_id, global_idx);
            }
        }

        // Update global counters
        if (is_volume) {
            global_cell_idx += static_cast<Index>(num_elements);
        } else if (is_surface) {
            global_face_idx += static_cast<Index>(num_elements);
        }
    }
};

}  // namespace mpfem

#endif  // MPFEM_MESH_MPHTXT_READER_HPP