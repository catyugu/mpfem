/**
 * @file vtu_writer.cpp
 * @brief VTU file writer implementation
 */

#include "vtu_writer.hpp"
#include "core/logger.hpp"
#include "mesh/element.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace mpfem {

// Base64 encoding table
static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

VTUWriter::VTUWriter() = default;

VTUWriter::~VTUWriter() {
    close();
}

bool VTUWriter::open(const std::string& filename) {
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        MPFEM_ERROR("Failed to open VTU file: " << filename);
        return false;
    }
    
    // Write XML header and VTK file begin
    file_ << "<?xml version=\"1.0\"?>\n";
    file_ << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"";
    file_ << " byte_order=\"LittleEndian\">\n";
    file_ << "  <UnstructuredGrid>\n";
    
    return true;
}

void VTUWriter::close() {
    if (file_.is_open()) {
        file_ << "  </UnstructuredGrid>\n";
        file_ << "</VTKFile>\n";
        file_.close();
    }
}

int VTUWriter::vtk_cell_type(ElementType type) {
    // VTK cell type numbers
    // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    switch (type) {
        case ElementType::Segment:     return 3;   // VTK_LINE
        case ElementType::Triangle:    return 5;   // VTK_TRIANGLE
        case ElementType::Quadrilateral: return 9; // VTK_QUAD
        case ElementType::Tetrahedron: return 10;  // VTK_TETRA
        case ElementType::Hexahedron:  return 12;  // VTK_HEXAHEDRON
        case ElementType::Wedge:       return 13;  // VTK_WEDGE
        case ElementType::Pyramid:     return 14;  // VTK_PYRAMID
        default: return 0;  // VTK_EMPTY_CELL
    }
}

void VTUWriter::write_mesh(const Mesh& mesh) {
    num_points_ = mesh.num_vertices();
    
    // Count total cells
    num_cells_ = mesh.num_cells();
    
    // Write Piece header
    file_ << "    <Piece NumberOfPoints=\"" << num_points_
          << "\" NumberOfCells=\"" << num_cells_ << "\">\n";
    
    // Write points
    write_points(mesh);
    
    // Write cells
    write_cells(mesh);
}

void VTUWriter::write_points(const Mesh& mesh) {
    file_ << "      <Points>\n";
    file_ << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    // Write all vertex coordinates
    for (SizeType i = 0; i < mesh.num_vertices(); ++i) {
        auto p = mesh.vertex(static_cast<Index>(i));
        file_ << std::scientific << std::setprecision(15)
              << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    
    file_ << "        </DataArray>\n";
    file_ << "      </Points>\n";
}

void VTUWriter::write_cells(const Mesh& mesh) {
    // Calculate total number of connectivity entries and offsets
    SizeType total_connectivity = 0;
    for (const auto& block : mesh.cell_blocks()) {
        total_connectivity += block.size() * block.nodes_per_element();
    }
    
    // Write cells section
    file_ << "      <Cells>\n";
    
    // Connectivity
    file_ << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    file_ << "          ";
    
    Index global_cell_idx = 0;
    for (const auto& block : mesh.cell_blocks()) {
        SizeType npe = block.nodes_per_element();
        for (SizeType e = 0; e < block.size(); ++e) {
            auto conn = block.element_vertices(e);
            for (SizeType i = 0; i < npe; ++i) {
                file_ << conn[i];
                if (i < npe - 1) file_ << " ";
            }
            file_ << "\n          ";
            ++global_cell_idx;
        }
    }
    file_ << "\n";
    file_ << "        </DataArray>\n";
    
    // Offsets (cumulative count of nodes per cell)
    file_ << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    file_ << "          ";
    SizeType offset = 0;
    for (const auto& block : mesh.cell_blocks()) {
        SizeType npe = block.nodes_per_element();
        for (SizeType e = 0; e < block.size(); ++e) {
            offset += npe;
            file_ << offset << " ";
        }
    }
    file_ << "\n";
    file_ << "        </DataArray>\n";
    
    // Cell types
    file_ << "        <DataArray type=\"Int8\" Name=\"types\" format=\"ascii\">\n";
    file_ << "          ";
    for (const auto& block : mesh.cell_blocks()) {
        int vtk_type = vtk_cell_type(block.type());
        for (SizeType e = 0; e < block.size(); ++e) {
            file_ << vtk_type << " ";
        }
    }
    file_ << "\n";
    file_ << "        </DataArray>\n";
    
    file_ << "      </Cells>\n";
}

void VTUWriter::write_point_data_scalar(const std::string& name, 
                                         const std::vector<Scalar>& values) {
    file_ << "        <DataArray type=\"Float64\" Name=\"" << name 
          << "\" format=\"ascii\">\n";
    file_ << "          ";
    for (SizeType i = 0; i < values.size(); ++i) {
        file_ << std::scientific << std::setprecision(15) << values[i];
        if ((i + 1) % 5 == 0) {
            file_ << "\n          ";
        } else {
            file_ << " ";
        }
    }
    file_ << "\n        </DataArray>\n";
}

void VTUWriter::write_point_data_vector(const std::string& name,
                                         const std::vector<Scalar>& values) {
    file_ << "        <DataArray type=\"Float64\" Name=\"" << name 
          << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    file_ << "          ";
    for (SizeType i = 0; i < values.size(); ++i) {
        file_ << std::scientific << std::setprecision(15) << values[i] << " ";
        if ((i + 1) % 9 == 0) {  // 3 components * 3 = 9 values per line
            file_ << "\n          ";
        }
    }
    file_ << "\n        </DataArray>\n";
}

void VTUWriter::write_field(const std::string& name,
                            const DynamicVector& solution,
                            const DoFHandler& dof_handler,
                            VtuField::Type type) {
    const auto* fe_space = dof_handler.fe_space();
    const Mesh& mesh = *fe_space->mesh();
    int n_components = fe_space->n_components();
    int dim = mesh.dimension();
    
    // For Lagrange elements, DoFs are assigned as:
    // - Scalar field: dof = vertex_id
    // - Vector field: dof = vertex_id * n_components + component
    
    if (type == VtuField::Type::Scalar) {
        std::vector<Scalar> nodal_values(mesh.num_vertices(), 0.0);
        
        for (SizeType node = 0; node < mesh.num_vertices(); ++node) {
            // For scalar field, DoF equals vertex ID
            Index dof = static_cast<Index>(node);
            if (dof < solution.size()) {
                nodal_values[node] = solution[dof];
            }
        }
        
        // Write point data
        file_ << "      <PointData>\n";
        write_point_data_scalar(name, nodal_values);
        file_ << "      </PointData>\n";
        
    } else if (type == VtuField::Type::Vector) {
        std::vector<Scalar> nodal_values(mesh.num_vertices() * 3, 0.0);
        
        for (SizeType node = 0; node < mesh.num_vertices(); ++node) {
            for (int comp = 0; comp < n_components && comp < dim; ++comp) {
                // For vector field, DoF = vertex_id * n_components + component
                Index dof = static_cast<Index>(node) * n_components + comp;
                if (dof < solution.size()) {
                    nodal_values[node * 3 + comp] = solution[dof];
                }
            }
            // Set remaining components to 0 (for 2D in 3D space)
            for (int comp = n_components; comp < 3; ++comp) {
                nodal_values[node * 3 + comp] = 0.0;
            }
        }
        
        file_ << "      <PointData>\n";
        write_point_data_vector(name, nodal_values);
        file_ << "      </PointData>\n";
    }
}

void VTUWriter::write_fields(const std::vector<VtuField>& fields) {
    if (fields.empty()) return;
    
    file_ << "      <PointData>\n";
    
    for (const auto& field : fields) {
        switch (field.type) {
            case VtuField::Type::Scalar:
                write_point_data_scalar(field.name, field.values);
                break;
            case VtuField::Type::Vector:
                write_point_data_vector(field.name, field.values);
                break;
            case VtuField::Type::Tensor:
                // TODO: Implement tensor output
                MPFEM_WARN("Tensor field output not yet implemented: " << field.name);
                break;
        }
    }
    
    file_ << "      </PointData>\n";
}

void VTUWriter::write_cell_data_scalar(const std::string& name,
                                        const std::vector<Scalar>& values) {
    file_ << "      <CellData>\n";
    file_ << "        <DataArray type=\"Float64\" Name=\"" << name 
          << "\" format=\"ascii\">\n";
    file_ << "          ";
    for (SizeType i = 0; i < values.size(); ++i) {
        file_ << std::scientific << std::setprecision(15) << values[i] << " ";
    }
    file_ << "\n        </DataArray>\n";
    file_ << "      </CellData>\n";
}

void VTUWriter::write_domain_ids(const Mesh& mesh) {
    std::vector<Scalar> domain_ids;
    domain_ids.reserve(mesh.num_cells());
    
    // Get domain ID for each cell
    for (const auto& block : mesh.cell_blocks()) {
        for (SizeType e = 0; e < block.size(); ++e) {
            Index domain_id = block.entity_id(e);
            domain_ids.push_back(static_cast<Scalar>(domain_id));
        }
    }
    
    file_ << "      <CellData>\n";
    file_ << "        <DataArray type=\"Int64\" Name=\"DomainID\" format=\"ascii\">\n";
    file_ << "          ";
    for (SizeType i = 0; i < domain_ids.size(); ++i) {
        file_ << static_cast<Index>(domain_ids[i]) << " ";
    }
    file_ << "\n        </DataArray>\n";
    file_ << "      </CellData>\n";
}

void VTUWriter::write(const std::string& filename,
                      const Mesh& mesh,
                      const std::vector<VtuField>& fields) {
    VTUWriter writer;
    if (!writer.open(filename)) {
        return;
    }
    writer.write_mesh(mesh);
    writer.write_fields(fields);
    writer.close();
}

std::string VTUWriter::base64_encode(const std::vector<unsigned char>& data) {
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);
    
    SizeType i = 0;
    while (i < data.size()) {
        unsigned int n = (data[i] << 16);
        if (i + 1 < data.size()) n |= (data[i + 1] << 8);
        if (i + 2 < data.size()) n |= data[i + 2];
        
        result += base64_chars[(n >> 18) & 0x3F];
        result += base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < data.size()) ? base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < data.size()) ? base64_chars[n & 0x3F] : '=';
        
        i += 3;
    }
    
    return result;
}

}  // namespace mpfem
