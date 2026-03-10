/**
 * @file result_writer.cpp
 * @brief Implementation of result writer
 */

#include "result_writer.hpp"
#include "mesh/mesh.hpp"
#include "mesh/element.hpp"
#include "core/logger.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace mpfem {

ResultWriter::ResultWriter() {
}

bool ResultWriter::write_vtu(const std::string& filename,
                              const Mesh& mesh,
                              const std::map<std::string, FieldData>& fields) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        MPFEM_ERROR("Failed to open file for writing: " << filename);
        return false;
    }
    
    // VTU header
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    
    // Get mesh statistics
    SizeType n_nodes = mesh.num_vertices();
    SizeType n_cells = mesh.num_cells();
    
    // Piece
    ofs << "    <Piece NumberOfPoints=\"" << n_nodes 
        << "\" NumberOfCells=\"" << n_cells << "\">\n";
    
    // Point data (fields)
    if (!fields.empty()) {
        ofs << "      <PointData>\n";
        for (const auto& [name, field] : fields) {
            if (field.data == nullptr) continue;
            
            if (field.n_components == 1) {
                ofs << "        <DataArray type=\"Float64\" Name=\"" 
                    << field.name << "\" format=\"ascii\">\n";
                ofs << "          ";
                for (Index i = 0; i < field.data->size(); ++i) {
                    ofs << std::scientific << std::setprecision(precision_) 
                        << (*field.data)(i) << " ";
                }
                ofs << "\n        </DataArray>\n";
            } else {
                ofs << "        <DataArray type=\"Float64\" Name=\"" 
                    << field.name << "\" NumberOfComponents=\"" 
                    << field.n_components << "\" format=\"ascii\">\n";
                ofs << "          ";
                for (Index i = 0; i < field.data->size(); ++i) {
                    ofs << std::scientific << std::setprecision(precision_) 
                        << (*field.data)(i) << " ";
                }
                ofs << "\n        </DataArray>\n";
            }
        }
        ofs << "      </PointData>\n";
    }
    
    // Points (coordinates)
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    ofs << "          ";
    for (Index i = 0; i < n_nodes; ++i) {
        Point<3> p = mesh.vertex(i);
        ofs << std::scientific << std::setprecision(precision_) 
            << p(0) << " " << p(1) << " " << p(2) << " ";
    }
    ofs << "\n        </DataArray>\n";
    ofs << "      </Points>\n";
    
    // Cells
    ofs << "      <Cells>\n";
    
    // Connectivity
    ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    ofs << "          ";
    write_vtu_cells(ofs, mesh);
    ofs << "\n        </DataArray>\n";
    
    // Offsets
    ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "          ";
    SizeType offset = 0;
    for (const auto& block : mesh.cell_blocks()) {
        for (SizeType i = 0; i < block.size(); ++i) {
            offset += block.element_type().num_nodes();
            ofs << offset << " ";
        }
    }
    ofs << "\n        </DataArray>\n";
    
    // Types
    ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for (const auto& block : mesh.cell_blocks()) {
        int n_nodes = block.element_type().num_nodes();
        int dim = block.element_type().dim();
        int vtk_type = get_vtk_cell_type(n_nodes, dim);
        for (SizeType i = 0; i < block.size(); ++i) {
            ofs << vtk_type << " ";
        }
    }
    ofs << "\n        </DataArray>\n";
    
    ofs << "      </Cells>\n";
    
    // Close
    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";
    
    ofs.close();
    
    MPFEM_INFO("Wrote VTU file: " << filename);
    return true;
}

bool ResultWriter::write_vtu(const std::string& filename,
                              const Mesh& mesh,
                              const DynamicVector& solution,
                              const std::string& field_name,
                              int n_components) {
    std::map<std::string, FieldData> fields;
    fields[field_name] = FieldData(field_name, &solution, n_components);
    return write_vtu(filename, mesh, fields);
}

bool ResultWriter::write_comsol(const std::string& filename,
                                 const Mesh& mesh,
                                 const std::map<std::string, FieldData>& fields,
                                 const std::string& model_name) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        MPFEM_ERROR("Failed to open file for writing: " << filename);
        return false;
    }
    
    // Header
    auto now = std::time(nullptr);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%b %d %Y, %H:%M", std::localtime(&now));
    
    ofs << "% Model:              " << model_name << "\n";
    ofs << "% Version:            mpfem 1.0\n";
    ofs << "% Date:               " << time_str << "\n";
    ofs << "% Dimension:          " << mesh.dimension() << "\n";
    ofs << "% Nodes:              " << mesh.num_vertices() << "\n";
    ofs << "% Expressions:        " << fields.size() << "\n";
    
    // Description
    ofs << "% Description:        ";
    bool first = true;
    for (const auto& [name, field] : fields) {
        if (!first) ofs << ", ";
        ofs << name;
        first = false;
    }
    ofs << "\n";
    
    ofs << "% Length unit:        m\n";
    
    // Column headers
    ofs << std::setw(25) << std::left << "x";
    ofs << std::setw(25) << std::left << "y";
    ofs << std::setw(25) << std::left << "z";
    for (const auto& [name, field] : fields) {
        std::string header = name;
        if (!field.unit.empty()) {
            header += " (" + field.unit + ")";
        }
        ofs << std::setw(25) << std::left << header;
    }
    ofs << "\n";
    
    // Data rows
    SizeType n_nodes = mesh.num_vertices();
    for (Index i = 0; i < static_cast<Index>(n_nodes); ++i) {
        Point<3> p = mesh.vertex(i);
        
        ofs << std::scientific << std::setprecision(precision_);
        ofs << std::setw(25) << p(0);
        ofs << std::setw(25) << p(1);
        ofs << std::setw(25) << p(2);
        
        for (const auto& [name, field] : fields) {
            if (field.data && i < static_cast<Index>(field.data->size())) {
                ofs << std::setw(25) << (*field.data)(i);
            } else {
                ofs << std::setw(25) << 0.0;
            }
        }
        ofs << "\n";
    }
    
    ofs.close();
    
    MPFEM_INFO("Wrote COMSOL format file: " << filename);
    return true;
}

bool ResultWriter::write_comsol(const std::string& filename,
                                 const Mesh& mesh,
                                 const DynamicVector& solution,
                                 const std::string& field_name,
                                 const std::string& unit) {
    std::map<std::string, FieldData> fields;
    FieldData fd(field_name, &solution, 1);
    fd.unit = unit;
    fields[field_name] = fd;
    return write_comsol(filename, mesh, fields);
}

void ResultWriter::write_vtu_cells(std::ofstream& ofs, const Mesh& mesh) {
    for (const auto& block : mesh.cell_blocks()) {
        for (SizeType i = 0; i < block.size(); ++i) {
            const auto& elem = block.element(i);
            const auto& nodes = elem.vertex_indices();
            for (Index node : nodes) {
                ofs << node << " ";
            }
        }
    }
}

int ResultWriter::get_vtk_cell_type(int n_nodes, int dim) const {
    // VTK cell types
    // See: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    switch (dim) {
        case 1:
            switch (n_nodes) {
                case 2: return 3;   // VTK_LINE
                case 3: return 21;  // VTK_QUADRATIC_EDGE
                default: return 3;
            }
        case 2:
            switch (n_nodes) {
                case 3: return 5;   // VTK_TRIANGLE
                case 4: return 9;   // VTK_QUAD
                case 6: return 22;  // VTK_QUADRATIC_TRIANGLE
                case 8: return 23;  // VTK_QUADRATIC_QUAD
                default: return 5;
            }
        case 3:
            switch (n_nodes) {
                case 4: return 10;  // VTK_TETRA
                case 5: return 14;  // VTK_PYRAMID
                case 6: return 13;  // VTK_WEDGE
                case 8: return 12;  // VTK_HEXAHEDRON
                case 10: return 24; // VTK_QUADRATIC_TETRA
                case 13: return 27; // VTK_QUADRATIC_PYRAMID (approximation)
                case 15: return 26; // VTK_QUADRATIC_WEDGE
                case 20: return 25; // VTK_QUADRATIC_HEXAHEDRON
                default: return 10;
            }
        default:
            return 1;  // VTK_EMPTY_CELL
    }
}

} // namespace mpfem
