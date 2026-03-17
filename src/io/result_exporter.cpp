#include "io/result_exporter.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <unordered_map>

namespace mpfem {

std::string ResultExporter::getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return std::string(buffer);
}

void ResultExporter::exportComsolText(const std::string& filename,
                                      const Mesh& mesh,
                                      const std::vector<FieldResult>& fields,
                                      const std::string& description) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open file for writing: " + filename);
    }

    // Determine number of points to export from first field
    // (for quadratic meshes, this may be less than mesh.numVertices())
    Index numExportPoints = mesh.numVertices();
    bool useCornerVertices = false;
    
    if (!fields.empty() && fields[0].nodalValues.size() < static_cast<size_t>(numExportPoints)) {
        numExportPoints = static_cast<Index>(fields[0].nodalValues.size());
        useCornerVertices = true;
    }

    // Use default floating-point format (not forced scientific)
    // This matches COMSOL's output style
    file << std::setprecision(16);

    // Write header
    file << "% Model:              busbar.mph\n";
    file << "% Version:            mpfem\n";
    file << "% Date:               " << getCurrentTimestamp() << "\n";
    file << "% Dimension:          " << mesh.dim() << "\n";
    file << "% Nodes:              " << numExportPoints << "\n";
    file << "% Expressions:        " << fields.size() << "\n";
    
    if (!description.empty()) {
        file << "% Description:        " << description << "\n";
    }
    
    // Write field names header line (matching COMSOL format)
    file << "% Length unit:        m\n";
    file << "x                       y                        z";
    for (const auto& field : fields) {
        file << "                        " << field.name;
        if (!field.unit.empty()) {
            file << " (" << field.unit << ")";
        }
    }
    file << "\n";

    // Get corner vertex indices if needed
    const std::vector<Index>* cornerIndices = nullptr;
    if (useCornerVertices) {
        cornerIndices = &mesh.cornerVertexIndices();
    }

    // Write data - one line per node (only export numExportPoints)
    for (Index i = 0; i < numExportPoints; ++i) {
        // Get the actual vertex index (for corner vertices or all vertices)
        Index vIdx = (cornerIndices) ? (*cornerIndices)[i] : i;
        const Vertex& v = mesh.vertex(vIdx);
        
        // Coordinates with COMSOL-style spacing
        file << v.x() << "       " << v.y() << "       " << v.z();
        
        // Field values
        for (const auto& field : fields) {
            if (i < static_cast<Index>(field.nodalValues.size())) {
                file << "       " << field.nodalValues[i];
            } else {
                file << "       0.0";
            }
        }
        file << "\n";
    }

    file.close();
    LOG_INFO << "Exported results to " << filename;
}

void ResultExporter::exportVtu(const std::string& filename,
                               const Mesh& mesh,
                               const std::vector<FieldResult>& fields) {
    std::map<std::string, std::vector<Vector3>> emptyVectors;
    exportVtuWithVectors(filename, mesh, fields, emptyVectors);
}

void ResultExporter::exportVtuWithVectors(const std::string& filename,
                                          const Mesh& mesh,
                                          const std::vector<FieldResult>& scalarFields,
                                          const std::map<std::string, std::vector<Vector3>>& vectorFields) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open file for writing: " + filename);
    }

    file << std::scientific << std::setprecision(10);

    // Determine if we need to export only corner vertices (for high-order meshes)
    Index numExportPoints = mesh.numVertices();
    bool useCornerVertices = false;
    const std::vector<Index>* cornerIndices = nullptr;
    
    // Check if scalar fields are smaller than total vertices
    if (!scalarFields.empty() && scalarFields[0].nodalValues.size() < static_cast<size_t>(numExportPoints)) {
        numExportPoints = static_cast<Index>(scalarFields[0].nodalValues.size());
        useCornerVertices = true;
        cornerIndices = &mesh.cornerVertexIndices();
    }
    // Check if vector fields are smaller than total vertices
    if (!vectorFields.empty()) {
        const auto& firstVec = vectorFields.begin()->second;
        if (firstVec.size() < static_cast<size_t>(numExportPoints)) {
            numExportPoints = static_cast<Index>(firstVec.size());
            useCornerVertices = true;
            cornerIndices = &mesh.cornerVertexIndices();
        }
    }

    // XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";

    // Piece
    file << "<Piece NumberOfPoints=\"" << numExportPoints 
         << "\" NumberOfCells=\"" << mesh.numElements() << "\">\n";

    // Point data
    file << "<PointData>\n";
    
    // Scalar fields
    for (const auto& field : scalarFields) {
        file << "<DataArray type=\"Float64\" Name=\"" << field.name 
             << "\" format=\"ascii\">\n";
        for (size_t i = 0; i < field.nodalValues.size(); ++i) {
            file << field.nodalValues[i];
            if ((i + 1) % 5 == 0) file << "\n";
            else file << " ";
        }
        if (field.nodalValues.size() % 5 != 0) file << "\n";
        file << "</DataArray>\n";
    }

    // Vector fields
    for (const auto& [name, values] : vectorFields) {
        file << "<DataArray type=\"Float64\" Name=\"" << name 
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < values.size(); ++i) {
            file << values[i].x() << " " << values[i].y() << " " << values[i].z() << "\n";
        }
        file << "</DataArray>\n";
    }

    file << "</PointData>\n";

    // Points (coordinates) - export only the vertices we need
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    if (useCornerVertices && cornerIndices) {
        // Export only corner vertices
        for (Index i = 0; i < numExportPoints; ++i) {
            const Vertex& v = mesh.vertex((*cornerIndices)[i]);
            file << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
    } else {
        // Export all vertices
        for (Index i = 0; i < mesh.numVertices(); ++i) {
            const Vertex& v = mesh.vertex(i);
            file << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
    }
    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells - need to remap vertex indices for corner-only export
    file << "<Cells>\n";

    // Connectivity
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    if (useCornerVertices && cornerIndices) {
        // Build a map from original vertex index to corner index
        std::unordered_map<Index, Index> vertexToCorner;
        for (Index i = 0; i < static_cast<Index>(cornerIndices->size()); ++i) {
            vertexToCorner[(*cornerIndices)[i]] = i;
        }
        for (Index i = 0; i < mesh.numElements(); ++i) {
            const Element& elem = mesh.element(i);
            for (int j = 0; j < elem.numCorners(); ++j) {
                if (j > 0) file << " ";
                auto it = vertexToCorner.find(elem.vertex(j));
                file << ((it != vertexToCorner.end()) ? it->second : 0);
            }
            file << "\n";
        }
    } else {
        for (Index i = 0; i < mesh.numElements(); ++i) {
            const Element& elem = mesh.element(i);
            for (int j = 0; j < elem.numCorners(); ++j) {
                if (j > 0) file << " ";
                file << elem.vertex(j);
            }
            file << "\n";
        }
    }
    file << "</DataArray>\n";

    // Offsets
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    Index offset = 0;
    for (Index i = 0; i < mesh.numElements(); ++i) {
        offset += mesh.element(i).numCorners();
        file << offset;
        if ((i + 1) % 10 == 0) file << "\n";
        else file << " ";
    }
    if (mesh.numElements() % 10 != 0) file << "\n";
    file << "</DataArray>\n";

    // Types
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < mesh.numElements(); ++i) {
        Geometry geom = mesh.element(i).geometry();
        int vtkType = 0;
        switch (geom) {
            case Geometry::Segment:     vtkType = 3; break;  // VTK_LINE
            case Geometry::Triangle:    vtkType = 5; break;  // VTK_TRIANGLE
            case Geometry::Square:      vtkType = 9; break;  // VTK_QUAD
            case Geometry::Tetrahedron: vtkType = 10; break; // VTK_TETRA
            case Geometry::Cube:        vtkType = 12; break; // VTK_HEXAHEDRON
            default:                    vtkType = 0; break;
        }
        file << vtkType;
        if ((i + 1) % 10 == 0) file << "\n";
        else file << " ";
    }
    if (mesh.numElements() % 10 != 0) file << "\n";
    file << "</DataArray>\n";

    file << "</Cells>\n";
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    LOG_INFO << "Exported VTU results to " << filename;
}

}  // namespace mpfem
