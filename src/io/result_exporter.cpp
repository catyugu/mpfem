#include "io/result_exporter.hpp"
#include "problem/steady_problem.hpp"
#include "problem/transient_problem.hpp"
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

// -----------------------------------------------------------------------------
// Helper: project GridFunction to corner vertices
// -----------------------------------------------------------------------------
static std::vector<Real> projectToCorners(const Mesh& mesh, const GridFunction& field) {
    Index numCorners = mesh.numCornerVertices();
    const auto& cornerIndices = mesh.cornerVertexIndices();
    
    std::vector<Real> result(numCorners);
    
    // For scalar fields, directly map corner vertices
    if (field.vdim() == 1) {
        for (Index i = 0; i < numCorners; ++i) {
            result[i] = field(cornerIndices[i]);
        }
    } else {
        // For vector fields, interleave components
        for (Index i = 0; i < numCorners; ++i) {
            Index base = cornerIndices[i] * 3;  // Assuming vdim=3 for displacement
            result[i * 3] = field(base);
            result[i * 3 + 1] = field(base + 1);
            result[i * 3 + 2] = field(base + 2);
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// COMSOL text export implementation
// -----------------------------------------------------------------------------
void ResultExporter::exportComsolText(const SteadyResult& result, const Mesh& mesh,
                                      const std::string& filename) {
    exportComsolTextImpl(result.fields, mesh, filename, -1);
}

void ResultExporter::exportComsolText(const TransientResult& result, const Mesh& mesh,
                                      const std::string& filename) {
    // Export all time steps to single file (COMSOL format)
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open file for writing: " + filename);
    }

    Index numExportPoints = mesh.numCornerVertices();
    const auto& cornerIndices = mesh.cornerVertexIndices();

    file << std::setprecision(16);

    // Header
    file << "% Model:              mpfem\n";
    file << "% Version:            1.0\n";
    file << "% Date:               " << getCurrentTimestamp() << "\n";
    file << "% Dimension:          " << mesh.dim() << "\n";
    file << "% Nodes:              " << numExportPoints << "\n";
    file << "% Expressions:        " << (result.numTimeSteps() * 3) << "\n";
    file << "% Description:        Electric potential, Temperature, Displacement magnitude\n";
    file << "% Length unit:        m\n";
    
    // Field names header: x y z V@t0 T@t0 disp@t0 V@t1 T@t1 disp@t1 ...
    file << "x                       y                        z";
    for (int i = 0; i < result.numTimeSteps(); ++i) {
        file << "                        V (V) @ t=" << result.times[i] 
             << "              T (K) @ t=" << result.times[i] 
             << "        solid.disp (m) @ t=" << result.times[i];
    }
    file << "\n";

    // Data - all time steps per row
    for (Index j = 0; j < numExportPoints; ++j) {
        const Vertex& v = mesh.vertex(cornerIndices[j]);
        file << v.x() << "       " << v.y() << "       " << v.z();
        
        for (int i = 0; i < result.numTimeSteps(); ++i) {
            const auto& fields = result.snapshots[i];
            
            // V - ElectricPotential
            if (fields.hasField(FieldId::ElectricPotential)) {
                const auto& V = fields.current(FieldId::ElectricPotential);
                file << "       " << V(cornerIndices[j]);
            } else {
                file << "       0.0";
            }
            
            // T - Temperature
            if (fields.hasField(FieldId::Temperature)) {
                const auto& T = fields.current(FieldId::Temperature);
                file << "       " << T(cornerIndices[j]);
            } else {
                file << "       0.0";
            }
            
            // displacement magnitude
            if (fields.hasField(FieldId::Displacement)) {
                const auto& u = fields.current(FieldId::Displacement);
                Index base = cornerIndices[j] * 3;
                Real dx = u(base);
                Real dy = u(base + 1);
                Real dz = u(base + 2);
                Real mag = std::sqrt(dx*dx + dy*dy + dz*dz);
                file << "       " << mag;
            } else {
                file << "       0.0";
            }
        }
        file << "\n";
    }

    file.close();
    LOG_INFO << "Exported transient results to " << filename;
}

void ResultExporter::exportComsolTextImpl(const FieldValues& fields, const Mesh& mesh,
                                          const std::string& filename, Real time) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open file for writing: " + filename);
    }

    Index numExportPoints = mesh.numCornerVertices();
    const auto& cornerIndices = mesh.cornerVertexIndices();

    file << std::setprecision(16);

    // Header
    file << "% Model:              mpfem\n";
    file << "% Version:            1.0\n";
    file << "% Date:               " << getCurrentTimestamp() << "\n";
    file << "% Dimension:          " << mesh.dim() << "\n";
    file << "% Nodes:              " << numExportPoints << "\n";
    file << "% Expressions:        3\n";  // V, T, displacement magnitude
    
    if (time >= 0) {
        file << "% Time:               " << time << "\n";
    }
    
    file << "% Length unit:        m\n";
    file << "x                       y                        z                       V (V)                     T (K)                     disp (m)\n";

    // Data
    for (Index i = 0; i < numExportPoints; ++i) {
        const Vertex& v = mesh.vertex(cornerIndices[i]);
        file << v.x() << "       " << v.y() << "       " << v.z();
        
        // V - ElectricPotential
        if (fields.hasField(FieldId::ElectricPotential)) {
            const auto& V = fields.current(FieldId::ElectricPotential);
            file << "       " << V(cornerIndices[i]);
        } else {
            file << "       0.0";
        }
        
        // T - Temperature
        if (fields.hasField(FieldId::Temperature)) {
            const auto& T = fields.current(FieldId::Temperature);
            file << "       " << T(cornerIndices[i]);
        } else {
            file << "       0.0";
        }
        
        // displacement magnitude
        if (fields.hasField(FieldId::Displacement)) {
            const auto& u = fields.current(FieldId::Displacement);
            Index base = cornerIndices[i] * 3;
            Real dx = u(base);
            Real dy = u(base + 1);
            Real dz = u(base + 2);
            Real mag = std::sqrt(dx*dx + dy*dy + dz*dz);
            file << "       " << mag;
        } else {
            file << "       0.0";
        }
        
        file << "\n";
    }

    file.close();
    LOG_INFO << "Exported results to " << filename;
}

// -----------------------------------------------------------------------------
// VTU export implementation
// -----------------------------------------------------------------------------
void ResultExporter::exportVtu(const SteadyResult& result, const Mesh& mesh,
                               const std::string& filename) {
    exportVtuImpl(result.fields, mesh, filename);
}

void ResultExporter::exportVtu(const TransientResult& result, const Mesh& mesh,
                               const std::string& filename) {
    for (int i = 0; i < result.numTimeSteps(); ++i) {
        std::ostringstream oss;
        oss << filename << "_" << i << ".vtu";
        exportVtuImpl(result.snapshots[i], mesh, oss.str());
    }
}

void ResultExporter::exportVtuImpl(const FieldValues& fields, const Mesh& mesh,
                                   const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open file for writing: " + filename);
    }

    file << std::scientific << std::setprecision(10);

    Index numExportPoints = mesh.numCornerVertices();
    const auto& cornerIndices = mesh.cornerVertexIndices();

    // XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << numExportPoints 
         << "\" NumberOfCells=\"" << mesh.numElements() << "\">\n";

    // Point data - scalar fields
    file << "<PointData>\n";
    
    // V - ElectricPotential
    if (fields.hasField(FieldId::ElectricPotential)) {
        const auto& V = fields.current(FieldId::ElectricPotential);
        file << "<DataArray type=\"Float64\" Name=\"V\" format=\"ascii\">\n";
        for (Index i = 0; i < numExportPoints; ++i) {
            file << V(cornerIndices[i]) << "\n";
        }
        file << "</DataArray>\n";
    }
    
    // T - Temperature
    if (fields.hasField(FieldId::Temperature)) {
        const auto& T = fields.current(FieldId::Temperature);
        file << "<DataArray type=\"Float64\" Name=\"T\" format=\"ascii\">\n";
        for (Index i = 0; i < numExportPoints; ++i) {
            file << T(cornerIndices[i]) << "\n";
        }
        file << "</DataArray>\n";
    }
    
    // displacement
    if (fields.hasField(FieldId::Displacement)) {
        const auto& u = fields.current(FieldId::Displacement);
        file << "<DataArray type=\"Float64\" Name=\"displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index i = 0; i < numExportPoints; ++i) {
            Index base = cornerIndices[i] * 3;
            file << u(base) << " " << u(base + 1) << " " << u(base + 2) << "\n";
        }
        file << "</DataArray>\n";
        
        // displacement magnitude
        file << "<DataArray type=\"Float64\" Name=\"disp_magnitude\" format=\"ascii\">\n";
        for (Index i = 0; i < numExportPoints; ++i) {
            Index base = cornerIndices[i] * 3;
            Real dx = u(base);
            Real dy = u(base + 1);
            Real dz = u(base + 2);
            file << std::sqrt(dx*dx + dy*dy + dz*dz) << "\n";
        }
        file << "</DataArray>\n";
    }
    
    file << "</PointData>\n";

    // Points
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index i = 0; i < numExportPoints; ++i) {
        const Vertex& v = mesh.vertex(cornerIndices[i]);
        file << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells
    file << "<Cells>\n";
    
    // Connectivity - remap vertex indices to corner indices
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    std::unordered_map<Index, Index> vertexToCorner;
    for (Index i = 0; i < numExportPoints; ++i) {
        vertexToCorner[cornerIndices[i]] = i;
    }
    for (Index i = 0; i < mesh.numElements(); ++i) {
        const Element& elem = mesh.element(i);
        for (int j = 0; j < elem.numCorners(); ++j) {
            if (j > 0) file << " ";
            file << vertexToCorner[elem.vertex(j)];
        }
        file << "\n";
    }
    file << "</DataArray>\n";

    // Offsets
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    Index offset = 0;
    for (Index i = 0; i < mesh.numElements(); ++i) {
        offset += mesh.element(i).numCorners();
        file << offset << " ";
    }
    file << "\n</DataArray>\n";

    // Types
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < mesh.numElements(); ++i) {
        Geometry geom = mesh.element(i).geometry();
        int vtkType = 0;
        switch (geom) {
            case Geometry::Segment:     vtkType = 3; break;
            case Geometry::Triangle:    vtkType = 5; break;
            case Geometry::Square:      vtkType = 9; break;
            case Geometry::Tetrahedron: vtkType = 10; break;
            case Geometry::Cube:        vtkType = 12; break;
            default:                    vtkType = 0; break;
        }
        file << vtkType << " ";
    }
    file << "\n</DataArray>\n";

    file << "</Cells>\n";
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    LOG_INFO << "Exported VTU results to " << filename;
}

}  // namespace mpfem
