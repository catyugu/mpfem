#include "io/result_exporter.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "mesh/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mpfem {

    namespace {

        using VertexComponentDofMap = std::unordered_map<Index, std::vector<Index>>;

        std::vector<Index> collectTopologyVertices(const Mesh& mesh)
        {
            std::vector<Index> vertices;
            vertices.reserve(static_cast<size_t>(mesh.numElements() * 4 + mesh.numBdrElements() * 4));

            for (Index elemIdx = 0; elemIdx < mesh.numElements(); ++elemIdx) {
                const Element elem = mesh.element(elemIdx);
                vertices.insert(vertices.end(), elem.vertices.begin(), elem.vertices.end());
            }

            for (Index bdrIdx = 0; bdrIdx < mesh.numBdrElements(); ++bdrIdx) {
                const Element elem = mesh.bdrElement(bdrIdx);
                vertices.insert(vertices.end(), elem.vertices.begin(), elem.vertices.end());
            }

            std::sort(vertices.begin(), vertices.end());
            vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
            return vertices;
        }

        VertexComponentDofMap buildVertexComponentDofs(const GridFunction& gf,
            const std::vector<Index>& vertexIndices)
        {
            VertexComponentDofMap dofMap;

            const FESpace* fes = gf.fes();
            if (!fes || !fes->mesh() || fes->vdim() <= 0) {
                return dofMap;
            }

            const Mesh* mesh = fes->mesh();
            std::unordered_set<Index> pending(vertexIndices.begin(), vertexIndices.end());
            dofMap.reserve(vertexIndices.size());

            for (Index elemIdx = 0; elemIdx < mesh->numElements() && !pending.empty(); ++elemIdx) {
                const ReferenceElement* refElem = fes->elementRefElement(elemIdx);
                if (!refElem) {
                    continue;
                }

                const DofLayout layout = refElem->basis().dofLayout();
                if (layout.numVertexDofs <= 0) {
                    continue;
                }

                std::vector<Index> elemDofs(refElem->numDofs() * fes->vdim(), InvalidIndex);
                fes->getElementDofs(elemIdx, elemDofs);

                const Element elem = mesh->element(elemIdx);
                for (int localVertex = 0; localVertex < static_cast<int>(elem.vertices.size()); ++localVertex) {
                    const Index vertexIdx = elem.vertices[static_cast<size_t>(localVertex)];
                    if (!pending.contains(vertexIdx)) {
                        continue;
                    }

                    const int localScalarDof = localVertex * layout.numVertexDofs;
                    std::vector<Index> componentDofs(fes->vdim(), InvalidIndex);
                    for (int c = 0; c < fes->vdim(); ++c) {
                        componentDofs[c] = elemDofs[localScalarDof * fes->vdim() + c];
                    }
                    dofMap.emplace(vertexIdx, std::move(componentDofs));
                    pending.erase(vertexIdx);
                }
            }

            return dofMap;
        }

        Real scalarAtVertex(const GridFunction& gf, const VertexComponentDofMap& dofMap, Index vertexIdx)
        {
            const auto it = dofMap.find(vertexIdx);
            if (it == dofMap.end() || it->second.empty()) {
                return 0.0;
            }

            const Index dof = it->second[0];
            if (dof == InvalidIndex || dof >= gf.numDofs()) {
                return 0.0;
            }
            return gf(dof);
        }

        Real vectorMagnitudeAtVertex(const GridFunction& gf, const VertexComponentDofMap& dofMap, Index vertexIdx)
        {
            const FESpace* fes = gf.fes();
            if (!fes || fes->vdim() < 3) {
                return 0.0;
            }

            const auto it = dofMap.find(vertexIdx);
            if (it == dofMap.end() || static_cast<int>(it->second.size()) < fes->vdim()) {
                return 0.0;
            }

            const Index dxDof = it->second[0];
            const Index dyDof = it->second[1];
            const Index dzDof = it->second[2];
            if (dxDof == InvalidIndex || dyDof == InvalidIndex || dzDof == InvalidIndex
                || dxDof >= gf.numDofs() || dyDof >= gf.numDofs() || dzDof >= gf.numDofs()) {
                return 0.0;
            }

            const Real dx = gf(dxDof);
            const Real dy = gf(dyDof);
            const Real dz = gf(dzDof);
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

    } // namespace

    std::string ResultExporter::getCurrentTimestamp()
    {
        std::time_t now = std::time(nullptr);
        std::tm* tm_info = std::localtime(&now);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
        return std::string(buffer);
    }

    // -----------------------------------------------------------------------------
    // COMSOL text export implementation
    // -----------------------------------------------------------------------------
    void ResultExporter::exportComsolText(const FieldValues& fields, const Mesh& mesh,
        const std::string& filename, Real time)
    {
        exportComsolTextImpl(fields, mesh, filename, time);
    }

    void ResultExporter::exportComsolText(const std::vector<FieldValues>& snapshots,
        const std::vector<Real>& times,
        const Mesh& mesh,
        const std::string& filename)
    {
        if (snapshots.size() != times.size()) {
            throw ArgumentException("ResultExporter::exportComsolText snapshots/times size mismatch");
        }

        // Export all time steps to single file (COMSOL format)
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw FileException("Cannot open file for writing: " + filename);
        }

        const std::vector<Index> topologyVertices = collectTopologyVertices(mesh);
        Index numExportPoints = static_cast<Index>(topologyVertices.size());
        std::unordered_map<Index, Index> pointIndexByVertex;
        pointIndexByVertex.reserve(topologyVertices.size());
        for (Index i = 0; i < numExportPoints; ++i) {
            pointIndexByVertex[topologyVertices[static_cast<size_t>(i)]] = i;
        }

        file << std::setprecision(16);

        // Header
        file << "% Model:              mpfem\n";
        file << "% Version:            1.0\n";
        file << "% Date:               " << getCurrentTimestamp() << "\n";
        file << "% Dimension:          " << mesh.dim() << "\n";
        file << "% Nodes:              " << numExportPoints << "\n";
        file << "% Expressions:        " << (snapshots.size() * 3) << "\n";
        file << "% Description:        Electric potential, Temperature, Displacement magnitude\n";
        file << "% Length unit:        m\n";

        // Field names header: x y z V@t0 T@t0 disp@t0 V@t1 T@t1 disp@t1 ...
        file << "x                       y                        z";
        for (size_t i = 0; i < snapshots.size(); ++i) {
            file << "                        V (V) @ t=" << times[i]
                 << "              T (K) @ t=" << times[i]
                 << "        solid.disp (m) @ t=" << times[i];
        }
        file << "\n";

        std::vector<const GridFunction*> vFields(snapshots.size(), nullptr);
        std::vector<const GridFunction*> tFields(snapshots.size(), nullptr);
        std::vector<const GridFunction*> uFields(snapshots.size(), nullptr);
        std::vector<VertexComponentDofMap> vDofMaps(snapshots.size());
        std::vector<VertexComponentDofMap> tDofMaps(snapshots.size());
        std::vector<VertexComponentDofMap> uDofMaps(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) {
            const auto& fields = snapshots[i];
            if (fields.hasField("V")) {
                vFields[i] = &fields.current("V");
                vDofMaps[i] = buildVertexComponentDofs(*vFields[i], topologyVertices);
            }
            if (fields.hasField("T")) {
                tFields[i] = &fields.current("T");
                tDofMaps[i] = buildVertexComponentDofs(*tFields[i], topologyVertices);
            }
            if (fields.hasField("u")) {
                uFields[i] = &fields.current("u");
                uDofMaps[i] = buildVertexComponentDofs(*uFields[i], topologyVertices);
            }
        }

        // Data - all time steps per row
        for (Index j = 0; j < numExportPoints; ++j) {
            const Vector3& v = mesh.node(topologyVertices[static_cast<size_t>(j)]);
            file << v.x() << "       " << v.y() << "       " << v.z();

            for (size_t idx = 0; idx < snapshots.size(); ++idx) {
                if (vFields[idx]) {
                    file << "       " << scalarAtVertex(*vFields[idx], vDofMaps[idx], topologyVertices[static_cast<size_t>(j)]);
                }
                else {
                    file << "       0.0";
                }

                if (tFields[idx]) {
                    file << "       " << scalarAtVertex(*tFields[idx], tDofMaps[idx], topologyVertices[static_cast<size_t>(j)]);
                }
                else {
                    file << "       0.0";
                }

                if (uFields[idx]) {
                    const Real mag = vectorMagnitudeAtVertex(*uFields[idx], uDofMaps[idx], topologyVertices[static_cast<size_t>(j)]);
                    file << "       " << mag;
                }
                else {
                    file << "       0.0";
                }
            }
            file << "\n";
        }

        file.close();
        LOG_INFO << "Exported transient results to " << filename;
    }

    void ResultExporter::exportComsolTextImpl(const FieldValues& fields, const Mesh& mesh,
        const std::string& filename, Real time)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw FileException("Cannot open file for writing: " + filename);
        }

        const std::vector<Index> topologyVertices = collectTopologyVertices(mesh);
        Index numExportPoints = static_cast<Index>(topologyVertices.size());
        std::unordered_map<Index, Index> pointIndexByVertex;
        pointIndexByVertex.reserve(topologyVertices.size());
        for (Index i = 0; i < numExportPoints; ++i) {
            pointIndexByVertex[topologyVertices[static_cast<size_t>(i)]] = i;
        }

        file << std::setprecision(16);

        // Header
        file << "% Model:              mpfem\n";
        file << "% Version:            1.0\n";
        file << "% Date:               " << getCurrentTimestamp() << "\n";
        file << "% Dimension:          " << mesh.dim() << "\n";
        file << "% Nodes:              " << numExportPoints << "\n";
        file << "% Expressions:        3\n"; // V, T, displacement magnitude

        if (time >= 0) {
            file << "% Time:               " << time << "\n";
        }

        file << "% Length unit:        m\n";
        file << "x                       y                        z                       V (V)                     T (K)                     disp (m)\n";

        const GridFunction* V = fields.hasField("V")
            ? &fields.current("V")
            : nullptr;
        const GridFunction* T = fields.hasField("T")
            ? &fields.current("T")
            : nullptr;
        const GridFunction* u = fields.hasField("u")
            ? &fields.current("u")
            : nullptr;

        const VertexComponentDofMap vDofMap = V ? buildVertexComponentDofs(*V, topologyVertices) : VertexComponentDofMap {};
        const VertexComponentDofMap tDofMap = T ? buildVertexComponentDofs(*T, topologyVertices) : VertexComponentDofMap {};
        const VertexComponentDofMap uDofMap = u ? buildVertexComponentDofs(*u, topologyVertices) : VertexComponentDofMap {};

        // Data
        for (Index i = 0; i < numExportPoints; ++i) {
            const Vector3& v = mesh.node(topologyVertices[static_cast<size_t>(i)]);
            file << v.x() << "       " << v.y() << "       " << v.z();

            if (V) {
                file << "       " << scalarAtVertex(*V, vDofMap, topologyVertices[static_cast<size_t>(i)]);
            }
            else {
                file << "       0.0";
            }

            if (T) {
                file << "       " << scalarAtVertex(*T, tDofMap, topologyVertices[static_cast<size_t>(i)]);
            }
            else {
                file << "       0.0";
            }

            if (u) {
                const Real mag = vectorMagnitudeAtVertex(*u, uDofMap, topologyVertices[static_cast<size_t>(i)]);
                file << "       " << mag;
            }
            else {
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
    void ResultExporter::exportVtu(const FieldValues& fields, const Mesh& mesh,
        const std::string& filename)
    {
        exportVtuImpl(fields, mesh, filename);
    }

    void ResultExporter::exportVtu(const std::vector<FieldValues>& snapshots,
        const Mesh& mesh,
        const std::string& filename)
    {
        for (size_t i = 0; i < snapshots.size(); ++i) {
            std::ostringstream oss;
            oss << filename << "_" << i << ".vtu";
            exportVtuImpl(snapshots[i], mesh, oss.str());
        }
    }

    void ResultExporter::exportVtuImpl(const FieldValues& fields, const Mesh& mesh,
        const std::string& filename)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw FileException("Cannot open file for writing: " + filename);
        }

        file << std::scientific << std::setprecision(10);

        const std::vector<Index> topologyVertices = collectTopologyVertices(mesh);
        Index numExportPoints = static_cast<Index>(topologyVertices.size());
        std::unordered_map<Index, Index> pointIndexByVertex;
        pointIndexByVertex.reserve(topologyVertices.size());
        for (Index i = 0; i < numExportPoints; ++i) {
            pointIndexByVertex[topologyVertices[static_cast<size_t>(i)]] = i;
        }

        // XML header
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "<UnstructuredGrid>\n";
        file << "<Piece NumberOfPoints=\"" << numExportPoints
             << "\" NumberOfCells=\"" << mesh.numElements() << "\">\n";

        const GridFunction* V = fields.hasField("V")
            ? &fields.current("V")
            : nullptr;
        const GridFunction* T = fields.hasField("T")
            ? &fields.current("T")
            : nullptr;
        const GridFunction* u = fields.hasField("u")
            ? &fields.current("u")
            : nullptr;

        const VertexComponentDofMap vDofMap = V ? buildVertexComponentDofs(*V, topologyVertices) : VertexComponentDofMap {};
        const VertexComponentDofMap tDofMap = T ? buildVertexComponentDofs(*T, topologyVertices) : VertexComponentDofMap {};
        const VertexComponentDofMap uDofMap = u ? buildVertexComponentDofs(*u, topologyVertices) : VertexComponentDofMap {};

        // Point data - scalar fields
        file << "<PointData>\n";

        if (V) {
            file << "<DataArray type=\"Float64\" Name=\"V\" format=\"ascii\">\n";
            for (Index i = 0; i < numExportPoints; ++i) {
                file << scalarAtVertex(*V, vDofMap, topologyVertices[static_cast<size_t>(i)]) << "\n";
            }
            file << "</DataArray>\n";
        }

        if (T) {
            file << "<DataArray type=\"Float64\" Name=\"T\" format=\"ascii\">\n";
            for (Index i = 0; i < numExportPoints; ++i) {
                file << scalarAtVertex(*T, tDofMap, topologyVertices[static_cast<size_t>(i)]) << "\n";
            }
            file << "</DataArray>\n";
        }

        if (u) {
            file << "<DataArray type=\"Float64\" Name=\"displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (Index i = 0; i < numExportPoints; ++i) {
                const auto it = uDofMap.find(topologyVertices[static_cast<size_t>(i)]);
                const Index dxDof = (it == uDofMap.end() || it->second.size() < 1) ? InvalidIndex : it->second[0];
                const Index dyDof = (it == uDofMap.end() || it->second.size() < 2) ? InvalidIndex : it->second[1];
                const Index dzDof = (it == uDofMap.end() || it->second.size() < 3) ? InvalidIndex : it->second[2];
                const Real dx = (dxDof == InvalidIndex || dxDof >= u->numDofs()) ? 0.0 : (*u)(dxDof);
                const Real dy = (dyDof == InvalidIndex || dyDof >= u->numDofs()) ? 0.0 : (*u)(dyDof);
                const Real dz = (dzDof == InvalidIndex || dzDof >= u->numDofs()) ? 0.0 : (*u)(dzDof);
                file << dx << " " << dy << " " << dz << "\n";
            }
            file << "</DataArray>\n";

            // displacement magnitude
            file << "<DataArray type=\"Float64\" Name=\"disp_magnitude\" format=\"ascii\">\n";
            for (Index i = 0; i < numExportPoints; ++i) {
                file << vectorMagnitudeAtVertex(*u, uDofMap, topologyVertices[static_cast<size_t>(i)]) << "\n";
            }
            file << "</DataArray>\n";
        }

        file << "</PointData>\n";

        // Points
        file << "<Points>\n";
        file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index i = 0; i < numExportPoints; ++i) {
            const Vector3& v = mesh.node(topologyVertices[static_cast<size_t>(i)]);
            file << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
        file << "</DataArray>\n";
        file << "</Points>\n";

        // Cells
        file << "<Cells>\n";

        // Connectivity - remap topology vertex ids to point indices
        file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
        for (Index i = 0; i < mesh.numElements(); ++i) {
            const Element elem = mesh.element(i);
            for (int j = 0; j < elem.numVertices(); ++j) {
                if (j > 0)
                    file << " ";
                file << pointIndexByVertex[elem.vertex(j)];
            }
            file << "\n";
        }
        file << "</DataArray>\n";

        // Offsets
        file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
        Index offset = 0;
        for (Index i = 0; i < mesh.numElements(); ++i) {
            offset += mesh.element(i).numVertices();
            file << offset << " ";
        }
        file << "\n</DataArray>\n";

        // Types
        file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (Index i = 0; i < mesh.numElements(); ++i) {
            Geometry geom = mesh.element(i).geometry;
            int vtkType = 0;
            switch (geom) {
            case Geometry::Segment:
                vtkType = 3;
                break;
            case Geometry::Triangle:
                vtkType = 5;
                break;
            case Geometry::Square:
                vtkType = 9;
                break;
            case Geometry::Tetrahedron:
                vtkType = 10;
                break;
            case Geometry::Cube:
                vtkType = 12;
                break;
            default:
                vtkType = 0;
                break;
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

} // namespace mpfem
