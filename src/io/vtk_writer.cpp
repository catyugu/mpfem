#include "mpfem/io/vtk_writer.hpp"
#include "mpfem/fem/fe_space.hpp"
#include "mpfem/core/logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace mpfem {

// ============================================================================
// VTKWriter implementation
// ============================================================================

void VTKWriter::WriteVTU(const std::string& filename, const Mesh& mesh,
                          const std::vector<std::pair<std::string, const GridFunction*>>& fields) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    MPFEM_ERROR("Failed to open file for writing: {}", filename);
    return;
  }

  // Count cells (elements)
  int num_points = mesh.NodeCount();
  int num_cells = 0;
  const auto& domain_groups = mesh.DomainElements();
  for (const auto& group : domain_groups) {
    num_cells += group.Count();
  }

  // Write VTK header
  ofs << "<?xml version=\"1.0\"?>\n";
  ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  ofs << "  <UnstructuredGrid>\n";
  ofs << "    <Piece NumberOfPoints=\"" << num_points
      << "\" NumberOfCells=\"" << num_cells << "\">\n";

  // Write points
  WritePoints(ofs, mesh);

  // Write cells
  WriteCells(ofs, mesh);

  // Write point data (field values)
  WritePointData(ofs, fields, num_points);

  ofs << "    </Piece>\n";
  ofs << "  </UnstructuredGrid>\n";
  ofs << "</VTKFile>\n";

  ofs.close();
  MPFEM_INFO("VTU file written: {}", filename);
}

void VTKWriter::WriteMesh(const std::string& filename, const Mesh& mesh) {
  std::vector<std::pair<std::string, const GridFunction*>> empty_fields;
  WriteVTU(filename, mesh, empty_fields);
}

void VTKWriter::WritePoints(std::ofstream& ofs, const Mesh& mesh) {
  const auto& nodes = mesh.Nodes();
  int num_points = nodes.Count();

  ofs << "      <Points>\n";
  ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  ofs << "          ";

  for (int i = 0; i < num_points; ++i) {
    Vec3 p = nodes.Get(i);
    ofs << std::setprecision(15) << p.x() << " " << p.y() << " " << p.z() << " ";
    if ((i + 1) % 3 == 0) ofs << "\n          ";
  }
  ofs << "\n        </DataArray>\n";
  ofs << "      </Points>\n";
}

void VTKWriter::WriteCells(std::ofstream& ofs, const Mesh& mesh) {
  const auto& domain_groups = mesh.DomainElements();

  int num_cells = 0;
  for (const auto& group : domain_groups) {
    num_cells += group.Count();
  }

  // Count total connectivity entries
  int total_connectivity = 0;
  for (const auto& group : domain_groups) {
    total_connectivity += group.Count() * group.VertsPerElement();
  }

  ofs << "      <Cells>\n";

  // Write connectivity
  ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  ofs << "          ";
  for (const auto& group : domain_groups) {
    for (int e = 0; e < group.Count(); ++e) {
      auto verts = group.GetElementVertices(e);
      for (int v : verts) {
        ofs << v << " ";
      }
    }
  }
  ofs << "\n        </DataArray>\n";

  // Write offsets
  ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  ofs << "          ";
  int offset = 0;
  for (const auto& group : domain_groups) {
    for (int e = 0; e < group.Count(); ++e) {
      offset += group.VertsPerElement();
      ofs << offset << " ";
    }
  }
  ofs << "\n        </DataArray>\n";

  // Write types
  ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  ofs << "          ";
  for (const auto& group : domain_groups) {
    int vtk_type = 0;
    switch (group.type) {
      case GeometryType::kTetrahedron:
        vtk_type = 10;  // VTK_TETRA
        break;
      case GeometryType::kHexahedron:
        vtk_type = 12;  // VTK_HEXAHEDRON
        break;
      case GeometryType::kWedge:
        vtk_type = 13;  // VTK_WEDGE
        break;
      case GeometryType::kPyramid:
        vtk_type = 14;  // VTK_PYRAMID
        break;
      case GeometryType::kTriangle:
        vtk_type = 5;   // VTK_TRIANGLE
        break;
      case GeometryType::kQuadrilateral:
        vtk_type = 9;   // VTK_QUAD
        break;
      default:
        vtk_type = 0;
    }
    for (int e = 0; e < group.Count(); ++e) {
      ofs << vtk_type << " ";
    }
  }
  ofs << "\n        </DataArray>\n";

  ofs << "      </Cells>\n";
}

void VTKWriter::WritePointData(std::ofstream& ofs,
                                 const std::vector<std::pair<std::string, const GridFunction*>>& fields,
                                 int num_points) {
  if (fields.empty()) return;

  ofs << "      <PointData>\n";

  for (const auto& [name, gf] : fields) {
    if (!gf) continue;

    int vdim = gf->GetFES()->GetVDim();
    const VectorXd& data = gf->Data();

    if (vdim == 1) {
      // Scalar field
      ofs << "        <DataArray type=\"Float64\" Name=\"" << name
          << "\" format=\"ascii\">\n";
      ofs << "          ";
      for (int i = 0; i < num_points; ++i) {
        ofs << std::setprecision(15) << data(i) << " ";
        if ((i + 1) % 10 == 0) ofs << "\n          ";
      }
      ofs << "\n        </DataArray>\n";
    } else if (vdim == 3) {
      // Vector field
      ofs << "        <DataArray type=\"Float64\" Name=\"" << name
          << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      ofs << "          ";
      for (int i = 0; i < num_points; ++i) {
        ofs << std::setprecision(15)
            << data(i * 3) << " "
            << data(i * 3 + 1) << " "
            << data(i * 3 + 2) << " ";
        if ((i + 1) % 3 == 0) ofs << "\n          ";
      }
      ofs << "\n        </DataArray>\n";
    }
  }

  ofs << "      </PointData>\n";
}

// ============================================================================
// ResultValidator implementation
// ============================================================================

bool ResultValidator::LoadComsolResult(const std::string& filename,
                                        std::vector<Vec3>& points,
                                        std::vector<double>& V,
                                        std::vector<double>& T,
                                        std::vector<double>& disp) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    MPFEM_ERROR("Failed to open COMSOL result file: {}", filename);
    return false;
  }

  std::string line;
  int num_points = 0;
  int num_expressions = 0;

  // Parse header
  while (std::getline(ifs, line)) {
    if (line.find("% Nodes:") != std::string::npos) {
      sscanf(line.c_str(), "%% Nodes: %d", &num_points);
    }
    if (line.find("% Expressions:") != std::string::npos) {
      sscanf(line.c_str(), "%% Expressions: %d", &num_expressions);
    }
    if (line.empty() || line[0] != '%') {
      break;  // End of header
    }
  }

  // Skip column headers
  std::getline(ifs, line);

  points.resize(num_points);
  V.resize(num_points);
  T.resize(num_points);
  disp.resize(num_points);

  // Read data
  for (int i = 0; i < num_points; ++i) {
    double x, y, z, v_val, t_val, d_val;
    ifs >> x >> y >> z >> v_val >> t_val >> d_val;
    points[i] = Vec3(x, y, z);
    V[i] = v_val;
    T[i] = t_val;
    disp[i] = d_val;
  }

  ifs.close();
  MPFEM_INFO("Loaded COMSOL result: {} points", num_points);
  return true;
}

ResultValidator::ComparisonResult ResultValidator::CompareField(
    const std::vector<double>& computed,
    const std::vector<double>& reference,
    const std::string& field_name) {
  ComparisonResult result;
  result.field_name = field_name;
  result.num_points = static_cast<int>(computed.size());

  if (computed.size() != reference.size()) {
    MPFEM_ERROR("Size mismatch in comparison: {} vs {}", computed.size(), reference.size());
    return result;
  }

  double sum_error = 0.0;
  double sum_sq_ref = 0.0;

  for (size_t i = 0; i < computed.size(); ++i) {
    double err = std::abs(computed[i] - reference[i]);
    sum_error += err;
    sum_sq_ref += reference[i] * reference[i];
    result.max_error = std::max(result.max_error, err);
  }

  result.mean_error = sum_error / computed.size();
  result.relative_error = std::sqrt(sum_error * sum_error / sum_sq_ref);

  return result;
}

std::vector<ResultValidator::ComparisonResult> ResultValidator::CompareWithComsol(
    const GridFunction& V, const GridFunction& T, const GridFunction& u,
    const std::string& comsol_result_file) {
  std::vector<ComparisonResult> results;

  std::vector<Vec3> points;
  std::vector<double> ref_V, ref_T, ref_disp;

  if (!LoadComsolResult(comsol_result_file, points, ref_V, ref_T, ref_disp)) {
    return results;
  }

  // Extract computed values at the same points
  std::vector<double> comp_V(points.size()), comp_T(points.size()), comp_disp(points.size());

  const Mesh* mesh = V.GetFES()->GetMesh();
  const auto& nodes = mesh->Nodes();

  // For now, assume the points are in the same order as mesh nodes
  // (This is typically true for COMSOL export)
  int num_nodes = mesh->NodeCount();
  if (static_cast<int>(points.size()) != num_nodes) {
    MPFEM_WARN("Point count mismatch: {} vs {}, assuming same ordering",
               points.size(), num_nodes);
  }

  const VectorXd& V_data = V.Data();
  const VectorXd& T_data = T.Data();
  const VectorXd& u_data = u.Data();

  int min_points = std::min(static_cast<int>(points.size()), num_nodes);

  for (int i = 0; i < min_points; ++i) {
    comp_V[i] = V_data(i);
    comp_T[i] = T_data(i);
    comp_disp[i] = std::sqrt(u_data(i * 3) * u_data(i * 3) +
                              u_data(i * 3 + 1) * u_data(i * 3 + 1) +
                              u_data(i * 3 + 2) * u_data(i * 3 + 2));
  }

  // Compare
  results.push_back(CompareField(comp_V, ref_V, "Electric Potential (V)"));
  results.push_back(CompareField(comp_T, ref_T, "Temperature (K)"));
  results.push_back(CompareField(comp_disp, ref_disp, "Displacement Magnitude (m)"));

  return results;
}

void ResultValidator::PrintReport(const std::vector<ComparisonResult>& results) {
  MPFEM_INFO("\n========== Result Comparison Report ==========");
  for (const auto& r : results) {
    MPFEM_INFO("\nField: {}", r.field_name);
    MPFEM_INFO("  Number of points: {}", r.num_points);
    MPFEM_INFO("  Maximum error:    {:.6e}", r.max_error);
    MPFEM_INFO("  Mean error:       {:.6e}", r.mean_error);
    MPFEM_INFO("  Relative error:   {:.6e}", r.relative_error);
  }
  MPFEM_INFO("\n==============================================");
}

}  // namespace mpfem
