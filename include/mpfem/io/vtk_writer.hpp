#pragma once

#include <string>
#include <vector>

#include "mpfem/mesh/mesh.hpp"
#include "mpfem/fem/grid_function.hpp"

namespace mpfem {

// ============================================================================
// VTKWriter - VTU 文件输出
// ============================================================================

class VTKWriter {
 public:
  VTKWriter() = default;

  // Write mesh and solution to VTU file
  void WriteVTU(const std::string& filename, const Mesh& mesh,
                const std::vector<std::pair<std::string, const GridFunction*>>& fields);

  // Write only mesh
  void WriteMesh(const std::string& filename, const Mesh& mesh);

 private:
  void WriteHeader(std::ofstream& ofs, int num_points, int num_cells);
  void WritePoints(std::ofstream& ofs, const Mesh& mesh);
  void WriteCells(std::ofstream& ofs, const Mesh& mesh);
  void WritePointData(std::ofstream& ofs,
                       const std::vector<std::pair<std::string, const GridFunction*>>& fields,
                       int num_points);
  void WriteFooter(std::ofstream& ofs);
};

// ============================================================================
// ResultValidator - 结果验证
// ============================================================================

class ResultValidator {
 public:
  struct ComparisonResult {
    std::string field_name;
    double max_error = 0.0;
    double mean_error = 0.0;
    double relative_error = 0.0;
    int num_points = 0;
  };

  // Load COMSOL result file
  static bool LoadComsolResult(const std::string& filename,
                                std::vector<Vec3>& points,
                                std::vector<double>& V,
                                std::vector<double>& T,
                                std::vector<double>& disp);

  // Compare with COMSOL result
  static ComparisonResult CompareField(const std::vector<double>& computed,
                                        const std::vector<double>& reference,
                                        const std::string& field_name);

  // Full comparison
  static std::vector<ComparisonResult> CompareWithComsol(
      const GridFunction& V, const GridFunction& T, const GridFunction& u,
      const std::string& comsol_result_file);

  // Print comparison report
  static void PrintReport(const std::vector<ComparisonResult>& results);
};

}  // namespace mpfem
