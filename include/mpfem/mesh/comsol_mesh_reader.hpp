#pragma once

#include <map>
#include <set>
#include <string>

namespace mpfem {

struct ComsolMeshSummary {
  int space_dimension = 0;
  int vertex_count = 0;
  int element_type_count = 0;
  std::map<std::string, int> element_count_by_type;
  int domain_count = 0;
  int boundary_count = 0;
};

class ComsolMeshReader {
 public:
  ComsolMeshSummary ReadSummary(const std::string& mesh_path) const;

 private:
  static bool IsBoundaryType(const std::string& type_name, int sdim);
  static bool IsDomainType(const std::string& type_name, int sdim);
  static std::string Trim(const std::string& value);
  static bool ParseLeadingInt(const std::string& line, int* value);
};

}  // namespace mpfem
