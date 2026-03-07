#include "mpfem/mesh/comsol_mesh_reader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mpfem {

namespace {

std::string TrimLocal(const std::string& value) {
  auto begin = value.begin();
  while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }

  auto end = value.end();
  while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
    --end;
  }

  return std::string(begin, end);
}

std::string NextDataLine(std::istream& input) {
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = TrimLocal(line);
    if (!trimmed.empty() && trimmed[0] != '#') {
      return line;
    }
  }
  return {};
}

std::string ParseTypeNameLine(const std::string& line) {
  std::istringstream stream(line);
  int length_hint = 0;
  std::string type_name;
  stream >> length_hint >> type_name;
  (void)length_hint;
  return type_name;
}

}  // namespace

ComsolMeshSummary ComsolMeshReader::ReadSummary(const std::string& mesh_path) const {
  std::ifstream input(mesh_path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open mesh file: " + mesh_path);
  }

  ComsolMeshSummary summary;
  std::unordered_map<std::string, std::unordered_set<int>> entity_ids_by_type;

  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    if (trimmed.find("# sdim") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &summary.space_dimension)) {
        throw std::runtime_error("Failed to parse # sdim line");
      }
      continue;
    }

    if (trimmed.find("# number of mesh vertices") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &summary.vertex_count)) {
        throw std::runtime_error("Failed to parse vertex count line");
      }
      continue;
    }

    if (trimmed.find("# number of element types") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &summary.element_type_count)) {
        throw std::runtime_error("Failed to parse element type count line");
      }

      for (int type_index = 0; type_index < summary.element_type_count; ++type_index) {
        const std::string type_name_line = NextDataLine(input);
        if (type_name_line.empty()) {
          throw std::runtime_error("Unexpected EOF while reading element type name");
        }
        const std::string type_name = ParseTypeNameLine(type_name_line);
        if (type_name.empty()) {
          throw std::runtime_error("Failed to parse element type name line: " + type_name_line);
        }

        int vertices_per_element = 0;
        int element_count = 0;

        const std::string vertices_per_element_line = NextDataLine(input);
        if (!ParseLeadingInt(vertices_per_element_line, &vertices_per_element)) {
          throw std::runtime_error("Failed to parse vertices-per-element");
        }
        (void)vertices_per_element;

        const std::string element_count_line = NextDataLine(input);
        if (!ParseLeadingInt(element_count_line, &element_count)) {
          throw std::runtime_error("Failed to parse element count");
        }

        summary.element_count_by_type[type_name] = element_count;

        for (int e = 0; e < element_count; ++e) {
          const std::string element_line = NextDataLine(input);
          if (element_line.empty()) {
            throw std::runtime_error("Unexpected EOF while skipping element connectivity");
          }
        }

        int entity_count = 0;
        const std::string entity_count_line = NextDataLine(input);
        if (!ParseLeadingInt(entity_count_line, &entity_count)) {
          throw std::runtime_error("Failed to parse geometric entity count");
        }

        auto& bucket = entity_ids_by_type[type_name];
        for (int id = 0; id < entity_count; ++id) {
          const std::string entity_line = NextDataLine(input);
          int entity_id = 0;
          if (!ParseLeadingInt(entity_line, &entity_id)) {
            throw std::runtime_error("Failed to parse geometric entity index");
          }
          bucket.insert(entity_id);
        }
      }
    }
  }

  if (summary.space_dimension <= 0) {
    throw std::runtime_error("Mesh summary is missing space dimension");
  }
  if (summary.vertex_count <= 0) {
    throw std::runtime_error("Mesh summary is missing vertex count");
  }

  std::unordered_set<int> domain_ids;
  std::unordered_set<int> boundary_ids;
  for (const auto& [type_name, ids] : entity_ids_by_type) {
    if (IsDomainType(type_name, summary.space_dimension)) {
      domain_ids.insert(ids.begin(), ids.end());
    }
    if (IsBoundaryType(type_name, summary.space_dimension)) {
      boundary_ids.insert(ids.begin(), ids.end());
    }
  }

  summary.domain_count = static_cast<int>(domain_ids.size());
  summary.boundary_count = static_cast<int>(boundary_ids.size());
  return summary;
}

bool ComsolMeshReader::IsBoundaryType(const std::string& type_name, const int sdim) {
  if (sdim == 3) {
    return type_name == "tri" || type_name == "quad";
  }
  if (sdim == 2) {
    return type_name == "edg";
  }
  return type_name == "vtx";
}

bool ComsolMeshReader::IsDomainType(const std::string& type_name, const int sdim) {
  if (sdim == 3) {
    return type_name == "tet" || type_name == "hex" || type_name == "pyr" || type_name == "prism";
  }
  if (sdim == 2) {
    return type_name == "tri" || type_name == "quad";
  }
  return type_name == "edg";
}

std::string ComsolMeshReader::Trim(const std::string& value) {
  auto begin = value.begin();
  while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }

  auto end = value.end();
  while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
    --end;
  }

  return std::string(begin, end);
}

bool ComsolMeshReader::ParseLeadingInt(const std::string& line, int* value) {
  if (value == nullptr) {
    return false;
  }

  std::istringstream stream(line);
  stream >> *value;
  return !stream.fail();
}

}  // namespace mpfem
