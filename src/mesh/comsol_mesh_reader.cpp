#include "mpfem/mesh/comsol_mesh_reader.hpp"
#include "mpfem/core/logger.hpp"

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

Mesh ComsolMeshReader::Read(const std::string& mesh_path) const {
  MPFEM_INFO("Reading mesh from: %s", mesh_path.c_str());

  std::ifstream input(mesh_path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open mesh file: " + mesh_path);
  }

  Mesh mesh;
  int space_dim = 0;
  int vertex_count = 0;
  int element_type_count = 0;

  // 解析头部信息和节点坐标
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    // 解析空间维度
    if (trimmed.find("# sdim") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &space_dim)) {
        throw std::runtime_error("Failed to parse # sdim line");
      }
      mesh.SetSpaceDim(space_dim);
      continue;
    }

    // 解析节点数
    if (trimmed.find("# number of mesh vertices") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &vertex_count)) {
        throw std::runtime_error("Failed to parse vertex count line");
      }
      mesh.Nodes().Resize(vertex_count);
      MPFEM_DEBUG("Mesh has %d vertices", vertex_count);
      continue;
    }

    // 跳过 lowest mesh vertex index 行
    if (trimmed.find("# lowest mesh vertex index") != std::string::npos) {
      continue;
    }

    // 解析节点坐标
    if (trimmed.find("# Mesh vertex coordinates") != std::string::npos) {
      for (int i = 0; i < vertex_count; ++i) {
        const std::string coord_line = NextDataLine(input);
        if (coord_line.empty()) {
          throw std::runtime_error("Unexpected EOF while reading vertex coordinates");
        }
        std::istringstream coord_stream(coord_line);
        coord_stream >> mesh.Nodes().x[i] >> mesh.Nodes().y[i] >> mesh.Nodes().z[i];
      }
      MPFEM_DEBUG("Finished reading vertex coordinates");
      continue;
    }

    // 解析单元类型数
    if (trimmed.find("# number of element types") != std::string::npos) {
      if (!ParseLeadingInt(trimmed, &element_type_count)) {
        throw std::runtime_error("Failed to parse element type count line");
      }
      MPFEM_DEBUG("Mesh has %d element types", element_type_count);

      // 解析所有单元类型
      // 格式：每种类型包含以下数据行：
      // 1. "数字 类型名" 行 (如 "3 vtx")
      // 2. 每单元顶点数
      // 3. 单元数
      // 4. N个单元连接数据
      // 5. 几何实体索引数
      // 6. N个几何实体ID
      for (int type_index = 0; type_index < element_type_count; ++type_index) {
        // 读取类型名行：格式为 "数字 类型名"
        const std::string type_name_line = NextDataLine(input);
        if (type_name_line.empty()) {
          throw std::runtime_error("Unexpected EOF while reading element type");
        }
        const std::string type_name = ParseTypeNameLine(type_name_line);
        if (type_name.empty()) {
          throw std::runtime_error("Failed to parse element type name from: " +
                                   Trim(type_name_line));
        }

        // 读取每单元顶点数
        const std::string verts_per_elem_line = NextDataLine(input);
        int verts_per_element = 0;
        if (verts_per_elem_line.empty() ||
            !ParseLeadingInt(verts_per_elem_line, &verts_per_element)) {
          throw std::runtime_error("Failed to parse vertices-per-element");
        }

        // 读取单元数
        const std::string elem_count_line = NextDataLine(input);
        int element_count = 0;
        if (elem_count_line.empty() ||
            !ParseLeadingInt(elem_count_line, &element_count)) {
          throw std::runtime_error("Failed to parse element count");
        }

        MPFEM_DEBUG("Element type '%s': %d elements, %d vertices per element",
                    type_name.c_str(), element_count, verts_per_element);

        // 创建单元组
        ElementGroup group;
        group.type = GeometryTypeFromString(type_name);
        group.entity_ids.resize(element_count);
        group.offsets.resize(element_count);
        group.vertices.reserve(element_count * verts_per_element);

        // 读取单元连接数据
        for (int e = 0; e < element_count; ++e) {
          const std::string elem_line = NextDataLine(input);
          if (elem_line.empty()) {
            throw std::runtime_error("Unexpected EOF while reading element connectivity");
          }

          std::istringstream elem_stream(elem_line);
          group.offsets[e] = static_cast<int>(group.vertices.size());
          for (int v = 0; v < verts_per_element; ++v) {
            int vertex_idx;
            elem_stream >> vertex_idx;
            group.vertices.push_back(vertex_idx);
          }
        }

        // 读取几何实体索引数
        const std::string entity_count_line = NextDataLine(input);
        int entity_count = 0;
        if (entity_count_line.empty() ||
            !ParseLeadingInt(entity_count_line, &entity_count)) {
          throw std::runtime_error("Failed to parse geometric entity count");
        }

        if (entity_count != element_count) {
          throw std::runtime_error("Entity count mismatch for type " + type_name +
                                   ": expected " + std::to_string(element_count) +
                                   ", got " + std::to_string(entity_count));
        }

        // 读取几何实体ID
        for (int e = 0; e < entity_count; ++e) {
          const std::string entity_line = NextDataLine(input);
          if (entity_line.empty() ||
              !ParseLeadingInt(entity_line, &group.entity_ids[e])) {
            throw std::runtime_error("Failed to parse geometric entity index");
          }
        }

        // 判断是域单元还是边界单元
        if (IsDomainType(type_name, space_dim)) {
          mesh.DomainElements().push_back(std::move(group));
        } else if (IsBoundaryType(type_name, space_dim)) {
          mesh.BoundaryElements().push_back(std::move(group));
        }
      }

      MPFEM_INFO("Mesh loaded: %d vertices, %zu domain element groups, %zu boundary element groups",
                 vertex_count, mesh.DomainElements().size(), mesh.BoundaryElements().size());
      break;  // 解析完成
    }
  }

  return mesh;
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
  return TrimLocal(value);
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
