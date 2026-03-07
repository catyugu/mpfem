#pragma once

#include <map>
#include <vector>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/mesh/element.hpp"

namespace mpfem {

// 节点坐标存储
struct NodeCoordinates {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  int Count() const { return static_cast<int>(x.size()); }

  Vec3 Get(int i) const { return Vec3(x[i], y[i], z[i]); }

  void Resize(int n) {
    x.resize(n);
    y.resize(n);
    z.resize(n);
  }
};

// 单元组（按类型分组存储）
struct ElementGroup {
  GeometryType type = GeometryType::kPoint;
  std::vector<int> vertices;      // 连续存储所有单元的顶点
  std::vector<int> entity_ids;    // 每个单元的几何实体ID
  std::vector<int> offsets;       // 每个单元的起始偏移

  int Count() const { return static_cast<int>(entity_ids.size()); }

  int VertsPerElement() const { return geom::NumVerts(type); }

  // 获取第i个单元的顶点索引
  std::vector<int> GetElementVertices(int i) const {
    const int n = VertsPerElement();
    const int offset = offsets[i];
    return std::vector<int>(vertices.begin() + offset,
                            vertices.begin() + offset + n);
  }
};

// 网格类
class Mesh {
 public:
  int SpaceDim() const { return space_dim_; }
  int NodeCount() const { return nodes_.Count(); }

  const NodeCoordinates& Nodes() const { return nodes_; }
  NodeCoordinates& Nodes() { return nodes_; }

  // 获取域单元组
  const std::vector<ElementGroup>& DomainElements() const { return domain_elements_; }
  std::vector<ElementGroup>& DomainElements() { return domain_elements_; }

  // 获取边界单元组
  const std::vector<ElementGroup>& BoundaryElements() const { return boundary_elements_; }
  std::vector<ElementGroup>& BoundaryElements() { return boundary_elements_; }

  // 获取特定类型的域单元组
  const ElementGroup* GetDomainElementGroup(GeometryType type) const {
    for (const auto& group : domain_elements_) {
      if (group.type == type) return &group;
    }
    return nullptr;
  }

  // 获取特定类型的边界单元组
  const ElementGroup* GetBoundaryElementGroup(GeometryType type) const {
    for (const auto& group : boundary_elements_) {
      if (group.type == type) return &group;
    }
    return nullptr;
  }

  // 获取所有域ID
  std::vector<int> GetDomainIds() const {
    std::vector<int> ids;
    for (const auto& group : domain_elements_) {
      for (int id : group.entity_ids) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
          ids.push_back(id);
        }
      }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  // 获取所有边界ID
  std::vector<int> GetBoundaryIds() const {
    std::vector<int> ids;
    for (const auto& group : boundary_elements_) {
      for (int id : group.entity_ids) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
          ids.push_back(id);
        }
      }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  void SetSpaceDim(int dim) { space_dim_ = dim; }

 private:
  int space_dim_ = 3;
  NodeCoordinates nodes_;
  std::vector<ElementGroup> domain_elements_;
  std::vector<ElementGroup> boundary_elements_;
};

}  // namespace mpfem
