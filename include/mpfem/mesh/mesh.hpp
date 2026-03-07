#pragma once

#include <map>
#include <vector>
#include <memory>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/mesh/element.hpp"
#include "mpfem/mesh/boundary_topology.hpp"

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

  // ==================== 边界拓扑相关方法 ====================
  
  // 构建边界拓扑（区分内边界和外边界）
  void BuildBoundaryTopology() {
    boundary_topology_.Build(domain_elements_, boundary_elements_);
  }
  
  // 获取边界拓扑
  const BoundaryTopology& GetBoundaryTopology() const {
    return boundary_topology_;
  }
  
  BoundaryTopology& GetBoundaryTopology() {
    return boundary_topology_;
  }
  
  // 判断边界是否为外边界
  bool IsExternalBoundary(int boundary_id) const {
    return boundary_topology_.IsExternalBoundary(boundary_id);
  }
  
  // 判断边界是否为内边界
  bool IsInternalBoundary(int boundary_id) const {
    return boundary_topology_.IsInternalBoundary(boundary_id);
  }
  
  // 获取外边界ID列表
  std::vector<int> GetExternalBoundaryIds() const {
    return boundary_topology_.GetExternalBoundaryIds();
  }
  
  // 获取内边界ID列表
  std::vector<int> GetInternalBoundaryIds() const {
    return boundary_topology_.GetInternalBoundaryIds();
  }
  
  // 获取边界面的详细信息
  const BoundaryFaceInfo* GetBoundaryFaceInfo(int boundary_id) const {
    return boundary_topology_.GetBoundaryFace(boundary_id);
  }

 private:
  int space_dim_ = 3;
  NodeCoordinates nodes_;
  std::vector<ElementGroup> domain_elements_;
  std::vector<ElementGroup> boundary_elements_;
  BoundaryTopology boundary_topology_;
};

}  // namespace mpfem
