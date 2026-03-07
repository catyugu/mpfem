#pragma once

#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include "mpfem/mesh/element.hpp"

namespace mpfem {

// 边界面信息
struct BoundaryFaceInfo {
  int boundary_id = -1;           // 边界ID (几何实体索引)
  int face_index = -1;            // 边界单元在BoundaryElements中的索引
  
  // 邻接元素信息
  int adjacent_element_count = 0; // 邻接元素数量 (1=外边界, 2=内边界)
  int element1_index = -1;        // 第一个邻接域元素索引
  int element1_local_face = -1;   // 第一个元素的局部面索引
  int element1_domain_id = -1;    // 第一个元素的域ID
  int element2_index = -1;        // 第二个邻接域元素索引 (内边界时有效)
  int element2_local_face = -1;   // 第二个元素的局部面索引
  int element2_domain_id = -1;    // 第二个元素的域ID (内边界时有效)
  
  // 判断是否为外边界 (只有一侧有元素)
  bool IsExternal() const { return adjacent_element_count == 1; }
  
  // 判断是否为内边界 (两侧都有元素，即域间界面)
  bool IsInternal() const { return adjacent_element_count == 2; }
  
  // 判断是否为跨域界面 (两侧元素属于不同域)
  bool IsDomainInterface() const {
    return IsInternal() && element1_domain_id != element2_domain_id;
  }
};

// 面顶点编码器 (用于快速比较面)
struct FaceVertexKey {
  std::vector<int> sorted_vertices;
  
  explicit FaceVertexKey(const std::vector<int>& verts) {
    sorted_vertices = verts;
    std::sort(sorted_vertices.begin(), sorted_vertices.end());
  }
  
  bool operator==(const FaceVertexKey& other) const {
    return sorted_vertices == other.sorted_vertices;
  }
  
  bool operator<(const FaceVertexKey& other) const {
    return sorted_vertices < other.sorted_vertices;
  }
};

// 域元素的面信息
struct DomainElementFaceInfo {
  int element_index = -1;       // 域元素索引
  int local_face_index = -1;    // 局部面索引 (0-based)
  int domain_id = -1;           // 域ID
  std::vector<int> face_vertices;  // 面顶点 (按参考单元顺序)
  FaceVertexKey key;            // 排序后的顶点 (用于比较)
  
  DomainElementFaceInfo() : key({}) {}
  DomainElementFaceInfo(int elem_idx, int local_face, int domain, 
                        const std::vector<int>& verts)
      : element_index(elem_idx), local_face_index(local_face), 
        domain_id(domain), face_vertices(verts), key(verts) {}
};

// 边界拓扑类
class BoundaryTopology {
 public:
  // 构建边界拓扑
  void Build(const std::vector<ElementGroup>& domain_elements,
             const std::vector<ElementGroup>& boundary_elements);
  
  // 获取所有边界信息
  const std::vector<BoundaryFaceInfo>& GetAllBoundaryFaces() const {
    return boundary_faces_;
  }
  
  // 获取特定边界ID的信息
  const BoundaryFaceInfo* GetBoundaryFace(int boundary_id) const {
    auto it = boundary_id_to_index_.find(boundary_id);
    if (it != boundary_id_to_index_.end()) {
      return &boundary_faces_[it->second];
    }
    return nullptr;
  }
  
  // 获取外边界ID列表
  std::vector<int> GetExternalBoundaryIds() const;
  
  // 获取内边界ID列表
  std::vector<int> GetInternalBoundaryIds() const;
  
  // 判断边界是否为外边界
  bool IsExternalBoundary(int boundary_id) const {
    const auto* info = GetBoundaryFace(boundary_id);
    return info && info->IsExternal();
  }
  
  // 判断边界是否为内边界
  bool IsInternalBoundary(int boundary_id) const {
    const auto* info = GetBoundaryFace(boundary_id);
    return info && info->IsInternal();
  }
  
  // 获取边界统计信息
  int ExternalBoundaryCount() const;
  int InternalBoundaryCount() const;
  
 private:
  std::vector<BoundaryFaceInfo> boundary_faces_;
  std::map<int, int> boundary_id_to_index_;  // boundary_id -> index in boundary_faces_
  
  // 获取域元素的面顶点
  std::vector<int> GetDomainElementFaceVertices(
      const ElementGroup& group, int elem_idx, int local_face) const;
};

}  // namespace mpfem
