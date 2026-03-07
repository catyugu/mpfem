#include "mpfem/mesh/boundary_topology.hpp"
#include "mpfem/core/logger.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace mpfem {

namespace {

// 四面体面顶点索引 (相对于四面体顶点)
// 面0: 顶点 0,2,1 (局部面0)
// 面1: 顶点 0,1,3 (局部面1)
// 面2: 顶点 0,3,2 (局部面2)
// 面3: 顶点 1,2,3 (局部面3)
constexpr int TetFaceVertexIndices[4][3] = {
  {0, 2, 1},
  {0, 1, 3},
  {0, 3, 2},
  {1, 2, 3}
};

// 金字塔面顶点索引
// 面0: 底面四边形 (顶点 0,3,2,1)
// 面1-4: 侧面三角形
constexpr int PyrFaceVertexIndices[5][4] = {
  {0, 3, 2, 1},  // 底面 (四边形, 4个顶点)
  {0, 1, 4, -1}, // 侧面0 (三角形)
  {1, 2, 4, -1}, // 侧面1 (三角形)
  {2, 3, 4, -1}, // 侧面2 (三角形)
  {3, 0, 4, -1}  // 侧面3 (三角形)
};

// 三棱柱 (Wedge) 面顶点索引
// 面0-1: 三角形端面
// 面2-4: 四边形侧面
constexpr int WedgeFaceVertexIndices[5][4] = {
  {0, 2, 1, -1}, // 三角形面0
  {3, 4, 5, -1}, // 三角形面1
  {0, 1, 4, 3},  // 四边形面0
  {1, 2, 5, 4},  // 四边形面1
  {2, 0, 3, 5}   // 四边形面2
};

// 六面体面顶点索引
constexpr int HexFaceVertexIndices[6][4] = {
  {0, 3, 2, 1}, // 面0 (-z)
  {0, 1, 5, 4}, // 面1 (-y)
  {1, 2, 6, 5}, // 面2 (+x)
  {2, 3, 7, 6}, // 面3 (+y)
  {0, 4, 7, 3}, // 面4 (-x)
  {4, 5, 6, 7}  // 面5 (+z)
};

// 三角形边顶点索引
constexpr int TriEdgeVertexIndices[3][2] = {
  {0, 1},
  {1, 2},
  {0, 2}
};

// 四边形边顶点索引
constexpr int QuadEdgeVertexIndices[4][2] = {
  {0, 1},
  {1, 2},
  {2, 3},
  {0, 3}
};

}  // namespace

void BoundaryTopology::Build(
    const std::vector<ElementGroup>& domain_elements,
    const std::vector<ElementGroup>& boundary_elements) {
  
  MPFEM_INFO("Building boundary topology...");
  
  // 步骤1: 收集所有域元素的所有面
  std::map<FaceVertexKey, std::vector<DomainElementFaceInfo>> face_to_elements;
  
  int total_domain_elements = 0;
  for (const auto& group : domain_elements) {
    total_domain_elements += group.Count();
  }
  MPFEM_DEBUG("Total domain elements: %d", total_domain_elements);
  
  // 全局域元素索引
  int global_elem_idx = 0;
  
  for (const auto& group : domain_elements) {
    const int nfaces = geom::NumFaces(group.type);
    
    for (int e = 0; e < group.Count(); ++e, ++global_elem_idx) {
      auto elem_verts = group.GetElementVertices(e);
      
      for (int f = 0; f < nfaces; ++f) {
        std::vector<int> face_verts;
        
        // 根据单元类型获取面顶点
        switch (group.type) {
          case GeometryType::kTetrahedron: {
            // 四面体有4个三角形面
            for (int v = 0; v < 3; ++v) {
              face_verts.push_back(elem_verts[TetFaceVertexIndices[f][v]]);
            }
            break;
          }
          case GeometryType::kPyramid: {
            // 金字塔有1个四边形底面和4个三角形侧面
            if (f == 0) {
              // 底面是四边形
              for (int v = 0; v < 4; ++v) {
                face_verts.push_back(elem_verts[PyrFaceVertexIndices[f][v]]);
              }
            } else {
              // 侧面是三角形
              for (int v = 0; v < 3; ++v) {
                if (PyrFaceVertexIndices[f][v] >= 0) {
                  face_verts.push_back(elem_verts[PyrFaceVertexIndices[f][v]]);
                }
              }
            }
            break;
          }
          case GeometryType::kWedge: {
            // 三棱柱有2个三角形面和3个四边形面
            if (f < 2) {
              // 三角形面
              for (int v = 0; v < 3; ++v) {
                if (WedgeFaceVertexIndices[f][v] >= 0) {
                  face_verts.push_back(elem_verts[WedgeFaceVertexIndices[f][v]]);
                }
              }
            } else {
              // 四边形面
              for (int v = 0; v < 4; ++v) {
                face_verts.push_back(elem_verts[WedgeFaceVertexIndices[f][v]]);
              }
            }
            break;
          }
          case GeometryType::kHexahedron: {
            // 六面体有6个四边形面
            for (int v = 0; v < 4; ++v) {
              face_verts.push_back(elem_verts[HexFaceVertexIndices[f][v]]);
            }
            break;
          }
          default:
            // 其他类型暂不处理
            continue;
        }
        
        DomainElementFaceInfo face_info(global_elem_idx, f, group.entity_ids[e], face_verts);
        FaceVertexKey key(face_verts);
        face_to_elements[key].push_back(face_info);
      }
    }
  }
  
  MPFEM_DEBUG("Collected %zu unique faces from domain elements", face_to_elements.size());
  
  // 步骤2: 处理边界元素，匹配域元素的面
  boundary_faces_.clear();
  boundary_id_to_index_.clear();
  
  // 用于跟踪已处理的面
  std::set<FaceVertexKey> processed_faces;
  
  // 遍历所有边界单元组
  for (const auto& bdr_group : boundary_elements) {
    // 只处理二维边界单元 (三角形、四边形)
    if (geom::Dimension(bdr_group.type) != 2) {
      continue;
    }
    
    for (int b = 0; b < bdr_group.Count(); ++b) {
      auto bdr_verts = bdr_group.GetElementVertices(b);
      FaceVertexKey bdr_key(bdr_verts);
      
      int boundary_id = bdr_group.entity_ids[b];
      
      BoundaryFaceInfo face_info;
      face_info.boundary_id = boundary_id;
      face_info.face_index = b;
      
      // 查找匹配的域元素面
      auto it = face_to_elements.find(bdr_key);
      if (it != face_to_elements.end()) {
        const auto& matching_faces = it->second;
        
        if (matching_faces.size() >= 1) {
          // 找到至少一个匹配
          face_info.adjacent_element_count = static_cast<int>(matching_faces.size());
          face_info.element1_index = matching_faces[0].element_index;
          face_info.element1_local_face = matching_faces[0].local_face_index;
          face_info.element1_domain_id = matching_faces[0].domain_id;
          
          if (matching_faces.size() >= 2) {
            face_info.element2_index = matching_faces[1].element_index;
            face_info.element2_local_face = matching_faces[1].local_face_index;
            face_info.element2_domain_id = matching_faces[1].domain_id;
          }
        }
      } else {
        // 没有找到匹配的域元素，可能是网格问题
        MPFEM_WARN("Boundary face %d (ID=%d) has no matching domain element face",
                   b, boundary_id);
        face_info.adjacent_element_count = 0;
      }
      
      // 更新边界ID到索引的映射
      auto id_it = boundary_id_to_index_.find(boundary_id);
      if (id_it == boundary_id_to_index_.end()) {
        // 第一次遇到这个边界ID
        boundary_id_to_index_[boundary_id] = static_cast<int>(boundary_faces_.size());
        boundary_faces_.push_back(face_info);
      } else {
        // 已经有这个边界ID的记录，可能需要更新统计
        // 对于同一个边界ID，可能有多个边界单元，但我们只存储第一次的信息
        // 这是一个简化，假设同一边界ID的所有边界单元有相同的邻接关系
      }
      
      processed_faces.insert(bdr_key);
    }
  }
  
  // 步骤3: 找出内边界（被两个域元素共享的面，但没有对应的边界元素）
  // 这些面在COMSOL网格中可能没有显式的边界元素
  int internal_face_count = 0;
  for (const auto& [key, faces] : face_to_elements) {
    if (faces.size() == 2 && processed_faces.find(key) == processed_faces.end()) {
      // 这是一个内部面（两个域元素共享，但没有边界元素）
      internal_face_count++;
    }
  }
  
  MPFEM_INFO("Boundary topology built: %zu boundaries (%d external, %d internal)",
             boundary_faces_.size(), ExternalBoundaryCount(), InternalBoundaryCount());
}

std::vector<int> BoundaryTopology::GetExternalBoundaryIds() const {
  std::vector<int> ids;
  for (const auto& face : boundary_faces_) {
    if (face.IsExternal()) {
      ids.push_back(face.boundary_id);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<int> BoundaryTopology::GetInternalBoundaryIds() const {
  std::vector<int> ids;
  for (const auto& face : boundary_faces_) {
    if (face.IsInternal()) {
      ids.push_back(face.boundary_id);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

int BoundaryTopology::ExternalBoundaryCount() const {
  int count = 0;
  for (const auto& face : boundary_faces_) {
    if (face.IsExternal()) {
      count++;
    }
  }
  return count;
}

int BoundaryTopology::InternalBoundaryCount() const {
  int count = 0;
  for (const auto& face : boundary_faces_) {
    if (face.IsInternal()) {
      count++;
    }
  }
  return count;
}

}  // namespace mpfem
