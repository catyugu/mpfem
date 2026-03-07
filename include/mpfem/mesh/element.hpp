#pragma once

#include <array>
#include <vector>
#include <string>

namespace mpfem {

// 几何类型枚举
enum class GeometryType {
  kPoint = 0,
  kSegment = 1,
  kTriangle = 2,
  kQuadrilateral = 3,
  kTetrahedron = 4,
  kHexahedron = 5,
  kWedge = 6,
  kPyramid = 7
};

// 几何类型常量
namespace geom {

constexpr int NumVerts(GeometryType type) {
  switch (type) {
    case GeometryType::kPoint:
      return 1;
    case GeometryType::kSegment:
      return 2;
    case GeometryType::kTriangle:
      return 3;
    case GeometryType::kQuadrilateral:
      return 4;
    case GeometryType::kTetrahedron:
      return 4;
    case GeometryType::kHexahedron:
      return 8;
    case GeometryType::kWedge:
      return 6;
    case GeometryType::kPyramid:
      return 5;
    default:
      return 0;
  }
}

constexpr int NumEdges(GeometryType type) {
  switch (type) {
    case GeometryType::kPoint:
      return 0;
    case GeometryType::kSegment:
      return 0;
    case GeometryType::kTriangle:
      return 3;
    case GeometryType::kQuadrilateral:
      return 4;
    case GeometryType::kTetrahedron:
      return 6;
    case GeometryType::kHexahedron:
      return 12;
    case GeometryType::kWedge:
      return 9;
    case GeometryType::kPyramid:
      return 8;
    default:
      return 0;
  }
}

constexpr int NumFaces(GeometryType type) {
  switch (type) {
    case GeometryType::kPoint:
      return 0;
    case GeometryType::kSegment:
      return 2;  // 两个端点
    case GeometryType::kTriangle:
      return 3;  // 三条边
    case GeometryType::kQuadrilateral:
      return 4;  // 四条边
    case GeometryType::kTetrahedron:
      return 4;  // 四个三角形面
    case GeometryType::kHexahedron:
      return 6;  // 六个四边形面
    case GeometryType::kWedge:
      return 5;  // 2三角形 + 3四边形
    case GeometryType::kPyramid:
      return 5;  // 1四边形 + 4三角形
    default:
      return 0;
  }
}

constexpr int Dimension(GeometryType type) {
  switch (type) {
    case GeometryType::kPoint:
      return 0;
    case GeometryType::kSegment:
      return 1;
    case GeometryType::kTriangle:
    case GeometryType::kQuadrilateral:
      return 2;
    case GeometryType::kTetrahedron:
    case GeometryType::kHexahedron:
    case GeometryType::kWedge:
    case GeometryType::kPyramid:
      return 3;
    default:
      return -1;
  }
}

// 四面体面顶点索引
inline constexpr std::array<std::array<int, 3>, 4> TetFaceVerts = {
    {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};

// 六面体面顶点索引
inline constexpr std::array<std::array<int, 4>, 6> HexFaceVerts = {
    {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {4, 5, 6, 7}}};

// 四面体参考单元体积 = 1/6
constexpr double TetReferenceVolume = 1.0 / 6.0;

// 六面体参考单元体积 = 8
constexpr double HexReferenceVolume = 8.0;

}  // namespace geom

// 单元索引结构
struct ElementIndex {
  GeometryType type = GeometryType::kPoint;
  int entity_id = -1;   // 域ID或边界ID
  std::vector<int> vertices;
};

// 从字符串解析几何类型
inline GeometryType GeometryTypeFromString(const std::string& name) {
  if (name == "vtx") return GeometryType::kPoint;
  if (name == "edg") return GeometryType::kSegment;
  if (name == "tri") return GeometryType::kTriangle;
  if (name == "quad") return GeometryType::kQuadrilateral;
  if (name == "tet") return GeometryType::kTetrahedron;
  if (name == "hex") return GeometryType::kHexahedron;
  if (name == "prism") return GeometryType::kWedge;
  if (name == "pyr") return GeometryType::kPyramid;
  return GeometryType::kPoint;
}

}  // namespace mpfem
