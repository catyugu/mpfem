#pragma once

#include <array>
#include <memory>
#include <vector>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/core/types.hpp"
#include "mpfem/mesh/element.hpp"

// Forward declaration
namespace mpfem {
class Mesh;
}

namespace mpfem {

// ============================================================================
// IntegrationPoint - 积分点
// ============================================================================

struct IntegrationPoint {
  double x = 0.0;  // 参考单元坐标
  double y = 0.0;
  double z = 0.0;
  double weight = 0.0;  // 积分权重

  IntegrationPoint() = default;
  IntegrationPoint(double x_, double y_, double z_, double w)
      : x(x_), y(y_), z(z_), weight(w) {}
};

// ============================================================================
// IntegrationRule - 积分规则
// ============================================================================

class IntegrationRule {
 public:
  IntegrationRule() = default;
  explicit IntegrationRule(int order) : order_(order) {}

  int GetOrder() const { return order_; }
  int GetNPoints() const { return static_cast<int>(points_.size()); }

  const IntegrationPoint& GetPoint(int i) const { return points_[i]; }
  IntegrationPoint& GetPoint(int i) { return points_[i]; }

  void AddPoint(const IntegrationPoint& ip) { points_.push_back(ip); }

  double GetWeight(int i) const { return points_[i].weight; }

 private:
  int order_ = 0;
  std::vector<IntegrationPoint> points_;
};

// ============================================================================
// IntegrationRules - 积分规则管理器
// ============================================================================

class IntegrationRules {
 public:
  static IntegrationRules& Instance() {
    static IntegrationRules instance;
    return instance;
  }

  // Get integration rule for given geometry type and order
  const IntegrationRule& Get(GeometryType geom, int order) const;

 private:
  IntegrationRules();
  IntegrationRules(const IntegrationRules&) = delete;
  IntegrationRules& operator=(const IntegrationRules&) = delete;

  void InitTetrahedronRules();
  void InitTriangleRules();
  void InitSegmentRules();
  void InitHexahedronRules();
  void InitPyramidRules();

  // [geometry_type][order] -> rule
  std::vector<std::vector<IntegrationRule>> tet_rules_;
  std::vector<std::vector<IntegrationRule>> tri_rules_;
  std::vector<std::vector<IntegrationRule>> seg_rules_;
  std::vector<std::vector<IntegrationRule>> hex_rules_;
  std::vector<std::vector<IntegrationRule>> pyr_rules_;
};

// ============================================================================
// ElementTransformation - 单元坐标变换
// ============================================================================

class ElementTransformation {
 public:
  ElementTransformation() = default;

  void SetMesh(const Mesh* mesh) { mesh_ = mesh; }
  void SetElement(const ElementGroup* group, int elem_idx);

  // Reference coordinates -> Physical coordinates
  Vec3 Transform(const IntegrationPoint& ip) const;

  // Jacobian matrix: J = d(x,y,z)/d(ξ,η,ζ)
  Mat3 CalcJacobian(const IntegrationPoint& ip) const;

  // Inverse Jacobian: J^{-1}
  Mat3 CalcInverseJacobian(const IntegrationPoint& ip) const;

  // Jacobian determinant
  double CalcWeight(const IntegrationPoint& ip) const;

  // Get element attribute (domain ID)
  int GetAttribute() const { return attribute_; }

  // Get element vertices
  std::vector<int> GetElementVertices() const;

 private:
  const Mesh* mesh_ = nullptr;
  const ElementGroup* group_ = nullptr;
  int elem_idx_ = -1;
  int attribute_ = -1;  // Domain ID

  // Cached node coordinates for current element
  std::vector<Vec3> node_coords_;
};

// ============================================================================
// FiniteElement - 有限元基类
// ============================================================================

class FiniteElement {
 public:
  virtual ~FiniteElement() = default;

  GeometryType GetGeomType() const { return geom_type_; }
  int GetDim() const { return dim_; }
  int GetDof() const { return dof_; }
  int GetOrder() const { return order_; }

  // Calculate shape functions at reference point (returns dof_ values)
  virtual void CalcShape(const IntegrationPoint& ip,
                          double* shape) const = 0;

  // Calculate shape function gradients in reference coordinates
  // Returns [dof_][dim_] matrix (stored row-major)
  virtual void CalcDShape(const IntegrationPoint& ip,
                          double* dshape) const = 0;

  // Get the default integration order for this element
  virtual int GetDefaultIntegrationOrder() const { return 2 * order_; }

 protected:
  FiniteElement(GeometryType geom, int dim, int dof, int order)
      : geom_type_(geom), dim_(dim), dof_(dof), order_(order) {}

  GeometryType geom_type_;
  int dim_ = 0;
  int dof_ = 0;
  int order_ = 0;
};

// ============================================================================
// H1FiniteElement - H1 (Lagrange) 有限元
// ============================================================================

class H1FiniteElement : public FiniteElement {
 public:
  static std::unique_ptr<H1FiniteElement> Create(GeometryType geom, int order);

 protected:
  H1FiniteElement(GeometryType geom, int dim, int dof, int order)
      : FiniteElement(geom, dim, dof, order) {}
};

// ============================================================================
// H1_TetrahedronElement - 四面体 Lagrange 元
// ============================================================================

class H1_TetrahedronElement : public H1FiniteElement {
 public:
  explicit H1_TetrahedronElement(int order);

  void CalcShape(const IntegrationPoint& ip, double* shape) const override;
  void CalcDShape(const IntegrationPoint& ip, double* dshape) const override;

 private:
  void Init(int order);

  // Node positions in reference tetrahedron
  std::vector<std::array<double, 3>> nodes_;
};

// ============================================================================
// H1_TriangleElement - 三角形 Lagrange 元
// ============================================================================

class H1_TriangleElement : public H1FiniteElement {
 public:
  explicit H1_TriangleElement(int order);

  void CalcShape(const IntegrationPoint& ip, double* shape) const override;
  void CalcDShape(const IntegrationPoint& ip, double* dshape) const override;

 private:
  void Init(int order);

  // Node positions in reference triangle
  std::vector<std::array<double, 2>> nodes_;
};

// ============================================================================
// H1_SegmentElement - 线段 Lagrange 元
// ============================================================================

class H1_SegmentElement : public H1FiniteElement {
 public:
  explicit H1_SegmentElement(int order);

  void CalcShape(const IntegrationPoint& ip, double* shape) const override;
  void CalcDShape(const IntegrationPoint& ip, double* dshape) const override;

 private:
  void Init(int order);

  // Node positions in reference segment [0, 1]
  std::vector<double> nodes_;
};

// ============================================================================
// H1_HexahedronElement - 六面体 Lagrange 元
// ============================================================================

class H1_HexahedronElement : public H1FiniteElement {
 public:
  explicit H1_HexahedronElement(int order);

  void CalcShape(const IntegrationPoint& ip, double* shape) const override;
  void CalcDShape(const IntegrationPoint& ip, double* dshape) const override;

 private:
  void Init(int order);

  // Node positions in reference hexahedron [0,1]^3
  std::vector<std::array<double, 3>> nodes_;
};

// ============================================================================
// H1_PyramidElement - 金字塔 Lagrange 元
// ============================================================================

class H1_PyramidElement : public H1FiniteElement {
 public:
  explicit H1_PyramidElement(int order);

  void CalcShape(const IntegrationPoint& ip, double* shape) const override;
  void CalcDShape(const IntegrationPoint& ip, double* dshape) const override;

 private:
  void Init(int order);

  // Node positions in reference pyramid
  std::vector<std::array<double, 3>> nodes_;
};

// ============================================================================
// FiniteElementCollection - 有限元集合
// ============================================================================

class FiniteElementCollection {
 public:
  virtual ~FiniteElementCollection() = default;

  virtual const FiniteElement* GetFiniteElement(GeometryType geom) const = 0;
  virtual FESpaceKind GetKind() const = 0;
  virtual int GetOrder() const = 0;
};

// ============================================================================
// H1_FECollection - H1 有限元集合
// ============================================================================

class H1_FECollection : public FiniteElementCollection {
 public:
  explicit H1_FECollection(int order, int dim = 3);

  const FiniteElement* GetFiniteElement(GeometryType geom) const override;
  FESpaceKind GetKind() const override { return FESpaceKind::kH1; }
  int GetOrder() const override { return order_; }

 private:
  int order_;
  int dim_;

  std::unique_ptr<H1_TetrahedronElement> tet_fe_;
  std::unique_ptr<H1_TriangleElement> tri_fe_;
  std::unique_ptr<H1_SegmentElement> seg_fe_;
  std::unique_ptr<H1_HexahedronElement> hex_fe_;
  std::unique_ptr<H1_PyramidElement> pyr_fe_;
};

// ============================================================================
// Utility functions
// ============================================================================

namespace fe {

// Get geometry type from dimension
inline GeometryType GetBoundaryGeomType(GeometryType vol_type, int face_idx) {
  switch (vol_type) {
    case GeometryType::kTetrahedron:
      return GeometryType::kTriangle;
    case GeometryType::kHexahedron:
      return GeometryType::kQuadrilateral;
    case GeometryType::kWedge:
      return (face_idx < 2) ? GeometryType::kTriangle
                            : GeometryType::kQuadrilateral;
    case GeometryType::kPyramid:
      return (face_idx == 0) ? GeometryType::kQuadrilateral
                             : GeometryType::kTriangle;
    default:
      return GeometryType::kPoint;
  }
}

// Get number of faces for a geometry type
inline int NumFaces(GeometryType type) {
  switch (type) {
    case GeometryType::kTetrahedron:
      return 4;
    case GeometryType::kHexahedron:
      return 6;
    case GeometryType::kWedge:
      return 5;
    case GeometryType::kPyramid:
      return 5;
    default:
      return 0;
  }
}

}  // namespace fe

}  // namespace mpfem
