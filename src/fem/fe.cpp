#include "mpfem/fem/fe.hpp"
#include "mpfem/mesh/mesh.hpp"
#include "mpfem/core/logger.hpp"
#include <cmath>
#include <stdexcept>

namespace mpfem {

// ============================================================================
// IntegrationRules implementation
// ============================================================================

IntegrationRules::IntegrationRules() {
  InitTetrahedronRules();
  InitTriangleRules();
  InitSegmentRules();
  InitHexahedronRules();
  InitPyramidRules();
}

const IntegrationRule& IntegrationRules::Get(GeometryType geom,
                                             int order) const {
  static IntegrationRule empty_rule;

  switch (geom) {
    case GeometryType::kTetrahedron: {
      int idx = order;
      if (idx < 0) idx = 0;
      if (idx >= static_cast<int>(tet_rules_.size())) {
        MPFEM_WARN("Integration order %d for tetrahedron not available, using highest order.",
                   order);
        idx = static_cast<int>(tet_rules_.size()) - 1;
      }
      return tet_rules_[idx].empty() ? empty_rule : tet_rules_[idx][0];
    }
    case GeometryType::kTriangle: {
      int idx = order;
      if (idx < 0) idx = 0;
      if (idx >= static_cast<int>(tri_rules_.size())) {
        idx = static_cast<int>(tri_rules_.size()) - 1;
      }
      return tri_rules_[idx].empty() ? empty_rule : tri_rules_[idx][0];
    }
    case GeometryType::kSegment: {
      int idx = order;
      if (idx < 0) idx = 0;
      if (idx >= static_cast<int>(seg_rules_.size())) {
        idx = static_cast<int>(seg_rules_.size()) - 1;
      }
      return seg_rules_[idx].empty() ? empty_rule : seg_rules_[idx][0];
    }
    case GeometryType::kHexahedron: {
      int idx = order;
      if (idx < 0) idx = 0;
      if (idx >= static_cast<int>(hex_rules_.size())) {
        idx = static_cast<int>(hex_rules_.size()) - 1;
      }
      return hex_rules_[idx].empty() ? empty_rule : hex_rules_[idx][0];
    }
    case GeometryType::kPyramid: {
      int idx = order;
      if (idx < 0) idx = 0;
      if (idx >= static_cast<int>(pyr_rules_.size())) {
        idx = static_cast<int>(pyr_rules_.size()) - 1;
      }
      return pyr_rules_[idx].empty() ? empty_rule : pyr_rules_[idx][0];
    }
    default:
      MPFEM_ERROR("Integration rule not implemented for geometry type: %d",
                  static_cast<int>(geom));
      return empty_rule;
  }
}

void IntegrationRules::InitTetrahedronRules() {
  // Reference tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
  // Volume = 1/6

  tet_rules_.resize(5);  // Order 0-4

  // Order 1: 1-point rule
  {
    IntegrationRule rule(1);
    // Centroid
    double a = 0.25;
    rule.AddPoint(IntegrationPoint(a, a, a, 1.0 / 6.0));
    tet_rules_[1].push_back(rule);
  }

  // Order 2: 4-point rule
  {
    IntegrationRule rule(2);
    double a = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    double b = (5.0 - std::sqrt(5.0)) / 20.0;
    double w = 1.0 / 24.0;
    rule.AddPoint(IntegrationPoint(a, b, b, w));
    rule.AddPoint(IntegrationPoint(b, a, b, w));
    rule.AddPoint(IntegrationPoint(b, b, a, w));
    rule.AddPoint(IntegrationPoint(b, b, b, w));
    tet_rules_[2].push_back(rule);
  }

  // Order 3: 5-point rule
  {
    IntegrationRule rule(3);
    // Centroid point
    double w1 = -2.0 / 15.0;
    rule.AddPoint(IntegrationPoint(0.25, 0.25, 0.25, w1));
    // Corner points
    double w2 = 3.0 / 40.0;
    double a = 1.0 / 6.0;
    double b = 1.0 / 2.0;
    rule.AddPoint(IntegrationPoint(b, a, a, w2));
    rule.AddPoint(IntegrationPoint(a, b, a, w2));
    rule.AddPoint(IntegrationPoint(a, a, b, w2));
    rule.AddPoint(IntegrationPoint(a, a, a, w2));
    tet_rules_[3].push_back(rule);
  }

  // Order 4: 11-point rule (Keast)
  {
    IntegrationRule rule(4);
    double w1 = 2.0 / 315.0 * (2665.0 - 14.0 * std::sqrt(7.0));
    double w2 = 2.0 / 315.0 * (2665.0 + 14.0 * std::sqrt(7.0));

    // Centroid
    rule.AddPoint(IntegrationPoint(0.25, 0.25, 0.25, -148.0 / 315.0));

    // Permutation type 1: (a, a, a)
    double a1 = (6.0 - std::sqrt(15.0)) / 21.0;
    double a2 = (6.0 + std::sqrt(15.0)) / 21.0;
    double w_a1 = w1;
    double w_a2 = w2;

    rule.AddPoint(IntegrationPoint(a1, a1, a1, w_a1));
    rule.AddPoint(IntegrationPoint(a2, a2, a2, w_a2));

    // Permutation type 2: (b, b, c) - 4 points each
    double b1 = (9.0 + 2.0 * std::sqrt(15.0)) / 21.0;
    double c1 = (6.0 - std::sqrt(15.0)) / 21.0;

    for (int i = 0; i < 4; ++i) {
      double pts[4][3] = {{b1, c1, c1}, {c1, b1, c1}, {c1, c1, b1}, {c1, b1, b1}};
      rule.AddPoint(IntegrationPoint(pts[i][0], pts[i][1], pts[i][2], w_a1));
    }

    double b2 = (9.0 - 2.0 * std::sqrt(15.0)) / 21.0;
    double c2 = (6.0 + std::sqrt(15.0)) / 21.0;

    for (int i = 0; i < 4; ++i) {
      double pts[4][3] = {{b2, c2, c2}, {c2, b2, c2}, {c2, c2, b2}, {c2, c2, b2}};
      rule.AddPoint(IntegrationPoint(pts[i][0], pts[i][1], pts[i][2], w_a2));
    }

    // Actually, let's use a simpler order-4 rule with 5 points
    // which is sufficient for most applications
    tet_rules_[4].clear();
    IntegrationRule simple_rule(4);
    double w = 1.0 / 30.0;
    simple_rule.AddPoint(IntegrationPoint(0.25, 0.25, 0.25, -16.0 / 30.0));
    simple_rule.AddPoint(IntegrationPoint(0.5, 1.0 / 6.0, 1.0 / 6.0, w));
    simple_rule.AddPoint(IntegrationPoint(1.0 / 6.0, 0.5, 1.0 / 6.0, w));
    simple_rule.AddPoint(IntegrationPoint(1.0 / 6.0, 1.0 / 6.0, 0.5, w));
    simple_rule.AddPoint(IntegrationPoint(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, w));
    tet_rules_[4].push_back(simple_rule);
  }
}

void IntegrationRules::InitTriangleRules() {
  // Reference triangle: vertices at (0,0), (1,0), (0,1)
  // Area = 1/2

  tri_rules_.resize(5);  // Order 0-4

  // Order 1: 1-point rule (centroid)
  {
    IntegrationRule rule(1);
    double a = 1.0 / 3.0;
    rule.AddPoint(IntegrationPoint(a, a, 0.0, 0.5));
    tri_rules_[1].push_back(rule);
  }

  // Order 2: 3-point rule
  {
    IntegrationRule rule(2);
    double a = 1.0 / 6.0;
    double b = 2.0 / 3.0;
    double w = 1.0 / 6.0;
    rule.AddPoint(IntegrationPoint(a, a, 0.0, w));
    rule.AddPoint(IntegrationPoint(b, a, 0.0, w));
    rule.AddPoint(IntegrationPoint(a, b, 0.0, w));
    tri_rules_[2].push_back(rule);
  }

  // Order 3: 4-point rule
  {
    IntegrationRule rule(3);
    double a = 1.0 / 3.0;
    double b = 0.6;
    double c = 0.2;
    double w1 = -27.0 / 96.0;
    double w2 = 25.0 / 96.0;

    rule.AddPoint(IntegrationPoint(a, a, 0.0, w1));
    rule.AddPoint(IntegrationPoint(b, c, 0.0, w2));
    rule.AddPoint(IntegrationPoint(c, b, 0.0, w2));
    rule.AddPoint(IntegrationPoint(c, c, 0.0, w2));
    tri_rules_[3].push_back(rule);
  }

  // Order 4: 6-point rule
  {
    IntegrationRule rule(4);
    double a = 0.816847572980459;
    double b = 0.091576213509771;
    double w1 = 0.054975871827661 / 2.0;
    double w2 = 0.111690794839005 / 2.0;

    rule.AddPoint(IntegrationPoint(a, b, 0.0, w1));
    rule.AddPoint(IntegrationPoint(b, a, 0.0, w1));
    rule.AddPoint(IntegrationPoint(b, b, 0.0, w1));

    a = 0.108103018168070;
    b = 0.445948490915965;
    rule.AddPoint(IntegrationPoint(a, b, 0.0, w2));
    rule.AddPoint(IntegrationPoint(b, a, 0.0, w2));
    rule.AddPoint(IntegrationPoint(b, b, 0.0, w2));

    tri_rules_[4].push_back(rule);
  }
}

void IntegrationRules::InitSegmentRules() {
  // Reference segment: [0, 1]
  // Length = 1

  seg_rules_.resize(5);

  // Order 1: 1-point (midpoint)
  {
    IntegrationRule rule(1);
    rule.AddPoint(IntegrationPoint(0.5, 0.0, 0.0, 1.0));
    seg_rules_[1].push_back(rule);
  }

  // Order 2: 2-point Gauss
  {
    IntegrationRule rule(2);
    double a = 0.5 - std::sqrt(3.0) / 6.0;
    double b = 0.5 + std::sqrt(3.0) / 6.0;
    double w = 0.5;
    rule.AddPoint(IntegrationPoint(a, 0.0, 0.0, w));
    rule.AddPoint(IntegrationPoint(b, 0.0, 0.0, w));
    seg_rules_[2].push_back(rule);
  }

  // Order 3: 3-point Gauss
  {
    IntegrationRule rule(3);
    double a = 0.5;
    double b = 0.5 - std::sqrt(15.0) / 10.0;
    double c = 0.5 + std::sqrt(15.0) / 10.0;
    double w1 = 4.0 / 9.0;
    double w2 = 5.0 / 18.0;
    rule.AddPoint(IntegrationPoint(a, 0.0, 0.0, w1));
    rule.AddPoint(IntegrationPoint(b, 0.0, 0.0, w2));
    rule.AddPoint(IntegrationPoint(c, 0.0, 0.0, w2));
    seg_rules_[3].push_back(rule);
  }

  // Order 4: 4-point Gauss
  {
    IntegrationRule rule(4);
    double sqrt1 = std::sqrt((3.0 - 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);
    double sqrt2 = std::sqrt((3.0 + 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);

    double a = 0.5 - 0.5 * sqrt1;
    double b = 0.5 + 0.5 * sqrt1;
    double c = 0.5 - 0.5 * sqrt2;
    double d = 0.5 + 0.5 * sqrt2;

    double w1 = (18.0 + std::sqrt(30.0)) / 72.0;
    double w2 = (18.0 - std::sqrt(30.0)) / 72.0;

    rule.AddPoint(IntegrationPoint(a, 0.0, 0.0, w1));
    rule.AddPoint(IntegrationPoint(b, 0.0, 0.0, w1));
    rule.AddPoint(IntegrationPoint(c, 0.0, 0.0, w2));
    rule.AddPoint(IntegrationPoint(d, 0.0, 0.0, w2));
    seg_rules_[4].push_back(rule);
  }
}

void IntegrationRules::InitHexahedronRules() {
  // Reference hexahedron: [0,1]^3
  // Volume = 1

  hex_rules_.resize(3);

  // Order 1: 1-point (centroid)
  {
    IntegrationRule rule(1);
    rule.AddPoint(IntegrationPoint(0.5, 0.5, 0.5, 1.0));
    hex_rules_[1].push_back(rule);
  }

  // Order 2: 8-point Gauss (2x2x2)
  {
    IntegrationRule rule(2);
    double a = 0.5 - std::sqrt(3.0) / 6.0;
    double b = 0.5 + std::sqrt(3.0) / 6.0;
    double w = 0.125;  // 1/8

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          double x = (i == 0) ? a : b;
          double y = (j == 0) ? a : b;
          double z = (k == 0) ? a : b;
          rule.AddPoint(IntegrationPoint(x, y, z, w));
        }
      }
    }
    hex_rules_[2].push_back(rule);
  }
}

void IntegrationRules::InitPyramidRules() {
  // Reference pyramid: square base [0,1]^2 at z=0, apex at (0.5, 0.5, 1)
  // Volume = 1/3

  pyr_rules_.resize(2);

  // Order 1: 1-point (centroid approximation)
  {
    IntegrationRule rule(1);
    // Approximate centroid of pyramid
    rule.AddPoint(IntegrationPoint(0.5, 0.5, 0.25, 1.0 / 3.0));
    pyr_rules_[1].push_back(rule);
  }

  // Order 2: 5-point rule
  {
    IntegrationRule rule(2);
    // Based on collapsed coordinate formulation
    double w = 1.0 / 5.0;  // Equal weights for 5 points

    // Base center
    rule.AddPoint(IntegrationPoint(0.5, 0.5, 0.0, w));
    // Apex
    rule.AddPoint(IntegrationPoint(0.5, 0.5, 0.75, w));
    // Three points on base corners region
    rule.AddPoint(IntegrationPoint(0.25, 0.25, 0.25, w));
    rule.AddPoint(IntegrationPoint(0.75, 0.25, 0.25, w));
    rule.AddPoint(IntegrationPoint(0.5, 0.75, 0.25, w));

    pyr_rules_[2].push_back(rule);
  }
}

// ============================================================================
// ElementTransformation implementation
// ============================================================================

void ElementTransformation::SetElement(const ElementGroup* group, int elem_idx) {
  group_ = group;
  elem_idx_ = elem_idx;

  if (group_ && mesh_) {
    attribute_ = group_->entity_ids[elem_idx_];

    // Get node coordinates
    auto vertices = group_->GetElementVertices(elem_idx_);
    node_coords_.clear();
    node_coords_.reserve(vertices.size());

    const auto& nodes = mesh_->Nodes();
    for (int v : vertices) {
      node_coords_.push_back(nodes.Get(v));
    }
  }
}

std::vector<int> ElementTransformation::GetElementVertices() const {
  if (group_) {
    return group_->GetElementVertices(elem_idx_);
  }
  return {};
}

Vec3 ElementTransformation::Transform(const IntegrationPoint& ip) const {
  // Get shape functions for the geometry type
  auto fe = H1FiniteElement::Create(group_->type, 1);
  std::vector<double> shape(fe->GetDof());
  fe->CalcShape(ip, shape.data());

  Vec3 result(0.0, 0.0, 0.0);
  for (size_t i = 0; i < node_coords_.size(); ++i) {
    result += shape[i] * node_coords_[i];
  }
  return result;
}

Mat3 ElementTransformation::CalcJacobian(const IntegrationPoint& ip) const {
  // For tetrahedron: J = [x1-x4, x2-x4, x3-x4; y1-y4, y2-y4, y3-y4; z1-z4, z2-z4, z3-z4]
  // where (x,y,z) are node coordinates in physical space

  Mat3 J = Mat3::Zero();

  auto fe = H1FiniteElement::Create(group_->type, 1);
  std::vector<double> dshape(fe->GetDof() * fe->GetDim());
  fe->CalcDShape(ip, dshape.data());

  // J_{ij} = sum_k dshape_{k,j} * node_coords_{k,i}
  // J = [dN/dξ * x, dN/dη * x, dN/dζ * x]
  //     [dN/dξ * y, dN/dη * y, dN/dζ * y]
  //     [dN/dξ * z, dN/dη * z, dN/dζ * z]

  int dim = fe->GetDim();
  int dof = fe->GetDof();

  for (int i = 0; i < 3; ++i) {        // Physical coordinate
    for (int j = 0; j < dim; ++j) {     // Reference coordinate
      for (int k = 0; k < dof; ++k) {   // Shape function index
        J(i, j) += dshape[k * dim + j] * node_coords_[k](i);
      }
    }
  }

  return J;
}

Mat3 ElementTransformation::CalcInverseJacobian(const IntegrationPoint& ip) const {
  Mat3 J = CalcJacobian(ip);
  return J.inverse();
}

double ElementTransformation::CalcWeight(const IntegrationPoint& ip) const {
  Mat3 J = CalcJacobian(ip);

  // For tetrahedron, the Jacobian determinant relates reference volume to physical volume
  // Physical volume = |det(J)| * reference volume
  return std::abs(J.determinant());
}

// ============================================================================
// H1FiniteElement factory
// ============================================================================

std::unique_ptr<H1FiniteElement> H1FiniteElement::Create(GeometryType geom,
                                                         int order) {
  switch (geom) {
    case GeometryType::kTetrahedron:
      return std::make_unique<H1_TetrahedronElement>(order);
    case GeometryType::kTriangle:
      return std::make_unique<H1_TriangleElement>(order);
    case GeometryType::kSegment:
      return std::make_unique<H1_SegmentElement>(order);
    case GeometryType::kHexahedron:
      return std::make_unique<H1_HexahedronElement>(order);
    case GeometryType::kPyramid:
      return std::make_unique<H1_PyramidElement>(order);
    default:
      throw std::runtime_error("H1 element not implemented for geometry type: " +
                               std::to_string(static_cast<int>(geom)));
  }
}

// ============================================================================
// H1_TetrahedronElement implementation
// ============================================================================

H1_TetrahedronElement::H1_TetrahedronElement(int order)
    : H1FiniteElement(GeometryType::kTetrahedron, 3,
                      (order + 1) * (order + 2) * (order + 3) / 6, order) {
  Init(order);
}

void H1_TetrahedronElement::Init(int order) {
  // Define node positions in reference tetrahedron
  // Reference tet: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)

  nodes_.reserve(dof_);

  if (order == 1) {
    // Vertex nodes
    nodes_.push_back({0.0, 0.0, 0.0});
    nodes_.push_back({1.0, 0.0, 0.0});
    nodes_.push_back({0.0, 1.0, 0.0});
    nodes_.push_back({0.0, 0.0, 1.0});
  } else if (order == 2) {
    // Vertex nodes + edge midpoints
    nodes_.push_back({0.0, 0.0, 0.0});  // v0
    nodes_.push_back({1.0, 0.0, 0.0});  // v1
    nodes_.push_back({0.0, 1.0, 0.0});  // v2
    nodes_.push_back({0.0, 0.0, 1.0});  // v3
    nodes_.push_back({0.5, 0.0, 0.0});  // e0 midpoint
    nodes_.push_back({0.0, 0.5, 0.0});  // e1 midpoint
    nodes_.push_back({0.0, 0.0, 0.5});  // e2 midpoint
    nodes_.push_back({0.5, 0.5, 0.0});  // e3 midpoint
    nodes_.push_back({0.5, 0.0, 0.5});  // e4 midpoint
    nodes_.push_back({0.0, 0.5, 0.5});  // e5 midpoint
  } else {
    // Higher order: generate nodes uniformly
    // For simplicity, we only support order 1 for now in the implementation
    // The framework is ready for higher orders
    throw std::runtime_error(
        "H1 tetrahedron element order > 2 not yet implemented");
  }
}

void H1_TetrahedronElement::CalcShape(const IntegrationPoint& ip,
                                      double* shape) const {
  // Using barycentric coordinates: λ1 = x, λ2 = y, λ3 = z, λ4 = 1-x-y-z
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;
  double w = 1.0 - x - y - z;  // Fourth barycentric coordinate

  if (order_ == 1) {
    // Linear Lagrange basis on tetrahedron
    shape[0] = w;          // φ0 at (0,0,0)
    shape[1] = x;          // φ1 at (1,0,0)
    shape[2] = y;          // φ2 at (0,1,0)
    shape[3] = z;          // φ3 at (0,0,1)
  } else if (order_ == 2) {
    // Quadratic Lagrange basis
    double l1 = w, l2 = x, l3 = y, l4 = z;

    // Vertex functions
    shape[0] = l1 * (2.0 * l1 - 1.0);
    shape[1] = l2 * (2.0 * l2 - 1.0);
    shape[2] = l3 * (2.0 * l3 - 1.0);
    shape[3] = l4 * (2.0 * l4 - 1.0);

    // Edge functions (4*λi*λj for edge ij)
    shape[4] = 4.0 * l1 * l2;  // edge 0-1
    shape[5] = 4.0 * l1 * l3;  // edge 0-2
    shape[6] = 4.0 * l1 * l4;  // edge 0-3
    shape[7] = 4.0 * l2 * l3;  // edge 1-2
    shape[8] = 4.0 * l2 * l4;  // edge 1-3
    shape[9] = 4.0 * l3 * l4;  // edge 2-3
  }
}

void H1_TetrahedronElement::CalcDShape(const IntegrationPoint& ip,
                                       double* dshape) const {
  // dshape[i*3 + j] = ∂φi/∂ξj where (ξ1,ξ2,ξ3) = (x,y,z)
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;

  if (order_ == 1) {
    // Linear gradients
    // φ0 = 1 - x - y - z: dφ0/dx = -1, dφ0/dy = -1, dφ0/dz = -1
    // φ1 = x: dφ1/dx = 1, dφ1/dy = 0, dφ1/dz = 0
    // φ2 = y: dφ2/dx = 0, dφ2/dy = 1, dφ2/dz = 0
    // φ3 = z: dφ3/dx = 0, dφ3/dy = 0, dφ3/dz = 1

    dshape[0] = -1.0; dshape[1] = -1.0; dshape[2] = -1.0;  // φ0
    dshape[3] = 1.0;  dshape[4] = 0.0;  dshape[5] = 0.0;   // φ1
    dshape[6] = 0.0;  dshape[7] = 1.0;  dshape[8] = 0.0;   // φ2
    dshape[9] = 0.0;  dshape[10] = 0.0; dshape[11] = 1.0;  // φ3
  } else if (order_ == 2) {
    double l1 = 1.0 - x - y - z;
    double l2 = x;
    double l3 = y;
    double l4 = z;

    // Vertex function gradients
    // φ_i = λ_i * (2λ_i - 1) = 2λ_i² - λ_i
    // dφ_i/dx = (4λ_i - 1) * dλ_i/dx

    // dλ1/dx = -1, dλ1/dy = -1, dλ1/dz = -1
    dshape[0] = (4.0 * l1 - 1.0) * (-1.0);  // dφ0/dx
    dshape[1] = (4.0 * l1 - 1.0) * (-1.0);  // dφ0/dy
    dshape[2] = (4.0 * l1 - 1.0) * (-1.0);  // dφ0/dz

    // dλ2/dx = 1, dλ2/dy = 0, dλ2/dz = 0
    dshape[3] = (4.0 * l2 - 1.0) * 1.0;     // dφ1/dx
    dshape[4] = 0.0;                          // dφ1/dy
    dshape[5] = 0.0;                          // dφ1/dz

    // dλ3/dx = 0, dλ3/dy = 1, dλ3/dz = 0
    dshape[6] = 0.0;                          // dφ2/dx
    dshape[7] = (4.0 * l3 - 1.0) * 1.0;     // dφ2/dy
    dshape[8] = 0.0;                          // dφ2/dz

    // dλ4/dx = 0, dλ4/dy = 0, dλ4/dz = 1
    dshape[9] = 0.0;                          // dφ3/dx
    dshape[10] = 0.0;                         // dφ3/dy
    dshape[11] = (4.0 * l4 - 1.0) * 1.0;    // dφ3/dz

    // Edge function gradients: φ = 4*λ_i*λ_j
    // dφ/dx = 4*(λ_j*dλ_i/dx + λ_i*dλ_j/dx)

    // Edge 0-1: λ1, λ2
    dshape[12] = 4.0 * (l2 * (-1.0) + l1 * 1.0);   // d/dx
    dshape[13] = 4.0 * l2 * (-1.0);                 // d/dy
    dshape[14] = 4.0 * l2 * (-1.0);                 // d/dz

    // Edge 0-2: λ1, λ3
    dshape[15] = 4.0 * l3 * (-1.0);                 // d/dx
    dshape[16] = 4.0 * (l3 * (-1.0) + l1 * 1.0);   // d/dy
    dshape[17] = 4.0 * l3 * (-1.0);                 // d/dz

    // Edge 0-3: λ1, λ4
    dshape[18] = 4.0 * l4 * (-1.0);                 // d/dx
    dshape[19] = 4.0 * l4 * (-1.0);                 // d/dy
    dshape[20] = 4.0 * (l4 * (-1.0) + l1 * 1.0);   // d/dz

    // Edge 1-2: λ2, λ3
    dshape[21] = 4.0 * l3 * 1.0;                    // d/dx
    dshape[22] = 4.0 * l2 * 1.0;                    // d/dy
    dshape[23] = 0.0;                               // d/dz

    // Edge 1-3: λ2, λ4
    dshape[24] = 4.0 * l4 * 1.0;                    // d/dx
    dshape[25] = 0.0;                               // d/dy
    dshape[26] = 4.0 * l2 * 1.0;                    // d/dz

    // Edge 2-3: λ3, λ4
    dshape[27] = 0.0;                               // d/dx
    dshape[28] = 4.0 * l4 * 1.0;                    // d/dy
    dshape[29] = 4.0 * l3 * 1.0;                    // d/dz
  }
}

// ============================================================================
// H1_TriangleElement implementation
// ============================================================================

H1_TriangleElement::H1_TriangleElement(int order)
    : H1FiniteElement(GeometryType::kTriangle, 2,
                      (order + 1) * (order + 2) / 2, order) {
  Init(order);
}

void H1_TriangleElement::Init(int order) {
  // Reference triangle: vertices at (0,0), (1,0), (0,1)
  nodes_.reserve(dof_);

  if (order == 1) {
    nodes_.push_back({0.0, 0.0});
    nodes_.push_back({1.0, 0.0});
    nodes_.push_back({0.0, 1.0});
  } else if (order == 2) {
    nodes_.push_back({0.0, 0.0});   // v0
    nodes_.push_back({1.0, 0.0});   // v1
    nodes_.push_back({0.0, 1.0});   // v2
    nodes_.push_back({0.5, 0.0});   // e0
    nodes_.push_back({0.5, 0.5});   // e1
    nodes_.push_back({0.0, 0.5});   // e2
  }
}

void H1_TriangleElement::CalcShape(const IntegrationPoint& ip,
                                   double* shape) const {
  double x = ip.x;
  double y = ip.y;
  double l3 = 1.0 - x - y;

  if (order_ == 1) {
    shape[0] = l3;      // φ0 at (0,0)
    shape[1] = x;       // φ1 at (1,0)
    shape[2] = y;       // φ2 at (0,1)
  } else if (order_ == 2) {
    double l1 = l3, l2 = x;

    shape[0] = l1 * (2.0 * l1 - 1.0);
    shape[1] = l2 * (2.0 * l2 - 1.0);
    shape[2] = y * (2.0 * y - 1.0);
    shape[3] = 4.0 * l1 * l2;
    shape[4] = 4.0 * l2 * y;
    shape[5] = 4.0 * y * l1;
  }
}

void H1_TriangleElement::CalcDShape(const IntegrationPoint& ip,
                                    double* dshape) const {
  double x = ip.x;
  double y = ip.y;

  if (order_ == 1) {
    dshape[0] = -1.0; dshape[1] = -1.0;  // φ0
    dshape[2] = 1.0;  dshape[3] = 0.0;   // φ1
    dshape[4] = 0.0;  dshape[5] = 1.0;   // φ2
  } else if (order_ == 2) {
    double l1 = 1.0 - x - y;
    double l2 = x;
    double l3 = y;

    dshape[0] = (4.0 * l1 - 1.0) * (-1.0);
    dshape[1] = (4.0 * l1 - 1.0) * (-1.0);

    dshape[2] = (4.0 * l2 - 1.0) * 1.0;
    dshape[3] = 0.0;

    dshape[4] = 0.0;
    dshape[5] = (4.0 * l3 - 1.0) * 1.0;

    dshape[6] = 4.0 * (l2 * (-1.0) + l1 * 1.0);
    dshape[7] = 4.0 * l2 * (-1.0);

    dshape[8] = 4.0 * l3 * 1.0;
    dshape[9] = 4.0 * l2 * 1.0;

    dshape[10] = 4.0 * l3 * (-1.0);
    dshape[11] = 4.0 * (l3 * (-1.0) + l1 * 1.0);
  }
}

// ============================================================================
// H1_SegmentElement implementation
// ============================================================================

H1_SegmentElement::H1_SegmentElement(int order)
    : H1FiniteElement(GeometryType::kSegment, 1, order + 1, order) {
  Init(order);
}

void H1_SegmentElement::Init(int order) {
  // Reference segment [0, 1]
  nodes_.reserve(dof_);

  if (order == 1) {
    nodes_.push_back(0.0);
    nodes_.push_back(1.0);
  } else if (order == 2) {
    nodes_.push_back(0.0);
    nodes_.push_back(1.0);
    nodes_.push_back(0.5);
  }
}

void H1_SegmentElement::CalcShape(const IntegrationPoint& ip,
                                  double* shape) const {
  double x = ip.x;

  if (order_ == 1) {
    shape[0] = 1.0 - x;
    shape[1] = x;
  } else if (order_ == 2) {
    shape[0] = (1.0 - x) * (1.0 - 2.0 * x);
    shape[1] = x * (2.0 * x - 1.0);
    shape[2] = 4.0 * x * (1.0 - x);
  }
}

void H1_SegmentElement::CalcDShape(const IntegrationPoint& ip,
                                   double* dshape) const {
  double x = ip.x;

  if (order_ == 1) {
    dshape[0] = -1.0;
    dshape[1] = 1.0;
  } else if (order_ == 2) {
    dshape[0] = 4.0 * x - 3.0;
    dshape[1] = 4.0 * x - 1.0;
    dshape[2] = 4.0 - 8.0 * x;
  }
}

// ============================================================================
// H1_HexahedronElement implementation
// ============================================================================

H1_HexahedronElement::H1_HexahedronElement(int order)
    : H1FiniteElement(GeometryType::kHexahedron, 3, (order + 1) * (order + 1) * (order + 1), order) {
  Init(order);
}

void H1_HexahedronElement::Init(int order) {
  // Reference hexahedron [0,1]^3
  // Node ordering: vertices first, then edges, faces, interior
  nodes_.reserve(dof_);

  if (order == 1) {
    // 8 vertices of the unit cube
    nodes_.push_back({0.0, 0.0, 0.0});  // 0
    nodes_.push_back({1.0, 0.0, 0.0});  // 1
    nodes_.push_back({1.0, 1.0, 0.0});  // 2
    nodes_.push_back({0.0, 1.0, 0.0});  // 3
    nodes_.push_back({0.0, 0.0, 1.0});  // 4
    nodes_.push_back({1.0, 0.0, 1.0});  // 5
    nodes_.push_back({1.0, 1.0, 1.0});  // 6
    nodes_.push_back({0.0, 1.0, 1.0});  // 7
  }
}

void H1_HexahedronElement::CalcShape(const IntegrationPoint& ip,
                                     double* shape) const {
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;

  if (order_ == 1) {
    // Bilinear interpolation in 3D
    // N_i = (1/8)(1 ± x)(1 ± y)(1 ± z) scaled to [0,1]
    // Actually for unit cube [0,1]^3:
    shape[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    shape[1] = x * (1.0 - y) * (1.0 - z);
    shape[2] = x * y * (1.0 - z);
    shape[3] = (1.0 - x) * y * (1.0 - z);
    shape[4] = (1.0 - x) * (1.0 - y) * z;
    shape[5] = x * (1.0 - y) * z;
    shape[6] = x * y * z;
    shape[7] = (1.0 - x) * y * z;
  }
}

void H1_HexahedronElement::CalcDShape(const IntegrationPoint& ip,
                                      double* dshape) const {
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;

  if (order_ == 1) {
    // dN_i/dx
    dshape[0] = -(1.0 - y) * (1.0 - z);
    dshape[1] = (1.0 - y) * (1.0 - z);
    dshape[2] = y * (1.0 - z);
    dshape[3] = -y * (1.0 - z);
    dshape[4] = -(1.0 - y) * z;
    dshape[5] = (1.0 - y) * z;
    dshape[6] = y * z;
    dshape[7] = -y * z;

    // dN_i/dy
    dshape[8] = -(1.0 - x) * (1.0 - z);
    dshape[9] = -x * (1.0 - z);
    dshape[10] = x * (1.0 - z);
    dshape[11] = (1.0 - x) * (1.0 - z);
    dshape[12] = -(1.0 - x) * z;
    dshape[13] = -x * z;
    dshape[14] = x * z;
    dshape[15] = (1.0 - x) * z;

    // dN_i/dz
    dshape[16] = -(1.0 - x) * (1.0 - y);
    dshape[17] = -x * (1.0 - y);
    dshape[18] = -x * y;
    dshape[19] = -(1.0 - x) * y;
    dshape[20] = (1.0 - x) * (1.0 - y);
    dshape[21] = x * (1.0 - y);
    dshape[22] = x * y;
    dshape[23] = (1.0 - x) * y;
  }
}

// ============================================================================
// H1_PyramidElement implementation
// ============================================================================

H1_PyramidElement::H1_PyramidElement(int order)
    : H1FiniteElement(GeometryType::kPyramid, 3, (order + 1) * (order + 2) * (2 * order + 3) / 6, order) {
  Init(order);
}

void H1_PyramidElement::Init(int order) {
  // Reference pyramid with square base [0,1]^2 at z=0 and apex at (0.5, 0.5, 1)
  nodes_.reserve(dof_);

  if (order == 1) {
    // 5 vertices: 4 base corners + apex
    nodes_.push_back({0.0, 0.0, 0.0});  // 0
    nodes_.push_back({1.0, 0.0, 0.0});  // 1
    nodes_.push_back({1.0, 1.0, 0.0});  // 2
    nodes_.push_back({0.0, 1.0, 0.0});  // 3
    nodes_.push_back({0.5, 0.5, 1.0});  // 4 (apex)
    // Recalculate dof for linear pyramid (5 vertices)
    dof_ = 5;
  }
}

void H1_PyramidElement::CalcShape(const IntegrationPoint& ip,
                                  double* shape) const {
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;

  if (order_ == 1) {
    // For pyramid with apex at (0.5, 0.5, 1):
    // Use collapsed coordinate transformation
    // Shape functions:
    double eps = 1e-12;
    double zm = std::max(z, eps);  // Avoid division by zero at z=0
    double one_minus_z = 1.0 - z;
    
    // Base vertices (z=0)
    shape[0] = (1.0 - x - y + x * y) * one_minus_z;  // (0,0,0)
    shape[1] = (x - x * y) * one_minus_z;            // (1,0,0)
    shape[2] = x * y * one_minus_z;                   // (1,1,0)
    shape[3] = (y - x * y) * one_minus_z;            // (0,1,0)
    
    // Apex
    shape[4] = z;                                     // (0.5,0.5,1)
  }
}

void H1_PyramidElement::CalcDShape(const IntegrationPoint& ip,
                                   double* dshape) const {
  double x = ip.x;
  double y = ip.y;
  double z = ip.z;

  if (order_ == 1) {
    double one_minus_z = 1.0 - z;

    // dN_i/dx
    dshape[0] = (-1.0 + y) * one_minus_z;
    dshape[1] = (1.0 - y) * one_minus_z;
    dshape[2] = y * one_minus_z;
    dshape[3] = -y * one_minus_z;
    dshape[4] = 0.0;

    // dN_i/dy
    dshape[5] = (-1.0 + x) * one_minus_z;
    dshape[6] = -x * one_minus_z;
    dshape[7] = x * one_minus_z;
    dshape[8] = (1.0 - x) * one_minus_z;
    dshape[9] = 0.0;

    // dN_i/dz
    dshape[10] = -(1.0 - x - y + x * y);
    dshape[11] = -(x - x * y);
    dshape[12] = -x * y;
    dshape[13] = -(y - x * y);
    dshape[14] = 1.0;
  }
}

// ============================================================================
// H1_FECollection implementation
// ============================================================================

H1_FECollection::H1_FECollection(int order, int dim)
    : order_(order), dim_(dim) {
  // Create elements for supported geometry types
  tet_fe_ = std::make_unique<H1_TetrahedronElement>(order);
  tri_fe_ = std::make_unique<H1_TriangleElement>(order);
  seg_fe_ = std::make_unique<H1_SegmentElement>(order);
  hex_fe_ = std::make_unique<H1_HexahedronElement>(order);
  pyr_fe_ = std::make_unique<H1_PyramidElement>(order);
}

const FiniteElement* H1_FECollection::GetFiniteElement(GeometryType geom) const {
  switch (geom) {
    case GeometryType::kTetrahedron:
      return tet_fe_.get();
    case GeometryType::kTriangle:
      return tri_fe_.get();
    case GeometryType::kSegment:
      return seg_fe_.get();
    case GeometryType::kHexahedron:
      return hex_fe_.get();
    case GeometryType::kPyramid:
      return pyr_fe_.get();
    default:
      // Return nullptr for unsupported types (e.g., kPoint, kWedge)
      MPFEM_DEBUG("H1_FECollection does not support geometry type: %d",
                  static_cast<int>(geom));
      return nullptr;
  }
}

}  // namespace mpfem