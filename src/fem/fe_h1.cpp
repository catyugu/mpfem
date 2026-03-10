/**
 * @file fe_h1.cpp
 * @brief Implementation of H1 Lagrange finite elements
 */

#include "fe_h1.hpp"
#include "core/exception.hpp"
#include <cmath>

namespace mpfem {

// ============================================================
// FE_Segment
// ============================================================

FE_Segment::FE_Segment(int degree, int n_components) {
    degree_ = degree;
    dim_ = 1;
    n_components_ = n_components;
    dofs_per_cell_ = (degree + 1) * n_components;
    geom_type_ = GeometryType::Segment;
    quad_order_ = 2 * degree;  // Exact for polynomials of degree 2*degree-1
    
    initialize_quadrature();
}

void FE_Segment::setup_quadrature() {
    auto [points, weights] = GaussLegendre1D::get(degree_ + 1);
    n_qpoints_ = static_cast<int>(points.size());
    qpoints_.resize(n_qpoints_);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        qpoints_[q].coord = Point<3>(points[q], 0, 0);
        qpoints_[q].weight = weights[q];
    }
}

void FE_Segment::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    
    if (degree_ == 1) {
        // Linear Lagrange: N0 = (1-xi)/2, N1 = (1+xi)/2
        values[0] = 0.5 * (1.0 - x);
        values[1] = 0.5 * (1.0 + x);
    } else if (degree_ == 2) {
        // Quadratic Lagrange: nodes at -1, 0, 1
        values[0] = 0.5 * x * (x - 1.0);
        values[1] = 1.0 - x * x;
        values[2] = 0.5 * x * (x + 1.0);
    }
    
    // Handle vector elements (copy shape values for each component)
    if (n_components_ > 1) {
        std::vector<Scalar> scalar_vals(values.begin(), values.begin() + degree_ + 1);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i <= degree_; ++i) {
                values[c * (degree_ + 1) + i] = scalar_vals[i];
            }
        }
    }
}

void FE_Segment::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    
    if (degree_ == 1) {
        // dN0/dxi = -0.5, dN1/dxi = 0.5
        grads[0] = Tensor<1, 3>(-0.5, 0, 0);
        grads[1] = Tensor<1, 3>(0.5, 0, 0);
    } else if (degree_ == 2) {
        grads[0] = Tensor<1, 3>(x - 0.5, 0, 0);
        grads[1] = Tensor<1, 3>(-2.0 * x, 0, 0);
        grads[2] = Tensor<1, 3>(x + 0.5, 0, 0);
    }
    
    // Handle vector elements
    if (n_components_ > 1) {
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + degree_ + 1);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i <= degree_; ++i) {
                grads[c * (degree_ + 1) + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Triangle
// ============================================================

FE_Triangle::FE_Triangle(int degree, int n_components) {
    degree_ = degree;
    dim_ = 2;
    n_components_ = n_components;
    dofs_per_cell_ = ((degree + 1) * (degree + 2) / 2) * n_components;
    geom_type_ = GeometryType::Triangle;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Triangle::setup_quadrature() {
    auto [points, weights] = TriangleQuadrature::get(2 * degree_);
    n_qpoints_ = static_cast<int>(points.size());
    qpoints_.resize(n_qpoints_);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        qpoints_[q].coord = Point<3>(points[q].x(), points[q].y(), 0);
        qpoints_[q].weight = weights[q];
    }
}

void FE_Triangle::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    
    // Barycentric coordinates: lambda0 = 1-x-y, lambda1 = x, lambda2 = y
    const Scalar l0 = 1.0 - x - y;
    const Scalar l1 = x;
    const Scalar l2 = y;
    
    if (degree_ == 1) {
        // Linear Lagrange on triangle
        values[0] = l0;
        values[1] = l1;
        values[2] = l2;
    } else if (degree_ == 2) {
        // Quadratic Lagrange: 6 nodes
        // Vertices: 0,1,2; Edges: 3(0-1), 4(1-2), 5(2-0)
        values[0] = l0 * (2.0 * l0 - 1.0);
        values[1] = l1 * (2.0 * l1 - 1.0);
        values[2] = l2 * (2.0 * l2 - 1.0);
        values[3] = 4.0 * l0 * l1;
        values[4] = 4.0 * l1 * l2;
        values[5] = 4.0 * l2 * l0;
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) / 2;
        std::vector<Scalar> scalar_vals(values.begin(), values.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = scalar_vals[i];
            }
        }
    }
}

void FE_Triangle::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    
    const Scalar l0 = 1.0 - x - y;
    const Scalar l1 = x;
    const Scalar l2 = y;
    
    if (degree_ == 1) {
        // d(l0)/d(xi,eta) = (-1,-1), d(l1) = (1,0), d(l2) = (0,1)
        grads[0] = Tensor<1, 3>(-1.0, -1.0, 0);
        grads[1] = Tensor<1, 3>(1.0, 0, 0);
        grads[2] = Tensor<1, 3>(0, 1.0, 0);
    } else if (degree_ == 2) {
        grads[0] = Tensor<1, 3>(1.0 - 4.0 * l0, 1.0 - 4.0 * l0, 0);
        grads[1] = Tensor<1, 3>(4.0 * l1 - 1.0, 0, 0);
        grads[2] = Tensor<1, 3>(0, 4.0 * l2 - 1.0, 0);
        grads[3] = Tensor<1, 3>(4.0 * (l0 - l1), -4.0 * l1, 0);
        grads[4] = Tensor<1, 3>(4.0 * l2, 4.0 * l1, 0);
        grads[5] = Tensor<1, 3>(-4.0 * l2, 4.0 * (l0 - l2), 0);
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) / 2;
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Quadrilateral
// ============================================================

FE_Quadrilateral::FE_Quadrilateral(int degree, int n_components) {
    degree_ = degree;
    dim_ = 2;
    n_components_ = n_components;
    dofs_per_cell_ = (degree + 1) * (degree + 1) * n_components;
    geom_type_ = GeometryType::Quadrilateral;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Quadrilateral::setup_quadrature() {
    auto [points1, weights1] = GaussLegendre1D::get(degree_ + 1);
    n_qpoints_ = static_cast<int>(points1.size() * points1.size());
    qpoints_.resize(n_qpoints_);
    
    int q = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        for (size_t j = 0; j < points1.size(); ++j) {
            qpoints_[q].coord = Point<3>(points1[i], points1[j], 0);
            qpoints_[q].weight = weights1[i] * weights1[j];
            ++q;
        }
    }
}

void FE_Quadrilateral::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    
    // 1D shape functions
    std::vector<Scalar> Nx(degree_ + 1), Ny(degree_ + 1);
    
    if (degree_ == 1) {
        Nx[0] = 0.5 * (1.0 - x); Ny[0] = 0.5 * (1.0 - y);
        Nx[1] = 0.5 * (1.0 + x); Ny[1] = 0.5 * (1.0 + y);
    } else if (degree_ == 2) {
        Nx[0] = 0.5 * x * (x - 1.0); Ny[0] = 0.5 * y * (y - 1.0);
        Nx[1] = 1.0 - x * x;       Ny[1] = 1.0 - y * y;
        Nx[2] = 0.5 * x * (x + 1.0); Ny[2] = 0.5 * y * (y + 1.0);
    }
    
    // Tensor product
    for (int j = 0; j <= degree_; ++j) {
        for (int i = 0; i <= degree_; ++i) {
            values[j * (degree_ + 1) + i] = Nx[i] * Ny[j];
        }
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 1);
        std::vector<Scalar> scalar_vals(values.begin(), values.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = scalar_vals[i];
            }
        }
    }
}

void FE_Quadrilateral::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    
    std::vector<Scalar> Nx(degree_ + 1), Ny(degree_ + 1);
    std::vector<Scalar> dNx(degree_ + 1), dNy(degree_ + 1);
    
    if (degree_ == 1) {
        Nx[0] = 0.5 * (1.0 - x); Ny[0] = 0.5 * (1.0 - y);
        Nx[1] = 0.5 * (1.0 + x); Ny[1] = 0.5 * (1.0 + y);
        dNx[0] = -0.5; dNy[0] = -0.5;
        dNx[1] = 0.5;  dNy[1] = 0.5;
    } else if (degree_ == 2) {
        Nx[0] = 0.5 * x * (x - 1.0); Ny[0] = 0.5 * y * (y - 1.0);
        Nx[1] = 1.0 - x * x;       Ny[1] = 1.0 - y * y;
        Nx[2] = 0.5 * x * (x + 1.0); Ny[2] = 0.5 * y * (y + 1.0);
        dNx[0] = x - 0.5; dNy[0] = y - 0.5;
        dNx[1] = -2.0 * x; dNy[1] = -2.0 * y;
        dNx[2] = x + 0.5; dNy[2] = y + 0.5;
    }
    
    for (int j = 0; j <= degree_; ++j) {
        for (int i = 0; i <= degree_; ++i) {
            int idx = j * (degree_ + 1) + i;
            grads[idx] = Tensor<1, 3>(dNx[i] * Ny[j], Nx[i] * dNy[j], 0);
        }
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 1);
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Tetrahedron
// ============================================================

FE_Tetrahedron::FE_Tetrahedron(int degree, int n_components) {
    degree_ = degree;
    dim_ = 3;
    n_components_ = n_components;
    dofs_per_cell_ = ((degree + 1) * (degree + 2) * (degree + 3) / 6) * n_components;
    geom_type_ = GeometryType::Tetrahedron;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Tetrahedron::setup_quadrature() {
    auto [points, weights] = TetrahedronQuadrature::get(2 * degree_);
    n_qpoints_ = static_cast<int>(points.size());
    qpoints_.resize(n_qpoints_);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        qpoints_[q].coord = points[q];
        qpoints_[q].weight = weights[q];
    }
}

void FE_Tetrahedron::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    const Scalar z = xi.z();
    
    // Barycentric coordinates
    const Scalar l0 = 1.0 - x - y - z;
    const Scalar l1 = x;
    const Scalar l2 = y;
    const Scalar l3 = z;
    
    if (degree_ == 1) {
        // Linear Lagrange on tetrahedron
        values[0] = l0;
        values[1] = l1;
        values[2] = l2;
        values[3] = l3;
    } else if (degree_ == 2) {
        // Quadratic Lagrange: 10 nodes
        // Vertices: 0,1,2,3; Edges: 4(0-1),5(1-2),6(2-0),7(0-3),8(1-3),9(2-3)
        values[0] = l0 * (2.0 * l0 - 1.0);
        values[1] = l1 * (2.0 * l1 - 1.0);
        values[2] = l2 * (2.0 * l2 - 1.0);
        values[3] = l3 * (2.0 * l3 - 1.0);
        values[4] = 4.0 * l0 * l1;
        values[5] = 4.0 * l1 * l2;
        values[6] = 4.0 * l2 * l0;
        values[7] = 4.0 * l0 * l3;
        values[8] = 4.0 * l1 * l3;
        values[9] = 4.0 * l2 * l3;
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) * (degree_ + 3) / 6;
        std::vector<Scalar> scalar_vals(values.begin(), values.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = scalar_vals[i];
            }
        }
    }
}

void FE_Tetrahedron::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    const Scalar z = xi.z();
    
    const Scalar l0 = 1.0 - x - y - z;
    const Scalar l1 = x;
    const Scalar l2 = y;
    const Scalar l3 = z;
    
    if (degree_ == 1) {
        grads[0] = Tensor<1, 3>(-1.0, -1.0, -1.0);
        grads[1] = Tensor<1, 3>(1.0, 0, 0);
        grads[2] = Tensor<1, 3>(0, 1.0, 0);
        grads[3] = Tensor<1, 3>(0, 0, 1.0);
    } else if (degree_ == 2) {
        grads[0] = Tensor<1, 3>(1.0 - 4.0 * l0, 1.0 - 4.0 * l0, 1.0 - 4.0 * l0);
        grads[1] = Tensor<1, 3>(4.0 * l1 - 1.0, 0, 0);
        grads[2] = Tensor<1, 3>(0, 4.0 * l2 - 1.0, 0);
        grads[3] = Tensor<1, 3>(0, 0, 4.0 * l3 - 1.0);
        grads[4] = Tensor<1, 3>(4.0 * (l0 - l1), -4.0 * l1, -4.0 * l1);
        grads[5] = Tensor<1, 3>(4.0 * l2, 4.0 * l1, 0);
        grads[6] = Tensor<1, 3>(-4.0 * l2, 4.0 * (l0 - l2), -4.0 * l2);
        grads[7] = Tensor<1, 3>(-4.0 * l3, -4.0 * l3, 4.0 * (l0 - l3));
        grads[8] = Tensor<1, 3>(4.0 * l3, 0, 4.0 * l1);
        grads[9] = Tensor<1, 3>(0, 4.0 * l3, 4.0 * l2);
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) * (degree_ + 3) / 6;
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Hexahedron
// ============================================================

FE_Hexahedron::FE_Hexahedron(int degree, int n_components) {
    degree_ = degree;
    dim_ = 3;
    n_components_ = n_components;
    dofs_per_cell_ = (degree + 1) * (degree + 1) * (degree + 1) * n_components;
    geom_type_ = GeometryType::Hexahedron;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Hexahedron::setup_quadrature() {
    auto [points1, weights1] = GaussLegendre1D::get(degree_ + 1);
    n_qpoints_ = static_cast<int>(points1.size() * points1.size() * points1.size());
    qpoints_.resize(n_qpoints_);
    
    int q = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        for (size_t j = 0; j < points1.size(); ++j) {
            for (size_t k = 0; k < points1.size(); ++k) {
                qpoints_[q].coord = Point<3>(points1[i], points1[j], points1[k]);
                qpoints_[q].weight = weights1[i] * weights1[j] * weights1[k];
                ++q;
            }
        }
    }
}

void FE_Hexahedron::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    const Scalar z = xi.z();
    
    std::vector<Scalar> Nx(degree_ + 1), Ny(degree_ + 1), Nz(degree_ + 1);
    
    if (degree_ == 1) {
        Nx[0] = 0.5 * (1.0 - x); Nx[1] = 0.5 * (1.0 + x);
        Ny[0] = 0.5 * (1.0 - y); Ny[1] = 0.5 * (1.0 + y);
        Nz[0] = 0.5 * (1.0 - z); Nz[1] = 0.5 * (1.0 + z);
    } else if (degree_ == 2) {
        Nx[0] = 0.5 * x * (x - 1.0); Nx[1] = 1.0 - x * x; Nx[2] = 0.5 * x * (x + 1.0);
        Ny[0] = 0.5 * y * (y - 1.0); Ny[1] = 1.0 - y * y; Ny[2] = 0.5 * y * (y + 1.0);
        Nz[0] = 0.5 * z * (z - 1.0); Nz[1] = 1.0 - z * z; Nz[2] = 0.5 * z * (z + 1.0);
    }
    
    // Tensor product
    int idx = 0;
    for (int k = 0; k <= degree_; ++k) {
        for (int j = 0; j <= degree_; ++j) {
            for (int i = 0; i <= degree_; ++i) {
                values[idx++] = Nx[i] * Ny[j] * Nz[k];
            }
        }
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 1) * (degree_ + 1);
        std::vector<Scalar> scalar_vals(values.begin(), values.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = scalar_vals[i];
            }
        }
    }
}

void FE_Hexahedron::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    const Scalar z = xi.z();
    
    std::vector<Scalar> Nx(degree_ + 1), Ny(degree_ + 1), Nz(degree_ + 1);
    std::vector<Scalar> dNx(degree_ + 1), dNy(degree_ + 1), dNz(degree_ + 1);
    
    if (degree_ == 1) {
        Nx[0] = 0.5 * (1.0 - x); Nx[1] = 0.5 * (1.0 + x);
        Ny[0] = 0.5 * (1.0 - y); Ny[1] = 0.5 * (1.0 + y);
        Nz[0] = 0.5 * (1.0 - z); Nz[1] = 0.5 * (1.0 + z);
        dNx[0] = -0.5; dNx[1] = 0.5;
        dNy[0] = -0.5; dNy[1] = 0.5;
        dNz[0] = -0.5; dNz[1] = 0.5;
    } else if (degree_ == 2) {
        Nx[0] = 0.5 * x * (x - 1.0); Nx[1] = 1.0 - x * x; Nx[2] = 0.5 * x * (x + 1.0);
        Ny[0] = 0.5 * y * (y - 1.0); Ny[1] = 1.0 - y * y; Ny[2] = 0.5 * y * (y + 1.0);
        Nz[0] = 0.5 * z * (z - 1.0); Nz[1] = 1.0 - z * z; Nz[2] = 0.5 * z * (z + 1.0);
        dNx[0] = x - 0.5; dNx[1] = -2.0 * x; dNx[2] = x + 0.5;
        dNy[0] = y - 0.5; dNy[1] = -2.0 * y; dNy[2] = y + 0.5;
        dNz[0] = z - 0.5; dNz[1] = -2.0 * z; dNz[2] = z + 0.5;
    }
    
    int idx = 0;
    for (int k = 0; k <= degree_; ++k) {
        for (int j = 0; j <= degree_; ++j) {
            for (int i = 0; i <= degree_; ++i) {
                grads[idx++] = Tensor<1, 3>(
                    dNx[i] * Ny[j] * Nz[k],
                    Nx[i] * dNy[j] * Nz[k],
                    Nx[i] * Ny[j] * dNz[k]
                );
            }
        }
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 1) * (degree_ + 1);
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Wedge
// ============================================================

FE_Wedge::FE_Wedge(int degree, int n_components) {
    degree_ = degree;
    dim_ = 3;
    n_components_ = n_components;
    // Wedge = triangle x line
    dofs_per_cell_ = ((degree + 1) * (degree + 2) / 2) * (degree + 1) * n_components;
    geom_type_ = GeometryType::Wedge;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Wedge::setup_quadrature() {
    auto [tri_pts, tri_wts] = TriangleQuadrature::get(2 * degree_);
    auto [line_pts, line_wts] = GaussLegendre1D::get(degree_ + 1);
    
    n_qpoints_ = static_cast<int>(tri_pts.size() * line_pts.size());
    qpoints_.resize(n_qpoints_);
    
    int q = 0;
    for (size_t t = 0; t < tri_pts.size(); ++t) {
        for (size_t l = 0; l < line_pts.size(); ++l) {
            qpoints_[q].coord = Point<3>(tri_pts[t].x(), tri_pts[t].y(), line_pts[l]);
            qpoints_[q].weight = tri_wts[t] * line_wts[l] * 0.5;  // 0.5 for triangle area
            ++q;
        }
    }
}

void FE_Wedge::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    
    // Wedge reference element: triangle (xi, eta) x line (zeta)
    // Triangle: xi >= 0, eta >= 0, xi + eta <= 1
    // Line: zeta in [-1, 1]
    Scalar x = xi.x(), y = xi.y(), z = xi.z();
    Scalar l0 = 1.0 - x - y;  // Triangle barycentric coordinates
    Scalar l1 = x;
    Scalar l2 = y;
    Scalar h0 = 0.5 * (1.0 - z);  // Linear in z
    Scalar h1 = 0.5 * (1.0 + z);
    
    // Linear wedge: 6 nodes
    // Bottom triangle: nodes 0,1,2 at z=-1
    // Top triangle: nodes 3,4,5 at z=+1
    int idx = 0;
    values[idx++] = l0 * h0;  // Node 0: (0,0,-1)
    values[idx++] = l1 * h0;  // Node 1: (1,0,-1)
    values[idx++] = l2 * h0;  // Node 2: (0,1,-1)
    values[idx++] = l0 * h1;  // Node 3: (0,0,+1)
    values[idx++] = l1 * h1;  // Node 4: (1,0,+1)
    values[idx++] = l2 * h1;  // Node 5: (0,1,+1)
    
    // Vector element: repeat for each component
    if (n_components_ > 1) {
        int scalar_dofs = 6;
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = values[i];
            }
        }
    }
}

void FE_Wedge::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    
    Scalar x = xi.x(), y = xi.y(), z = xi.z();
    Scalar l0 = 1.0 - x - y;
    Scalar l1 = x;
    Scalar l2 = y;
    Scalar h0 = 0.5 * (1.0 - z);
    Scalar h1 = 0.5 * (1.0 + z);
    
    // Gradients of barycentric coords: dl0/dx=-1, dl0/dy=-1, dl1/dx=1, dl1/dy=0, dl2/dx=0, dl2/dy=1
    // Gradients of h coords: dh0/dz=-0.5, dh1/dz=0.5
    
    int idx = 0;
    // Node 0: N0 = l0 * h0
    grads[idx++] = Tensor<1, 3>(-h0, -h0, -0.5 * l0);
    // Node 1: N1 = l1 * h0
    grads[idx++] = Tensor<1, 3>(h0, 0.0, -0.5 * l1);
    // Node 2: N2 = l2 * h0
    grads[idx++] = Tensor<1, 3>(0.0, h0, -0.5 * l2);
    // Node 3: N3 = l0 * h1
    grads[idx++] = Tensor<1, 3>(-h1, -h1, 0.5 * l0);
    // Node 4: N4 = l1 * h1
    grads[idx++] = Tensor<1, 3>(h1, 0.0, 0.5 * l1);
    // Node 5: N5 = l2 * h1
    grads[idx++] = Tensor<1, 3>(0.0, h1, 0.5 * l2);
    
    if (n_components_ > 1) {
        int scalar_dofs = 6;
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// FE_Pyramid
// ============================================================

FE_Pyramid::FE_Pyramid(int degree, int n_components) {
    degree_ = degree;
    dim_ = 3;
    n_components_ = n_components;
    dofs_per_cell_ = (degree + 1) * (degree + 2) * (2 * degree + 3) / 6 * n_components;
    geom_type_ = GeometryType::Pyramid;
    quad_order_ = 2 * degree;
    
    initialize_quadrature();
}

void FE_Pyramid::setup_quadrature() {
    auto [pts, wts] = PyramidQuadrature::get(2 * degree_);
    n_qpoints_ = static_cast<int>(pts.size());
    qpoints_.resize(n_qpoints_);
    
    for (size_t q = 0; q < pts.size(); ++q) {
        qpoints_[q].coord = pts[q];
        qpoints_[q].weight = wts[q];
    }
}

void FE_Pyramid::shape_values(const Point<3>& xi, std::vector<Scalar>& values) const {
    values.resize(dofs_per_cell_);
    
    // Pyramid reference element:
    // Base: square on z=0 plane, corners at (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)
    // Apex: (0,0,1)
    Scalar x = xi.x(), y = xi.y(), z = xi.z();
    
    // Handle the apex case (z=1) specially
    if (std::abs(z - 1.0) < 1e-14) {
        values[0] = 0.0; values[1] = 0.0; values[2] = 0.0; values[3] = 0.0;
        values[4] = 1.0;
    } else {
        // Using virtual node formulation:
        // N_i = (1/4) * (1 + xi_i*x) * (1 + eta_i*y) * (1-z) for base nodes
        // N_4 = z for apex
        Scalar omz = 1.0 - z;
        
        values[0] = 0.25 * (1.0 - x) * (1.0 - y) * omz;  // (-1,-1)
        values[1] = 0.25 * (1.0 + x) * (1.0 - y) * omz;  // (1,-1)
        values[2] = 0.25 * (1.0 + x) * (1.0 + y) * omz;  // (1,1)
        values[3] = 0.25 * (1.0 - x) * (1.0 + y) * omz;  // (-1,1)
        values[4] = z;                                     // Apex
    }
    
    // Vector element
    if (n_components_ > 1) {
        int scalar_dofs = 5;
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                values[c * scalar_dofs + i] = values[i];
            }
        }
    }
}

void FE_Pyramid::shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(dofs_per_cell_);
    
    Scalar x = xi.x(), y = xi.y(), z = xi.z();
    
    // Handle the apex case specially
    if (std::abs(z - 1.0) < 1e-14) {
        // At apex: gradients are defined by limit
        grads[0] = Tensor<1, 3>(-0.25, -0.25, 0.5);
        grads[1] = Tensor<1, 3>(0.25, -0.25, 0.5);
        grads[2] = Tensor<1, 3>(0.25, 0.25, 0.5);
        grads[3] = Tensor<1, 3>(-0.25, 0.25, 0.5);
        grads[4] = Tensor<1, 3>(0.0, 0.0, 1.0);
    } else {
        // Pyramid shape functions for base nodes (using virtual node formulation):
        // N_i = (1/4) * (1 + xi_i*x) * (1 + eta_i*y) * (1-z)
        // where (xi_i, eta_i) are the corner coordinates: (-1,-1), (1,-1), (1,1), (-1,1)
        // N_4 = z (apex)
        
        Scalar omz = 1.0 - z;
        
        // Node 0: (-1, -1) -> N_0 = (1/4) * (1-x) * (1-y) * (1-z)
        grads[0] = Tensor<1, 3>(-0.25 * (1-y) * omz, -0.25 * (1-x) * omz, -0.25 * (1-x) * (1-y));
        
        // Node 1: (1, -1) -> N_1 = (1/4) * (1+x) * (1-y) * (1-z)
        grads[1] = Tensor<1, 3>(0.25 * (1-y) * omz, -0.25 * (1+x) * omz, -0.25 * (1+x) * (1-y));
        
        // Node 2: (1, 1) -> N_2 = (1/4) * (1+x) * (1+y) * (1-z)
        grads[2] = Tensor<1, 3>(0.25 * (1+y) * omz, 0.25 * (1+x) * omz, -0.25 * (1+x) * (1+y));
        
        // Node 3: (-1, 1) -> N_3 = (1/4) * (1-x) * (1+y) * (1-z)
        grads[3] = Tensor<1, 3>(-0.25 * (1+y) * omz, 0.25 * (1-x) * omz, -0.25 * (1-x) * (1+y));
        
        // Node 4 (apex): N_4 = z
        grads[4] = Tensor<1, 3>(0.0, 0.0, 1.0);
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = 5;
        std::vector<Tensor<1, 3>> scalar_grads(grads.begin(), grads.begin() + scalar_dofs);
        for (int c = 1; c < n_components_; ++c) {
            for (int i = 0; i < scalar_dofs; ++i) {
                grads[c * scalar_dofs + i] = scalar_grads[i];
            }
        }
    }
}

// ============================================================
// Factory function
// ============================================================

std::unique_ptr<FiniteElement> create_fe(GeometryType type, int degree, int n_components) {
    switch (type) {
        case GeometryType::Segment:
            return std::make_unique<FE_Segment>(degree, n_components);
        case GeometryType::Triangle:
            return std::make_unique<FE_Triangle>(degree, n_components);
        case GeometryType::Quadrilateral:
            return std::make_unique<FE_Quadrilateral>(degree, n_components);
        case GeometryType::Tetrahedron:
            return std::make_unique<FE_Tetrahedron>(degree, n_components);
        case GeometryType::Hexahedron:
            return std::make_unique<FE_Hexahedron>(degree, n_components);
        case GeometryType::Wedge:
            return std::make_unique<FE_Wedge>(degree, n_components);
        case GeometryType::Pyramid:
            return std::make_unique<FE_Pyramid>(degree, n_components);
        default:
            MPFEM_THROW(NotImplementedError, "Unsupported element type");
    }
}

}  // namespace mpfem