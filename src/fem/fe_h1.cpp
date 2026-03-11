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
    // Linear: 3 * 2 = 6 nodes
    // Quadratic (Serendipity): 6 corners + 9 edges = 15 nodes
    // Formula: ((degree+1)*(degree+2)/2) * (degree+1) gives:
    //   degree=1: 3 * 2 = 6
    //   degree=2: 6 * 3 = 18 (full Lagrange, includes face nodes)
    // For Serendipity, degree=2 has 15 nodes
    if (degree == 1) {
        dofs_per_cell_ = 6 * n_components;
    } else if (degree == 2) {
        dofs_per_cell_ = 15 * n_components;  // Serendipity: 6 corners + 9 edges
    } else {
        dofs_per_cell_ = ((degree + 1) * (degree + 2) / 2) * (degree + 1) * n_components;
    }
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
    
    if (degree_ == 1) {
        Scalar h0 = 0.5 * (1.0 - z);  // Linear in z
        Scalar h1 = 0.5 * (1.0 + z);
        
        // Linear wedge: 6 nodes
        // Bottom triangle: nodes 0,1,2 at z=-1
        // Top triangle: nodes 3,4,5 at z=+1
        values[0] = l0 * h0;  // Node 0: (0,0,-1)
        values[1] = l1 * h0;  // Node 1: (1,0,-1)
        values[2] = l2 * h0;  // Node 2: (0,1,-1)
        values[3] = l0 * h1;  // Node 3: (0,0,+1)
        values[4] = l1 * h1;  // Node 4: (1,0,+1)
        values[5] = l2 * h1;  // Node 5: (0,1,+1)
    } else if (degree_ == 2) {
        // Quadratic wedge: 15 nodes (Serendipity)
        // Corner nodes (6): bottom(0,1,2), top(3,4,5)
        // Edge nodes (9): bottom edges(6,7,8), top edges(9,10,11), vertical edges(12,13,14)
        
        // Triangle shape functions (quadratic): use hierarchical approach
        // Vertex functions: phi_i = l_i * (2*l_i - 1)
        // Edge functions: phi_ij = 4 * l_i * l_j
        
        // Line shape functions in z
        Scalar h0 = 0.5 * z * (z - 1.0);  // z = -1
        Scalar h1 = 1.0 - z * z;           // z = 0 (middle)
        Scalar h2 = 0.5 * z * (z + 1.0);  // z = +1
        
        // Corner nodes (0-5): bottom at z=-1, top at z=+1
        // Using tensor product: triangle_vertex * line_vertex
        // At z=-1: h0=1, h1=0, h2=0
        // At z=+1: h0=0, h1=0, h2=1
        
        // Bottom triangle vertices (z=-1): h0 contributes
        values[0] = l0 * (2.0 * l0 - 1.0) * h0;  // Node 0: corner (0,0,-1)
        values[1] = l1 * (2.0 * l1 - 1.0) * h0;  // Node 1: corner (1,0,-1)
        values[2] = l2 * (2.0 * l2 - 1.0) * h0;  // Node 2: corner (0,1,-1)
        
        // Top triangle vertices (z=+1): h2 contributes
        values[3] = l0 * (2.0 * l0 - 1.0) * h2;  // Node 3: corner (0,0,+1)
        values[4] = l1 * (2.0 * l1 - 1.0) * h2;  // Node 4: corner (1,0,+1)
        values[5] = l2 * (2.0 * l2 - 1.0) * h2;  // Node 5: corner (0,1,+1)
        
        // Bottom triangle edge nodes (z=-1): edges at z=-1
        values[6] = 4.0 * l0 * l1 * h0;  // Node 6: edge (0-1) midpoint at z=-1
        values[7] = 4.0 * l1 * l2 * h0;  // Node 7: edge (1-2) midpoint at z=-1
        values[8] = 4.0 * l2 * l0 * h0;  // Node 8: edge (2-0) midpoint at z=-1
        
        // Top triangle edge nodes (z=+1): edges at z=+1
        values[9]  = 4.0 * l0 * l1 * h2;  // Node 9: edge (3-4) midpoint at z=+1
        values[10] = 4.0 * l1 * l2 * h2;  // Node 10: edge (4-5) midpoint at z=+1
        values[11] = 4.0 * l2 * l0 * h2;  // Node 11: edge (5-3) midpoint at z=+1
        
        // Vertical edge nodes (z=0): connecting bottom to top
        // These are at the midpoints of vertical edges, z=0
        // Triangle vertex functions * h1 (z=0)
        values[12] = l0 * (2.0 * l0 - 1.0) * h1;  // Node 12: edge (0-3) midpoint
        values[13] = l1 * (2.0 * l1 - 1.0) * h1;  // Node 13: edge (1-4) midpoint
        values[14] = l2 * (2.0 * l2 - 1.0) * h1;  // Node 14: edge (2-5) midpoint
    }
    
    // Vector element: repeat for each component
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) / 2 * (degree_ + 1);
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
    
    if (degree_ == 1) {
        Scalar h0 = 0.5 * (1.0 - z);
        Scalar h1 = 0.5 * (1.0 + z);
        
        // Gradients of barycentric coords: dl0/dx=-1, dl0/dy=-1, dl1/dx=1, dl1/dy=0, dl2/dx=0, dl2/dy=1
        // Gradients of h coords: dh0/dz=-0.5, dh1/dz=0.5
        
        // Node 0: N0 = l0 * h0
        grads[0] = Tensor<1, 3>(-h0, -h0, -0.5 * l0);
        // Node 1: N1 = l1 * h0
        grads[1] = Tensor<1, 3>(h0, 0.0, -0.5 * l1);
        // Node 2: N2 = l2 * h0
        grads[2] = Tensor<1, 3>(0.0, h0, -0.5 * l2);
        // Node 3: N3 = l0 * h1
        grads[3] = Tensor<1, 3>(-h1, -h1, 0.5 * l0);
        // Node 4: N4 = l1 * h1
        grads[4] = Tensor<1, 3>(h1, 0.0, 0.5 * l1);
        // Node 5: N5 = l2 * h1
        grads[5] = Tensor<1, 3>(0.0, h1, 0.5 * l2);
    } else if (degree_ == 2) {
        // Quadratic wedge gradients (15 nodes)
        // Derivatives of triangle barycentric coordinates
        // dl0/dx = -1, dl0/dy = -1
        // dl1/dx = 1, dl1/dy = 0
        // dl2/dx = 0, dl2/dy = 1
        
        // Line shape functions in z
        Scalar h0 = 0.5 * z * (z - 1.0);
        Scalar h1 = 1.0 - z * z;
        Scalar h2 = 0.5 * z * (z + 1.0);
        
        // Derivatives of line shape functions
        Scalar dh0 = z - 0.5;
        Scalar dh1 = -2.0 * z;
        Scalar dh2 = z + 0.5;
        
        // Triangle vertex functions: phi_i = l_i * (2*l_i - 1)
        // d(phi_i)/dx = (4*l_i - 1) * dl_i/dx
        // d(phi_0)/dx = (4*l0 - 1) * (-1) = 1 - 4*l0
        // d(phi_0)/dy = (4*l0 - 1) * (-1) = 1 - 4*l0
        // d(phi_1)/dx = (4*l1 - 1) * 1 = 4*l1 - 1
        // d(phi_1)/dy = 0
        // d(phi_2)/dx = 0
        // d(phi_2)/dy = (4*l2 - 1) * 1 = 4*l2 - 1
        
        // Triangle edge functions: phi_ij = 4 * l_i * l_j
        // d(phi_01)/dx = 4 * (dl0/dx * l1 + l0 * dl1/dx) = 4 * (-l1 + l0) = 4*(l0 - l1)
        // d(phi_01)/dy = 4 * (dl0/dy * l1 + l0 * dl1/dy) = 4 * (-l1 + 0) = -4*l1
        
        // Corner nodes (0-5)
        // Node 0: l0*(2*l0-1)*h0
        grads[0] = Tensor<1, 3>((1.0 - 4.0*l0) * h0, (1.0 - 4.0*l0) * h0, l0*(2.0*l0-1.0)*dh0);
        // Node 1: l1*(2*l1-1)*h0
        grads[1] = Tensor<1, 3>((4.0*l1 - 1.0) * h0, 0.0, l1*(2.0*l1-1.0)*dh0);
        // Node 2: l2*(2*l2-1)*h0
        grads[2] = Tensor<1, 3>(0.0, (4.0*l2 - 1.0) * h0, l2*(2.0*l2-1.0)*dh0);
        // Node 3: l0*(2*l0-1)*h2
        grads[3] = Tensor<1, 3>((1.0 - 4.0*l0) * h2, (1.0 - 4.0*l0) * h2, l0*(2.0*l0-1.0)*dh2);
        // Node 4: l1*(2*l1-1)*h2
        grads[4] = Tensor<1, 3>((4.0*l1 - 1.0) * h2, 0.0, l1*(2.0*l1-1.0)*dh2);
        // Node 5: l2*(2*l2-1)*h2
        grads[5] = Tensor<1, 3>(0.0, (4.0*l2 - 1.0) * h2, l2*(2.0*l2-1.0)*dh2);
        
        // Bottom edge nodes (6-8)
        // Node 6: 4*l0*l1*h0
        grads[6] = Tensor<1, 3>(4.0*(l0 - l1) * h0, -4.0*l1 * h0, 4.0*l0*l1*dh0);
        // Node 7: 4*l1*l2*h0
        grads[7] = Tensor<1, 3>(4.0*l2 * h0, 4.0*l1 * h0, 4.0*l1*l2*dh0);
        // Node 8: 4*l2*l0*h0
        grads[8] = Tensor<1, 3>(-4.0*l2 * h0, 4.0*(l0 - l2) * h0, 4.0*l2*l0*dh0);
        
        // Top edge nodes (9-11)
        // Node 9: 4*l0*l1*h2
        grads[9] = Tensor<1, 3>(4.0*(l0 - l1) * h2, -4.0*l1 * h2, 4.0*l0*l1*dh2);
        // Node 10: 4*l1*l2*h2
        grads[10] = Tensor<1, 3>(4.0*l2 * h2, 4.0*l1 * h2, 4.0*l1*l2*dh2);
        // Node 11: 4*l2*l0*h2
        grads[11] = Tensor<1, 3>(-4.0*l2 * h2, 4.0*(l0 - l2) * h2, 4.0*l2*l0*dh2);
        
        // Vertical edge nodes (12-14)
        // Node 12: l0*(2*l0-1)*h1
        grads[12] = Tensor<1, 3>((1.0 - 4.0*l0) * h1, (1.0 - 4.0*l0) * h1, l0*(2.0*l0-1.0)*dh1);
        // Node 13: l1*(2*l1-1)*h1
        grads[13] = Tensor<1, 3>((4.0*l1 - 1.0) * h1, 0.0, l1*(2.0*l1-1.0)*dh1);
        // Node 14: l2*(2*l2-1)*h1
        grads[14] = Tensor<1, 3>(0.0, (4.0*l2 - 1.0) * h1, l2*(2.0*l2-1.0)*dh1);
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ + 1) * (degree_ + 2) / 2 * (degree_ + 1);
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
    // Pyramid nodes:
    // Linear: 5 nodes (4 base corners + 1 apex)
    // Quadratic (Serendipity): 13 nodes (5 corners + 8 edge midpoints)
    // Full Lagrange: (degree+1)*(degree+2)*(2*degree+3)/6
    //   degree=1: 5
    //   degree=2: 14 (includes internal node)
    // For Serendipity, degree=2 has 13 nodes (no internal node)
    if (degree == 1) {
        dofs_per_cell_ = 5 * n_components;
    } else if (degree == 2) {
        dofs_per_cell_ = 13 * n_components;  // Serendipity: 5 corners + 8 edges
    } else {
        dofs_per_cell_ = (degree + 1) * (degree + 2) * (2 * degree + 3) / 6 * n_components;
    }
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
    
    if (degree_ == 1) {
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
    } else if (degree_ == 2) {
        // Quadratic pyramid: 13 nodes (Serendipity)
        // Corner nodes (5): base(0,1,2,3) + apex(4)
        // Edge nodes (8): base edges(5,6,7,8) + vertical edges(9,10,11,12)
        
        // Handle the apex case (z=1) specially
        if (std::abs(z - 1.0) < 1e-14) {
            // At apex, all base functions vanish
            for (int i = 0; i < 12; ++i) values[i] = 0.0;
            values[4] = 1.0;  // Apex vertex
        } else {
            Scalar omz = 1.0 - z;  // 1 - z
            Scalar r = 1.0 / omz;  // 1 / (1-z), defined for z < 1
            
            // Normalized coordinates for pyramid
            Scalar xi_hat = x * r;   // x / (1-z)
            Scalar eta_hat = y * r;  // y / (1-z)
            
            // Base corner nodes (0-3): quadratic vertex functions on square
            // Using tensor product of 1D quadratic Lagrange on [-1, 1]:
            // L_-1(t) = t*(t-1)/2, L_0(t) = 1-t^2, L_+1(t) = t*(t+1)/2
            
            // For pyramid, we multiply by (1-z) to collapse to apex
            // Base nodes at z=0: N_i = L(xi_hat) * L(eta_hat) * (1-z)
            
            // 1D quadratic functions on [-1, 1]
            Scalar Lxm = xi_hat * (xi_hat - 1.0) * 0.5;   // L_-1(xi)
            Scalar Lx0 = 1.0 - xi_hat * xi_hat;            // L_0(xi)
            Scalar Lxp = xi_hat * (xi_hat + 1.0) * 0.5;   // L_+1(xi)
            
            Scalar Lym = eta_hat * (eta_hat - 1.0) * 0.5; // L_-1(eta)
            Scalar Ly0 = 1.0 - eta_hat * eta_hat;          // L_0(eta)
            Scalar Lyp = eta_hat * (eta_hat + 1.0) * 0.5; // L_+1(eta)
            
            // Base corner nodes (vertices of base square at z=0)
            values[0] = Lxm * Lym * omz;  // Node 0: (-1,-1,0)
            values[1] = Lxp * Lym * omz;  // Node 1: (1,-1,0)
            values[2] = Lxp * Lyp * omz;  // Node 2: (1,1,0)
            values[3] = Lxm * Lyp * omz;  // Node 3: (-1,1,0)
            
            // Apex node (4)
            values[4] = z;
            
            // Base edge nodes (5-8): midpoints of base edges
            // These are at xi_hat=0 or eta_hat=0
            values[5] = Lx0 * Lym * omz;  // Node 5: edge (0-1) midpoint at (0,-1,0)
            values[6] = Lxp * Ly0 * omz;  // Node 6: edge (1-2) midpoint at (1,0,0)
            values[7] = Lx0 * Lyp * omz;  // Node 7: edge (2-3) midpoint at (0,1,0)
            values[8] = Lxm * Ly0 * omz;  // Node 8: edge (3-0) midpoint at (-1,0,0)
            
            // Vertical edge nodes (9-12): midpoints of edges from base to apex
            // At z=0.5, the normalized coordinates are 2x and 2y
            // These nodes connect base corners to apex
            // Use linear interpolation in z: at z=0.5, weight = 0.5
            // The edge functions should be: phi_i * (1-z) + correction for apex
            
            // For edge from corner i to apex:
            // Midpoint at z=0.5, coordinates are (corner_x/2, corner_y/2, 0.5)
            // Shape function: L_i(xi_hat, eta_hat) * (1-2z) * (1-z) when z < 0.5
            //                  and similar for z > 0.5
            
            // Simpler formulation using collapsed coordinates:
            // Vertical edge functions: 4*z*(1-z) at the base vertex position
            // When z=0: function = 0
            // When z=0.5: function = 1 at the edge midpoint
            // When z=1: function = 0
            
            Scalar edge_z_factor = 4.0 * z * omz;  // = 4*z*(1-z)
            
            // These are the linear vertex functions times 4*z*(1-z)
            // Linear vertex: (1/4) * (1 +/- x) * (1 +/- y) * (1-z)
            // For edge midpoint: we need 4*z*(1-z) * linear_vertex / (1-z)
            //                  = 4*z * linear_vertex_without_z_factor
            
            values[9]  = edge_z_factor * 0.25 * (1.0 - x*r) * (1.0 - y*r);  // Edge (0-4): (-1,-1) to apex
            values[10] = edge_z_factor * 0.25 * (1.0 + x*r) * (1.0 - y*r);  // Edge (1-4): (1,-1) to apex
            values[11] = edge_z_factor * 0.25 * (1.0 + x*r) * (1.0 + y*r);  // Edge (2-4): (1,1) to apex
            values[12] = edge_z_factor * 0.25 * (1.0 - x*r) * (1.0 + y*r);  // Edge (3-4): (-1,1) to apex
        }
    }
    
    // Vector element
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ == 1) ? 5 : 13;
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
    
    if (degree_ == 1) {
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
    } else if (degree_ == 2) {
        // Quadratic pyramid gradients (13 nodes)
        if (std::abs(z - 1.0) < 1e-14) {
            // At apex
            for (int i = 0; i < 12; ++i) grads[i] = Tensor<1, 3>(0, 0, 0);
            grads[4] = Tensor<1, 3>(0.0, 0.0, 1.0);  // Apex
        } else {
            Scalar omz = 1.0 - z;
            Scalar r = 1.0 / omz;
            Scalar r2 = r * r;
            
            Scalar xi_hat = x * r;
            Scalar eta_hat = y * r;
            
            // 1D quadratic functions and derivatives
            Scalar Lxm = xi_hat * (xi_hat - 1.0) * 0.5;
            Scalar Lx0 = 1.0 - xi_hat * xi_hat;
            Scalar Lxp = xi_hat * (xi_hat + 1.0) * 0.5;
            
            Scalar Lym = eta_hat * (eta_hat - 1.0) * 0.5;
            Scalar Ly0 = 1.0 - eta_hat * eta_hat;
            Scalar Lyp = eta_hat * (eta_hat + 1.0) * 0.5;
            
            // Derivatives of 1D functions w.r.t. xi_hat
            Scalar dLxm = xi_hat - 0.5;
            Scalar dLx0 = -2.0 * xi_hat;
            Scalar dLxp = xi_hat + 0.5;
            
            // Derivatives of 1D functions w.r.t. eta_hat
            Scalar dLym = eta_hat - 0.5;
            Scalar dLy0 = -2.0 * eta_hat;
            Scalar dLyp = eta_hat + 0.5;
            
            // Chain rule: d/dx = d/d(xi_hat) * d(xi_hat)/dx = d/d(xi_hat) * r
            //             d/dz includes terms from d(xi_hat)/dz = -x * r^2 and d(eta_hat)/dz = -y * r^2
            
            // Base corner nodes (0-3)
            // N_0 = Lxm * Lym * omz
            Scalar dN0_dx = (dLxm * Lym) * r * omz;
            Scalar dN0_dy = (Lxm * dLym) * r * omz;
            Scalar dN0_dz = Lxm * Lym + (dLxm * Lym * (-x * r2) + Lxm * dLym * (-y * r2)) * omz;
            grads[0] = Tensor<1, 3>(dN0_dx, dN0_dy, -Lxm * Lym + dN0_dz);
            
            // N_1 = Lxp * Lym * omz
            Scalar dN1_dx = (dLxp * Lym) * r * omz;
            Scalar dN1_dy = (Lxp * dLym) * r * omz;
            Scalar dN1_dz = Lxp * Lym + (dLxp * Lym * (-x * r2) + Lxp * dLym * (-y * r2)) * omz;
            grads[1] = Tensor<1, 3>(dN1_dx, dN1_dy, -Lxp * Lym + dN1_dz);
            
            // N_2 = Lxp * Lyp * omz
            Scalar dN2_dx = (dLxp * Lyp) * r * omz;
            Scalar dN2_dy = (Lxp * dLyp) * r * omz;
            Scalar dN2_dz = Lxp * Lyp + (dLxp * Lyp * (-x * r2) + Lxp * dLyp * (-y * r2)) * omz;
            grads[2] = Tensor<1, 3>(dN2_dx, dN2_dy, -Lxp * Lyp + dN2_dz);
            
            // N_3 = Lxm * Lyp * omz
            Scalar dN3_dx = (dLxm * Lyp) * r * omz;
            Scalar dN3_dy = (Lxm * dLyp) * r * omz;
            Scalar dN3_dz = Lxm * Lyp + (dLxm * Lyp * (-x * r2) + Lxm * dLyp * (-y * r2)) * omz;
            grads[3] = Tensor<1, 3>(dN3_dx, dN3_dy, -Lxm * Lyp + dN3_dz);
            
            // Node 4 (apex): N_4 = z
            grads[4] = Tensor<1, 3>(0.0, 0.0, 1.0);
            
            // Base edge nodes (5-8)
            // N_5 = Lx0 * Lym * omz
            Scalar dN5_dx = (dLx0 * Lym) * r * omz;
            Scalar dN5_dy = (Lx0 * dLym) * r * omz;
            Scalar dN5_dz = Lx0 * Lym + (dLx0 * Lym * (-x * r2) + Lx0 * dLym * (-y * r2)) * omz;
            grads[5] = Tensor<1, 3>(dN5_dx, dN5_dy, -Lx0 * Lym + dN5_dz);
            
            // N_6 = Lxp * Ly0 * omz
            Scalar dN6_dx = (dLxp * Ly0) * r * omz;
            Scalar dN6_dy = (Lxp * dLy0) * r * omz;
            Scalar dN6_dz = Lxp * Ly0 + (dLxp * Ly0 * (-x * r2) + Lxp * dLy0 * (-y * r2)) * omz;
            grads[6] = Tensor<1, 3>(dN6_dx, dN6_dy, -Lxp * Ly0 + dN6_dz);
            
            // N_7 = Lx0 * Lyp * omz
            Scalar dN7_dx = (dLx0 * Lyp) * r * omz;
            Scalar dN7_dy = (Lx0 * dLyp) * r * omz;
            Scalar dN7_dz = Lx0 * Lyp + (dLx0 * Lyp * (-x * r2) + Lx0 * dLyp * (-y * r2)) * omz;
            grads[7] = Tensor<1, 3>(dN7_dx, dN7_dy, -Lx0 * Lyp + dN7_dz);
            
            // N_8 = Lxm * Ly0 * omz
            Scalar dN8_dx = (dLxm * Ly0) * r * omz;
            Scalar dN8_dy = (Lxm * dLy0) * r * omz;
            Scalar dN8_dz = Lxm * Ly0 + (dLxm * Ly0 * (-x * r2) + Lxm * dLy0 * (-y * r2)) * omz;
            grads[8] = Tensor<1, 3>(dN8_dx, dN8_dy, -Lxm * Ly0 + dN8_dz);
            
            // Vertical edge nodes (9-12)
            Scalar edge_z_factor = 4.0 * z * omz;  // 4*z*(1-z)
            Scalar d_edge_z_dz = 4.0 * (omz - z);   // 4*(1-z) - 4*z = 4*(1-2z)
            
            // Linear vertex functions without z-factor
            Scalar v0 = 0.25 * (1.0 - xi_hat) * (1.0 - eta_hat);
            Scalar v1 = 0.25 * (1.0 + xi_hat) * (1.0 - eta_hat);
            Scalar v2 = 0.25 * (1.0 + xi_hat) * (1.0 + eta_hat);
            Scalar v3 = 0.25 * (1.0 - xi_hat) * (1.0 + eta_hat);
            
            // Derivatives of linear vertex functions
            // dv0/dxi_hat = -0.25 * (1 - eta_hat), dv0/deta_hat = -0.25 * (1 - xi_hat)
            Scalar dv0_dx = -0.25 * (1.0 - eta_hat) * r;
            Scalar dv0_dy = -0.25 * (1.0 - xi_hat) * r;
            Scalar dv1_dx = 0.25 * (1.0 - eta_hat) * r;
            Scalar dv1_dy = -0.25 * (1.0 + xi_hat) * r;
            Scalar dv2_dx = 0.25 * (1.0 + eta_hat) * r;
            Scalar dv2_dy = 0.25 * (1.0 + xi_hat) * r;
            Scalar dv3_dx = -0.25 * (1.0 + eta_hat) * r;
            Scalar dv3_dy = 0.25 * (1.0 - xi_hat) * r;
            
            // Additional terms from chain rule for z-derivative
            Scalar dv0_dz_extra = (-0.25 * (1.0 - eta_hat) * (-x * r2) + -0.25 * (1.0 - xi_hat) * (-y * r2));
            Scalar dv1_dz_extra = (0.25 * (1.0 - eta_hat) * (-x * r2) + -0.25 * (1.0 + xi_hat) * (-y * r2));
            Scalar dv2_dz_extra = (0.25 * (1.0 + eta_hat) * (-x * r2) + 0.25 * (1.0 + xi_hat) * (-y * r2));
            Scalar dv3_dz_extra = (-0.25 * (1.0 + eta_hat) * (-x * r2) + 0.25 * (1.0 - xi_hat) * (-y * r2));
            
            // N_9 = edge_z_factor * v0
            grads[9] = Tensor<1, 3>(edge_z_factor * dv0_dx,
                                     edge_z_factor * dv0_dy,
                                     d_edge_z_dz * v0 + edge_z_factor * dv0_dz_extra);
            
            // N_10 = edge_z_factor * v1
            grads[10] = Tensor<1, 3>(edge_z_factor * dv1_dx,
                                      edge_z_factor * dv1_dy,
                                      d_edge_z_dz * v1 + edge_z_factor * dv1_dz_extra);
            
            // N_11 = edge_z_factor * v2
            grads[11] = Tensor<1, 3>(edge_z_factor * dv2_dx,
                                      edge_z_factor * dv2_dy,
                                      d_edge_z_dz * v2 + edge_z_factor * dv2_dz_extra);
            
            // N_12 = edge_z_factor * v3
            grads[12] = Tensor<1, 3>(edge_z_factor * dv3_dx,
                                      edge_z_factor * dv3_dy,
                                      d_edge_z_dz * v3 + edge_z_factor * dv3_dz_extra);
        }
    }
    
    if (n_components_ > 1) {
        int scalar_dofs = (degree_ == 1) ? 5 : 13;
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