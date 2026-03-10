/**
 * @file fe_h1.hpp
 * @brief H1 conforming Lagrange finite elements
 */

#ifndef MPFEM_FEM_FE_H1_HPP
#define MPFEM_FEM_FE_H1_HPP

#include "fe_base.hpp"
#include <array>

namespace mpfem {

/**
 * @brief H1 Lagrange element on line segment [-1, 1]
 */
class FE_Segment : public FiniteElement {
public:
    explicit FE_Segment(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference triangle (0,0), (1,0), (0,1)
 */
class FE_Triangle : public FiniteElement {
public:
    explicit FE_Triangle(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference quadrilateral [-1,1] x [-1,1]
 */
class FE_Quadrilateral : public FiniteElement {
public:
    explicit FE_Quadrilateral(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference tetrahedron
 * Vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 */
class FE_Tetrahedron : public FiniteElement {
public:
    explicit FE_Tetrahedron(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference hexahedron [-1,1]^3
 */
class FE_Hexahedron : public FiniteElement {
public:
    explicit FE_Hexahedron(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference wedge/prism
 * Triangle at z=-1, z=1 with linear interpolation in z
 */
class FE_Wedge : public FiniteElement {
public:
    explicit FE_Wedge(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

/**
 * @brief H1 Lagrange element on reference pyramid
 * Square base at z=0, apex at (0,0,1)
 */
class FE_Pyramid : public FiniteElement {
public:
    explicit FE_Pyramid(int degree, int n_components = 1);
    
    void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const override;
    void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const override;
    
private:
    void setup_quadrature() override;
};

}  // namespace mpfem

#endif  // MPFEM_FEM_FE_H1_HPP