/**
 * @file fe_base.hpp
 * @brief Base class for finite elements
 */

#ifndef MPFEM_FEM_FE_BASE_HPP
#define MPFEM_FEM_FE_BASE_HPP

#include "core/types.hpp"
#include "mesh/element.hpp"
#include "quadrature.hpp"
#include <vector>
#include <memory>

namespace mpfem {

/**
 * @brief Integration point with coordinates and weight
 */
struct IntegrationPoint {
    Point<3> coord;      ///< Coordinates in reference element
    Scalar weight;       ///< Quadrature weight
    
    IntegrationPoint() = default;
    IntegrationPoint(Scalar x, Scalar y, Scalar z, Scalar w) 
        : coord(x, y, z), weight(w) {}
};

/**
 * @brief Abstract base class for finite elements
 * 
 * Defines the interface for evaluating shape functions and their derivatives
 * on reference elements.
 */
class FiniteElement {
public:
    virtual ~FiniteElement() = default;
    
    /// Get the polynomial degree
    int degree() const { return degree_; }
    
    /// Get the spatial dimension of the reference element
    int dim() const { return dim_; }
    
    /// Get the number of components (1 for scalar, >1 for vector)
    int n_components() const { return n_components_; }
    
    /// Get the number of degrees of freedom per cell
    int dofs_per_cell() const { return dofs_per_cell_; }
    
    /// Get the geometry type
    GeometryType geometry_type() const { return geom_type_; }
    
    /// Get number of quadrature points
    int n_quadrature_points() const { return n_qpoints_; }
    
    /// Get the quadrature order
    int quadrature_order() const { return quad_order_; }
    
    /**
     * @brief Evaluate shape function values at reference coordinate
     * @param xi Reference coordinate
     * @param values Output vector of shape function values (size = dofs_per_cell)
     */
    virtual void shape_values(const Point<3>& xi, std::vector<Scalar>& values) const = 0;
    
    /**
     * @brief Evaluate shape function gradients at reference coordinate
     * @param xi Reference coordinate
     * @param grads Output vector of gradients (size = dofs_per_cell)
     */
    virtual void shape_gradients(const Point<3>& xi, std::vector<Tensor<1, 3>>& grads) const = 0;
    
    /**
     * @brief Evaluate shape function values at all quadrature points
     * @param values Output matrix [n_qpoints x dofs_per_cell]
     */
    virtual void shape_values_at_qpoints(std::vector<std::vector<Scalar>>& values) const {
        values.resize(n_qpoints_);
        for (int q = 0; q < n_qpoints_; ++q) {
            shape_values(qpoints_[q].coord, values[q]);
        }
    }
    
    /**
     * @brief Evaluate shape function gradients at all quadrature points
     * @param grads Output matrix [n_qpoints x dofs_per_cell]
     */
    virtual void shape_gradients_at_qpoints(std::vector<std::vector<Tensor<1, 3>>>& grads) const {
        grads.resize(n_qpoints_);
        for (int q = 0; q < n_qpoints_; ++q) {
            shape_gradients(qpoints_[q].coord, grads[q]);
        }
    }
    
    /// Get quadrature point
    const IntegrationPoint& quadrature_point(int q) const { return qpoints_[q]; }
    
    /// Get all quadrature points
    const std::vector<IntegrationPoint>& quadrature_points() const { return qpoints_; }
    
    /// Get shape value at quadrature point (precomputed)
    Scalar shape_value(int i, int q) const { 
        return shape_values_[q * dofs_per_cell_ + i]; 
    }
    
    /// Get shape gradient at quadrature point (precomputed)
    const Tensor<1, 3>& shape_gradient(int i, int q) const { 
        return shape_grads_[q * dofs_per_cell_ + i]; 
    }
    
protected:
    int degree_ = 1;                    ///< Polynomial degree
    int dim_ = 0;                       ///< Spatial dimension
    int n_components_ = 1;              ///< Number of components
    int dofs_per_cell_ = 0;             ///< DoFs per cell
    int n_qpoints_ = 0;                 ///< Number of quadrature points
    int quad_order_ = 1;                ///< Quadrature order
    GeometryType geom_type_ = GeometryType::Invalid;
    
    std::vector<IntegrationPoint> qpoints_;          ///< Quadrature points
    std::vector<Scalar> shape_values_;               ///< Precomputed shape values at quad points
    std::vector<Tensor<1, 3>> shape_grads_;          ///< Precomputed shape gradients at quad points
    
    /// Initialize quadrature and precompute shape functions
    void initialize_quadrature() {
        setup_quadrature();
        precompute_shape_functions();
    }
    
    /// Setup quadrature rule (to be implemented by derived classes)
    virtual void setup_quadrature() = 0;
    
    /// Precompute shape functions at quadrature points
    virtual void precompute_shape_functions() {
        shape_values_.resize(n_qpoints_ * dofs_per_cell_);
        shape_grads_.resize(n_qpoints_ * dofs_per_cell_);
        
        for (int q = 0; q < n_qpoints_; ++q) {
            std::vector<Scalar> vals(dofs_per_cell_);
            std::vector<Tensor<1, 3>> grads(dofs_per_cell_);
            
            shape_values(qpoints_[q].coord, vals);
            shape_gradients(qpoints_[q].coord, grads);
            
            for (int i = 0; i < dofs_per_cell_; ++i) {
                shape_values_[q * dofs_per_cell_ + i] = vals[i];
                shape_grads_[q * dofs_per_cell_ + i] = grads[i];
            }
        }
    }
};

/**
 * @brief Factory function to create finite element
 * @param type Element geometry type
 * @param degree Polynomial degree
 * @param n_components Number of components (1 for scalar, dim for vector)
 * @return Unique pointer to finite element
 */
std::unique_ptr<FiniteElement> create_fe(GeometryType type, int degree, int n_components = 1);

}  // namespace mpfem

#endif  // MPFEM_FEM_FE_BASE_HPP