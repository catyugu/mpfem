#ifndef MPFEM_GRID_FUNCTION_HPP
#define MPFEM_GRID_FUNCTION_HPP

#include "fe/fe_space.hpp"
#include "fe/quadrature.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace mpfem {

// Forward declaration
class ElementTransform;

/**
 * @brief Vector type for grid function values.
 */
using GridVector = Eigen::VectorXd;

/**
 * @brief Single field value storage and interpolation.
 * 
 * GridFunction stores the nodal values of a finite element field
 * and provides interpolation capabilities.
 */
class GridFunction {
public:
    /// Default constructor
    GridFunction() = default;

    /// Construct from FE space
    explicit GridFunction(const FESpace* fes);

    /// Construct from FE space with initial value
    GridFunction(const FESpace* fes, Real initValue);

    // -------------------------------------------------------------------------
    // FE Space access
    // -------------------------------------------------------------------------

    /// Get the FE space
    const FESpace* fes() const { return fes_; }

    /// Get number of degrees of freedom
    Index numDofs() const { return static_cast<Index>(values_.size()); }

    /// Get vector dimension (1 = scalar, 2/3 = vector)
    int vdim() const { return fes_ ? fes_->vdim() : 1; }

    // -------------------------------------------------------------------------
    // Value access
    // -------------------------------------------------------------------------

    /// Get all values
    const GridVector& values() const { return values_; }
    GridVector& values() { return values_; }

    /// Get value by index
    Real operator()(Index i) const { return values_[i]; }
    Real& operator()(Index i) { return values_[i]; }

    /// Set all values
    void setConstant(Real c) { values_.setConstant(c); }

    /// Set from vector
    void setValues(const GridVector& v) { values_ = v; }

    /// Set from raw pointer
    void setValues(const Real* data, Index size);

    // -------------------------------------------------------------------------
    // Interpolation
    // -------------------------------------------------------------------------

    /**
     * @brief Evaluate field at a point in reference coordinates.
     * @param elemIdx Element index.
     * @param xi Reference coordinates (size = dim).
     * @return Field value.
     */
    Real eval(Index elemIdx, const Real* xi) const;

    /**
     * @brief Evaluate field at an integration point.
     */
    Real eval(Index elemIdx, const IntegrationPoint& ip) const {
        return eval(elemIdx, &ip.xi);
    }

    /**
     * @brief Evaluate vector field at a point in reference coordinates.
     * @param elemIdx Element index.
     * @param xi Reference coordinates.
     * @return Vector field value (size = vdim).
     */
    Vector3 evalVector(Index elemIdx, const Real* xi) const;

    /**
     * @brief Compute gradient of scalar field in physical coordinates.
     * @param elemIdx Element index.
     * @param xi Reference coordinates.
     * @param trans Element transformation (non-const, will be modified to set integration point).
     * @return Gradient in physical coordinates.
     */
    Vector3 gradient(Index elemIdx, const Real* xi, ElementTransform& trans) const;

    /**
     * @brief Compute gradient at integration point.
     */
    Vector3 gradient(Index elemIdx, const IntegrationPoint& ip, 
                     ElementTransform& trans) const {
        return gradient(elemIdx, &ip.xi, trans);
    }

    // -------------------------------------------------------------------------
    // Element-level operations
    // -------------------------------------------------------------------------

    /**
     * @brief Get element DOF values.
     * @param elemIdx Element index.
     * @param values Output values (size = numElementDofs).
     */
    void getElementValues(Index elemIdx, std::vector<Real>& values) const;

    /**
     * @brief Get element DOF values as Eigen vector.
     */
    Eigen::VectorXd getElementValues(Index elemIdx) const;

    /**
     * @brief Add contribution to element DOF values.
     * @param elemIdx Element index.
     * @param localValues Local contribution.
     * @param scale Scaling factor.
     */
    void addElementValues(Index elemIdx, const Eigen::VectorXd& localValues, Real scale = 1.0);

    /**
     * @brief Set element DOF values.
     */
    void setElementValues(Index elemIdx, const Eigen::VectorXd& localValues);

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    /// Compute L2 norm
    Real l2Norm() const { return values_.norm(); }

    /// Compute max norm
    Real maxNorm() const { return values_.cwiseAbs().maxCoeff(); }

    /// Compute min value
    Real minValue() const { return values_.minCoeff(); }

    /// Compute max value
    Real maxValue() const { return values_.maxCoeff(); }

    /// Set to zero
    void setZero() { values_.setZero(); }

    /// Clear
    void clear() {
        values_.resize(0);
        fes_ = nullptr;
    }

private:
    const FESpace* fes_ = nullptr;
    GridVector values_;
    mutable std::vector<Index> dofCache_;  ///< Cache for DOF indices
};

}  // namespace mpfem

#endif  // MPFEM_GRID_FUNCTION_HPP
