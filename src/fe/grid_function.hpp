#ifndef MPFEM_GRID_FUNCTION_HPP
#define MPFEM_GRID_FUNCTION_HPP

#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include <Eigen/Dense>

namespace mpfem {

class ElementTransform;

/**
 * @brief 场函数 - 最小化设计
 */
class GridFunction {
public:
    GridFunction() = default;
    explicit GridFunction(const FESpace* fes) : fes_(fes) {
        if (fes_) values_.resize(fes_->numDofs());
    }
    
    GridFunction(const FESpace* fes, Real initVal) : fes_(fes) {
        if (fes_) {
            values_.resize(fes_->numDofs());
            values_.setConstant(initVal);
        }
    }
    
    const FESpace* fes() const { return fes_; }
    Index numDofs() const { return values_.size(); }
    int vdim() const { return fes_ ? fes_->vdim() : 1; }
    
    const Eigen::VectorXd& values() const { return values_; }
    Eigen::VectorXd& values() { return values_; }
    Real operator()(Index i) const { return values_[i]; }
    Real& operator()(Index i) { return values_[i]; }
    
    void setZero() { values_.setZero(); }
    void setConstant(Real c) { values_.setConstant(c); }
    void setValues(const Eigen::VectorXd& v) { values_ = v; }
    
    Real l2Norm() const { return values_.norm(); }
    Real maxNorm() const { return values_.cwiseAbs().maxCoeff(); }
    Real minValue() const { return values_.minCoeff(); }
    Real maxValue() const { return values_.maxCoeff(); }
    
    Eigen::VectorXd getElementValues(Index elem) const {
        if (!fes_) return Eigen::VectorXd();
        std::vector<Index> dofs;
        fes_->getElementDofs(elem, dofs);
        Eigen::VectorXd result(dofs.size());
        for (size_t i = 0; i < dofs.size(); ++i)
            result[i] = values_[dofs[i]];
        return result;
    }
    
    Real eval(Index elem, const Real* xi) const;
    Vector3 gradient(Index elem, const Real* xi, ElementTransform& trans) const;
    
    /**
     * @brief Project field values to corner vertices only.
     * 
     * For high-order elements, the mesh contains more vertices than geometric corners.
     * This method extracts only the values at corner vertices for comparison with
     * reference solutions that only provide corner vertex values (e.g., COMSOL exports).
     * 
     * @param mesh The mesh to get corner vertex count from.
     * @return Vector of values at corner vertices only.
     * 
     * For scalar fields (vdim=1): returns values at corner vertices.
     * For vector fields (vdim>1): returns interleaved values [vx0,vy0,vz0, vx1,vy1,vz1, ...]
     */
    Eigen::VectorXd projectToCorners(const Mesh& mesh) const;
    
private:
    const FESpace* fes_ = nullptr;
    Eigen::VectorXd values_;
};

}  // namespace mpfem

#endif  // MPFEM_GRID_FUNCTION_HPP
