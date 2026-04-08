#ifndef MPFEM_GRID_FUNCTION_HPP
#define MPFEM_GRID_FUNCTION_HPP

#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include <Eigen/Dense>
#include <cstdint>

namespace mpfem {

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
    
    void setZero() { values_.setZero(); ++revision_; }
    void setConstant(Real c) { values_.setConstant(c); ++revision_; }
    void setValues(const Eigen::VectorXd& v) { values_ = v; ++revision_; }

    void markUpdated() { ++revision_; }
    std::uint64_t revision() const { return revision_; }
    
    Real l2Norm() const { return values_.norm(); }
    Real maxNorm() const { return values_.cwiseAbs().maxCoeff(); }
    Real minValue() const { return values_.minCoeff(); }
    Real maxValue() const { return values_.maxCoeff(); }
    
    Eigen::VectorXd getElementValues(Index elem) const {
        if (!fes_) return Eigen::VectorXd();
        const int totalDofs = fes_->numElementDofs(elem);
        std::vector<Index> dofs(totalDofs);
        fes_->getElementDofs(elem, std::span<Index>{dofs.data(), static_cast<size_t>(totalDofs)});
        Eigen::VectorXd result(dofs.size());
        for (size_t i = 0; i < dofs.size(); ++i)
            result[i] = values_[dofs[i]];
        return result;
    }
    
    Real eval(Index elem, const Real* xi) const;
    Vector3 gradient(Index elem, const Real* xi, const Matrix3& invJacobianTranspose) const;
    
private:
    const FESpace* fes_ = nullptr;
    Eigen::VectorXd values_;
    std::uint64_t revision_ = 0;
};

}  // namespace mpfem

#endif  // MPFEM_GRID_FUNCTION_HPP
