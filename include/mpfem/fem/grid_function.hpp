#pragma once

#include <memory>
#include <vector>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/fem/fe_space.hpp"

namespace mpfem {

// Forward declarations
class Coefficient;
class VectorCoefficient;
class ElementTransformation;
class IntegrationPoint;

// ============================================================================
// GridFunction - 网格函数
// ============================================================================

class GridFunction {
 public:
  GridFunction() = default;
  explicit GridFunction(const FiniteElementSpace* fes);

  // Initialize with FE space
  void SetSpace(const FiniteElementSpace* fes);

  // Get the FE space
  const FiniteElementSpace* GetFES() const { return fes_; }

  // Get the size of the vector
  int Size() const { return static_cast<int>(data_.size()); }

  // Access the underlying vector data
  VectorXd& Data() { return data_; }
  const VectorXd& Data() const { return data_; }

  // Element access
  double& operator()(int i) { return data_(i); }
  double operator()(int i) const { return data_(i); }

  // Get the value at a node
  double GetValue(int node_idx) const {
    if (fes_->GetVDim() == 1) {
      return data_(node_idx);
    } else {
      // For vector fields, return magnitude or first component
      return data_(node_idx * fes_->GetVDim());
    }
  }

  // Get vector value at a node (for vector fields)
  void GetVectorValue(int node_idx, Vec3& val) const {
    int vdim = fes_->GetVDim();
    for (int c = 0; c < vdim && c < 3; ++c) {
      val(c) = data_(node_idx * vdim + c);
    }
  }

  // Get the value at an integration point within an element
  double GetValue(const ElementGroup& group, int elem_idx,
                  const IntegrationPoint& ip) const;

  // Get the gradient at an integration point
  void GetGradient(const ElementGroup& group, int elem_idx,
                   const IntegrationPoint& ip, Vec3& grad) const;

  // Get the gradient with transformation
  void GetGradient(ElementTransformation& T, const IntegrationPoint& ip,
                   Vec3& grad) const;

  // Project a coefficient onto this grid function
  void ProjectCoefficient(const Coefficient& coeff);

  // Set all values to zero
  void SetZero() { data_.setZero(); }

  // Set all values to a constant
  void SetConstant(double c) { data_.setConstant(c); }

  // Get min and max values
  double Min() const { return data_.minCoeff(); }
  double Max() const { return data_.maxCoeff(); }

  // Compute the L2 norm
  double Norm() const { return data_.norm(); }

  // Compute the L2 norm squared
  double NormSquared() const { return data_.squaredNorm(); }

 private:
  const FiniteElementSpace* fes_ = nullptr;
  VectorXd data_;
};

// ============================================================================
// GridFunctionArray - Array of grid functions
// ============================================================================

class GridFunctionArray {
 public:
  GridFunctionArray() = default;

  void AddGridFunction(const std::string& name,
                       std::unique_ptr<GridFunction> gf) {
    gfs_[name] = std::move(gf);
  }

  GridFunction* GetGridFunction(const std::string& name) {
    auto it = gfs_.find(name);
    if (it != gfs_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  const GridFunction* GetGridFunction(const std::string& name) const {
    auto it = gfs_.find(name);
    if (it != gfs_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  std::map<std::string, std::unique_ptr<GridFunction>> gfs_;
};

}  // namespace mpfem
