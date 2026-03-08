#pragma once

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "mpfem/core/eigen_types.hpp"

namespace mpfem {

// Forward declarations
class ElementTransformation;
class IntegrationPoint;
class GridFunction;

// ============================================================================
// Coefficient - 标量系数基类
// ============================================================================

class Coefficient {
 public:
  virtual ~Coefficient() = default;

  // Evaluate coefficient at a point
  virtual double Eval(ElementTransformation& T,
                      const IntegrationPoint& ip) const = 0;
};

// ============================================================================
// ConstantCoefficient - 常数系数
// ============================================================================

class ConstantCoefficient : public Coefficient {
 public:
  explicit ConstantCoefficient(double c = 1.0) : constant_(c) {}

  void SetConstant(double c) { constant_ = c; }
  double GetConstant() const { return constant_; }

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override {
    (void)T;
    (void)ip;
    return constant_;
  }

 private:
  double constant_;
};

// ============================================================================
// PWConstCoefficient - 分片常数系数（按域ID）
// ============================================================================

class PWConstCoefficient : public Coefficient {
 public:
  PWConstCoefficient() = default;

  // Set value for a specific domain ID
  void SetValue(int domain_id, double value) { values_[domain_id] = value; }

  // Get value for a specific domain ID
  double GetValue(int domain_id) const {
    auto it = values_.find(domain_id);
    if (it != values_.end()) {
      return it->second;
    }
    return 0.0;  // Default value
  }

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override;

 private:
  std::map<int, double> values_;  // domain_id -> value
};

// ============================================================================
// FunctionCoefficient - 函数系数
// ============================================================================

class FunctionCoefficient : public Coefficient {
 public:
  using FuncType = std::function<double(const Vec3&)>;

  explicit FunctionCoefficient(FuncType func) : func_(std::move(func)) {}

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override;

 private:
  FuncType func_;
};

// ============================================================================
// GridFunctionCoefficient - 网格函数系数
// ============================================================================

class GridFunctionCoefficient : public Coefficient {
 public:
  GridFunctionCoefficient() = default;
  explicit GridFunctionCoefficient(const GridFunction* gf, int comp = 0)
      : gf_(gf), comp_(comp) {}

  void SetGridFunction(const GridFunction* gf) { gf_ = gf; }
  void SetComponent(int comp) { comp_ = comp; }

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override;

 private:
  const GridFunction* gf_ = nullptr;
  int comp_ = 0;
};

// ============================================================================
// VectorCoefficient - 向量系数基类
// ============================================================================

class VectorCoefficient {
 public:
  virtual ~VectorCoefficient() = default;

  virtual int GetVDim() const = 0;

  // Evaluate coefficient at a point, result stored in V
  virtual void Eval(Vec3& V, ElementTransformation& T,
                    const IntegrationPoint& ip) const = 0;
};

// ============================================================================
// ConstantVectorCoefficient - 常向量系数
// ============================================================================

class ConstantVectorCoefficient : public VectorCoefficient {
 public:
  explicit ConstantVectorCoefficient(const Vec3& v = Vec3::Zero()) : vec_(v) {}

  void SetVector(const Vec3& v) { vec_ = v; }

  int GetVDim() const override { return 3; }

  void Eval(Vec3& V, ElementTransformation& T,
            const IntegrationPoint& ip) const override {
    (void)T;
    (void)ip;
    V = vec_;
  }

 private:
  Vec3 vec_;
};

// ============================================================================
// GridFunctionGradientCoefficient - 网格函数梯度系数
// ============================================================================

class GridFunctionGradientCoefficient : public VectorCoefficient {
 public:
  explicit GridFunctionGradientCoefficient(const GridFunction* gf)
      : gf_(gf) {}

  int GetVDim() const override { return 3; }

  void Eval(Vec3& V, ElementTransformation& T,
            const IntegrationPoint& ip) const override;

 private:
  const GridFunction* gf_;
};

// ============================================================================
// MatrixCoefficient - 矩阵系数基类
// ============================================================================

class MatrixCoefficient {
 public:
  virtual ~MatrixCoefficient() = default;

  virtual int GetVDim() const = 0;

  // Evaluate coefficient at a point, result stored in M
  virtual void Eval(Mat3& M, ElementTransformation& T,
                    const IntegrationPoint& ip) const = 0;
};

// ============================================================================
// IdentityMatrixCoefficient - 单位矩阵系数
// ============================================================================

class IdentityMatrixCoefficient : public MatrixCoefficient {
 public:
  explicit IdentityMatrixCoefficient(int dim = 3) : dim_(dim) {}

  int GetVDim() const override { return dim_; }

  void Eval(Mat3& M, ElementTransformation& T,
            const IntegrationPoint& ip) const override {
    (void)T;
    (void)ip;
    M.setIdentity();
  }

 private:
  int dim_;
};

// ============================================================================
// IsotropicElasticityCoefficient - 各向同性弹性系数
// ============================================================================

class IsotropicElasticityCoefficient : public MatrixCoefficient {
 public:
  IsotropicElasticityCoefficient(double E, double nu)
      : E_(E), nu_(nu), dim_(3) {
    // Precompute Lamé parameters
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    mu_ = E / (2.0 * (1.0 + nu));
  }

  int GetVDim() const override { return dim_; }

  void Eval(Mat3& M, ElementTransformation& T,
            const IntegrationPoint& ip) const override;

  // Get the elasticity tensor in Voigt notation
  // Returns 6x6 matrix for 3D elasticity
  Mat6 GetElasticityTensor() const;

 private:
  double E_;    // Young's modulus
  double nu_;   // Poisson's ratio
  double lambda_;  // Lamé's first parameter
  double mu_;      // Lamé's second parameter (shear modulus)
  int dim_;
};

// ============================================================================
// PWMatrixCoefficient - 分片矩阵系数
// ============================================================================

class PWMatrixCoefficient : public MatrixCoefficient {
 public:
  PWMatrixCoefficient(int dim = 3) : dim_(dim) {}

  int GetVDim() const override { return dim_; }

  void SetCoefficient(int domain_id, std::shared_ptr<MatrixCoefficient> coef) {
    coefficients_[domain_id] = coef;
  }

  void Eval(Mat3& M, ElementTransformation& T,
            const IntegrationPoint& ip) const override;

 private:
  int dim_;
  std::map<int, std::shared_ptr<MatrixCoefficient>> coefficients_;
};

// ============================================================================
// Helper function to create material property coefficient
// ============================================================================

std::unique_ptr<PWConstCoefficient> CreateMaterialPropertyCoefficient(
    const std::map<int, double>& domain_values);

std::unique_ptr<PWMatrixCoefficient> CreateElasticityCoefficient(
    const std::map<int, std::pair<double, double>>& E_nu_values);

}  // namespace mpfem
