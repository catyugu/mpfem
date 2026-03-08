#pragma once

#include <memory>

#include "mpfem/fem/integrator.hpp"
#include "mpfem/fem/coefficient.hpp"

namespace mpfem {

// ============================================================================
// DiffusionIntegrator - 扩散积分器: a(u,v) = (Q ∇u, ∇v)
// 用于: 静电场 (Q = 电导率σ), 热传导 (Q = 热导率k)
// ============================================================================

class DiffusionIntegrator : public BilinearFormIntegrator {
 public:
  DiffusionIntegrator() : Q_(nullptr) {}
  explicit DiffusionIntegrator(Coefficient* Q) : Q_(Q) {}

  void SetCoefficient(Coefficient* Q) { Q_ = Q; }

  void AssembleElementMatrix(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::MatrixXd& elmat) override;

  int GetIntegrationOrder() const override { return 2; }

 private:
  Coefficient* Q_;
};

// ============================================================================
// MassIntegrator - 质量积分器: a(u,v) = (Q u, v)
// 用于: 瞬态问题的时间导数项
// ============================================================================

class MassIntegrator : public BilinearFormIntegrator {
 public:
  MassIntegrator() : Q_(nullptr) {}
  explicit MassIntegrator(Coefficient* Q) : Q_(Q) {}

  void SetCoefficient(Coefficient* Q) { Q_ = Q; }

  void AssembleElementMatrix(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::MatrixXd& elmat) override;

  int GetIntegrationOrder() const override { return 2; }

 private:
  Coefficient* Q_;
};

// ============================================================================
// ElasticityIntegrator - 弹性力学积分器
// 用于: 固体力学 (刚度矩阵)
// ============================================================================

class ElasticityIntegrator : public BilinearFormIntegrator {
 public:
  ElasticityIntegrator() : lambda_(nullptr), mu_(nullptr) {}
  ElasticityIntegrator(Coefficient* lambda, Coefficient* mu)
      : lambda_(lambda), mu_(mu) {}

  void SetLameParameters(Coefficient* lambda, Coefficient* mu) {
    lambda_ = lambda;
    mu_ = mu;
  }

  void AssembleElementMatrix(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::MatrixXd& elmat) override;

  int GetIntegrationOrder() const override { return 2; }

 private:
  Coefficient* lambda_;
  Coefficient* mu_;
};

// ============================================================================
// VectorMassIntegrator - 矢量质量积分器: a(u,v) = (Q u, v)
// 用于: 矢量场的质量矩阵
// ============================================================================

class VectorMassIntegrator : public BilinearFormIntegrator {
 public:
  VectorMassIntegrator() : Q_(nullptr) {}
  explicit VectorMassIntegrator(Coefficient* Q) : Q_(Q) {}

  void SetCoefficient(Coefficient* Q) { Q_ = Q; }

  void AssembleElementMatrix(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::MatrixXd& elmat) override;

 private:
  Coefficient* Q_;
};

// ============================================================================
// DomainLFIntegrator - 区域线性形式积分器: L(v) = (f, v)
// 用于: 焦耳热源、体力等
// ============================================================================

class DomainLFIntegrator : public LinearFormIntegrator {
 public:
  DomainLFIntegrator() : Q_(nullptr) {}
  explicit DomainLFIntegrator(Coefficient* Q) : Q_(Q) {}

  void SetCoefficient(Coefficient* Q) { Q_ = Q; }

  void AssembleElementVector(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::VectorXd& elvec) override;

  int GetIntegrationOrder() const override { return 2; }

 private:
  Coefficient* Q_;
};

// ============================================================================
// BoundaryLFIntegrator - 边界线性形式积分器: L(v) = (g, v)_∂Ω
// 用于: Neumann 边界条件
// ============================================================================

class BoundaryLFIntegrator : public LinearFormIntegrator {
 public:
  BoundaryLFIntegrator() : Q_(nullptr) {}
  explicit BoundaryLFIntegrator(Coefficient* Q) : Q_(Q) {}

  void SetCoefficient(Coefficient* Q) { Q_ = Q; }

  void AssembleElementVector(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::VectorXd& elvec) override {
    // For domain elements, this does nothing
    (void)el;
    (void)T;
    elvec.setZero();
  }

  void AssembleBoundaryVector(const FiniteElement& el,
                               ElementTransformation& T,
                               Eigen::VectorXd& elvec) override;

 private:
  Coefficient* Q_;
};

// ============================================================================
// ConvectionIntegrator - 对流换热边界积分器
// 用于: 对流换热边界条件: q = h(T - T_ext)
// 边界项: h * (T, v) - h * T_ext * v
// ============================================================================

class ConvectionIntegrator : public BilinearFormIntegrator {
 public:
  ConvectionIntegrator() : h_(nullptr), T_ext_(nullptr) {}
  ConvectionIntegrator(Coefficient* h, Coefficient* T_ext)
      : h_(h), T_ext_(T_ext) {}

  void SetHeatTransferCoefficient(Coefficient* h) { h_ = h; }
  void SetExternalTemperature(Coefficient* T_ext) { T_ext_ = T_ext; }

  void AssembleElementMatrix(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::MatrixXd& elmat) override {
    // For domain elements, this does nothing
    (void)el;
    (void)T;
    elmat.setZero();
  }

  void AssembleFaceMatrix(const FiniteElement& el,
                           ElementTransformation& T,
                           Eigen::MatrixXd& elmat) override;

 private:
  Coefficient* h_;
  Coefficient* T_ext_;
};

// ============================================================================
// ConvectionRHSIntegrator - 对流换热右端项积分器
// 用于: 对流换热的右端项: h * T_ext
// ============================================================================

class ConvectionRHSIntegrator : public LinearFormIntegrator {
 public:
  ConvectionRHSIntegrator() : h_(nullptr), T_ext_(nullptr) {}
  ConvectionRHSIntegrator(Coefficient* h, Coefficient* T_ext)
      : h_(h), T_ext_(T_ext) {}

  void SetHeatTransferCoefficient(Coefficient* h) { h_ = h; }
  void SetExternalTemperature(Coefficient* T_ext) { T_ext_ = T_ext; }

  void AssembleElementVector(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::VectorXd& elvec) override {
    (void)el;
    (void)T;
    elvec.setZero();
  }

  void AssembleBoundaryVector(const FiniteElement& el,
                               ElementTransformation& T,
                               Eigen::VectorXd& elvec) override;

 private:
  Coefficient* h_;
  Coefficient* T_ext_;
};

// ============================================================================
// ThermalExpansionIntegrator - 热膨胀积分器
// 用于: 热膨胀应力作为初始应变
// ============================================================================

class ThermalExpansionIntegrator : public LinearFormIntegrator {
 public:
  ThermalExpansionIntegrator()
      : alpha_(nullptr), T_(nullptr), T_ref_(293.15), lambda_(nullptr), mu_(nullptr) {}
  ThermalExpansionIntegrator(Coefficient* alpha, Coefficient* T,
                              double T_ref, Coefficient* lambda, Coefficient* mu)
      : alpha_(alpha), T_(T), T_ref_(T_ref), lambda_(lambda), mu_(mu) {}

  void SetThermalExpansionCoefficient(Coefficient* alpha) { alpha_ = alpha; }
  void SetTemperatureField(Coefficient* T) { T_ = T; }
  void SetReferenceTemperature(double T_ref) { T_ref_ = T_ref; }
  void SetLameParameters(Coefficient* lambda, Coefficient* mu) {
    lambda_ = lambda;
    mu_ = mu;
  }

  void AssembleElementVector(const FiniteElement& el,
                              ElementTransformation& T,
                              Eigen::VectorXd& elvec) override;

 private:
  Coefficient* alpha_;  // Thermal expansion coefficient
  Coefficient* T_;      // Temperature field
  double T_ref_;        // Reference temperature
  Coefficient* lambda_; // Lamé's first parameter
  Coefficient* mu_;     // Lamé's second parameter
};

}  // namespace mpfem
