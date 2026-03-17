#include "integrators.hpp"

namespace mpfem {

// =============================================================================
// 标量场积分器
// =============================================================================

void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                 ElementTransform& trans,
                                                 Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    Eigen::MatrixXd gradMat(nd, dim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real coef = evalCoef(trans);
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            for (int d = 0; d < dim; ++d)
                gradMat(i, d) = physGrad[d];
        }
        
        elmat.noalias() += w * coef * (gradMat * gradMat.transpose());
    }
}

void MassIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                            ElementTransform& trans,
                                            Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real coef = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elmat.noalias() += w * coef * (phiMap * phiMap.transpose());
    }
}

void DomainLFIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                ElementTransform& trans,
                                                Vector& elvec) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real f = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * f * phiMap;
    }
}

void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
                                               FacetElementTransform& trans,
                                               Vector& elvec) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real g = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * g * phiMap;
    }
}

void ConvectionMassIntegrator::assembleFaceMatrix(const ReferenceElement& ref,
                                                   FacetElementTransform& trans,
                                                   Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real h = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elmat.noalias() += w * h * (phiMap * phiMap.transpose());
    }
}

void ConvectionLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
                                                 FacetElementTransform& trans,
                                                 Vector& elvec) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    if (!Tinf_) return;
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real h = evalCoef(trans);
        const Real Tinf = Tinf_->eval(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * h * Tinf * phiMap;
    }
}

// =============================================================================
// 向量场积分器（弹性力学）
// =============================================================================

void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Matrix& elmat,
                                                  int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    const int totalDofs = nd * vdim;
    
    elmat.setZero(totalDofs, totalDofs);
    
    // 获取材料属性
    Real E = E_ ? E_->eval(trans) : 1.0;
    Real nu = nu_ ? nu_->eval(trans) : 0.3;
    
    // Lame参数
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    
    // 预计算本构矩阵 C (各向同性) - 使用固定大小矩阵
    Eigen::Matrix<Real, 6, 6> C;
    C.setZero();
    C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2.0 * mu;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
    C(3, 3) = C(4, 4) = C(5, 5) = mu;
    
    // 使用固定大小栈数组
    // B 矩阵: 6 x (nd * vdim), CB 矩阵: 6 x (nd * vdim)
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> CB_full;
    
    // 使用 Map 来创建正确大小的视图
    auto B = B_full.leftCols(totalDofs);
    auto CB = CB_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // B矩阵: 应变-位移关系矩阵 (6 x nd*3)
        B.setZero();
        
        for (int a = 0; a < nd; ++a) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[a].data(), physGrad.data());
            
            int col = a * vdim;
            B(0, col + 0) = physGrad[0];
            B(1, col + 1) = physGrad[1];
            B(2, col + 2) = physGrad[2];
            B(3, col + 1) = physGrad[2];
            B(3, col + 2) = physGrad[1];
            B(4, col + 0) = physGrad[2];
            B(4, col + 2) = physGrad[0];
            B(5, col + 0) = physGrad[1];
            B(5, col + 1) = physGrad[0];
        }
        
        // 刚度矩阵: K = B^T * C * B
        CB.noalias() = C * B;
        elmat.noalias() += w * (B.transpose() * CB);
    }
}

void ThermalLoadIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                   ElementTransform& trans,
                                                   Vector& elvec,
                                                   int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    const int totalDofs = nd * vdim;
    
    elvec.setZero(totalDofs);
    
    if (!alphaT_) return;
    
    // 使用固定大小栈数组
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    auto B = B_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        // 获取材料属性
        Real E = E_ ? E_->eval(trans) : 1.0;
        Real nu = nu_ ? nu_->eval(trans) : 0.3;
        
        // 获取热膨胀应变: alphaT_->eval(trans) 返回 alpha_T * (T - Tref)
        // 对于 ThermalExpansionCoefficient，这已经包含了温度差
        Real thermalStrain = alphaT_->eval(trans);
        
        if (std::abs(thermalStrain) < 1e-20) continue;
        
        // Lame参数
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        
        // 热应力: sigma_th = (3*lambda + 2*mu) * thermalStrain * I
        // 其中 thermalStrain = alpha_T * (T - Tref)
        Real diag = (3.0 * lambda + 2.0 * mu) * thermalStrain;
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // B矩阵
        B.setZero();
        
        for (int a = 0; a < nd; ++a) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[a].data(), physGrad.data());
            
            int col = a * vdim;
            B(0, col + 0) = physGrad[0];
            B(1, col + 1) = physGrad[1];
            B(2, col + 2) = physGrad[2];
            B(3, col + 1) = physGrad[2];
            B(3, col + 2) = physGrad[1];
            B(4, col + 0) = physGrad[2];
            B(4, col + 2) = physGrad[0];
            B(5, col + 0) = physGrad[1];
            B(5, col + 1) = physGrad[0];
        }
        
        // 热载荷向量: f = B^T * sigma_th (负号因为热膨胀产生初始应变)
        // 手动计算避免临时向量
        for (int i = 0; i < totalDofs; ++i) {
            elvec(i) -= w * (B(0, i) + B(1, i) + B(2, i)) * diag;
        }
    }
}

}  // namespace mpfem