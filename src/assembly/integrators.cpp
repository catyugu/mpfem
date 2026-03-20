#include "integrators.hpp"

namespace mpfem {

// =============================================================================
// Diffusion integrator (matrix coefficient)
// =============================================================================

void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                 ElementTransform& trans,
                                                 Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    Eigen::MatrixXd gradMat(nd, 3);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        Matrix3 D = Matrix3::Identity();
        if (coef_) coef_->eval(trans, D);
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            gradMat.row(i) = physGrad;
        }
        
        Eigen::MatrixXd Dgrad = gradMat * D;
        elmat.noalias() += w * (gradMat * Dgrad.transpose());
    }
}

// =============================================================================
// Mass integrator
// =============================================================================

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
        Real coef = 1.0;
        if (coef_) coef_->eval(trans, coef);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elmat.noalias() += w * coef * (phiMap * phiMap.transpose());
    }
}

// =============================================================================
// Domain load integrator
// =============================================================================

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
        Real f = 1.0;
        if (coef_) coef_->eval(trans, f);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * f * phiMap;
    }
}

// =============================================================================
// Boundary load integrator
// =============================================================================

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
        Real g = 1.0;
        if (coef_) coef_->eval(trans, g);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * g * phiMap;
    }
}

// =============================================================================
// Convection integrators (Robin BC)
// =============================================================================

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
        Real h = 1.0;
        if (coef_) coef_->eval(trans, h);
        
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
        Real h = 1.0;
        if (coef_) coef_->eval(trans, h);
        Real Tinf;
        Tinf_->eval(trans, Tinf);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        elvec.noalias() += w * h * Tinf * phiMap;
    }
}

// =============================================================================
// Elasticity integrators
// =============================================================================

void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Matrix& elmat,
                                                  int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    const int totalDofs = nd * vdim;
    
    elmat.setZero(totalDofs, totalDofs);
    
    Real E_val = 1.0, nu_val = 0.3;
    if (E_) E_->eval(trans, E_val);
    if (nu_) nu_->eval(trans, nu_val);
    
    Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
    Real mu = E_val / (2.0 * (1.0 + nu_val));
    
    Eigen::Matrix<Real, 6, 6> C;
    C.setZero();
    C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2.0 * mu;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
    C(3, 3) = C(4, 4) = C(5, 5) = mu;
    
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> CB_full;
    
    auto B = B_full.leftCols(totalDofs);
    auto CB = CB_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
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
    
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    auto B = B_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        Real E_val = 1.0, nu_val = 0.3;
        if (E_) E_->eval(trans, E_val);
        if (nu_) nu_->eval(trans, nu_val);
        
        Real thermalStrain;
        alphaT_->eval(trans, thermalStrain);
        
        if (std::abs(thermalStrain) < 1e-20) continue;
        
        Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
        Real mu = E_val / (2.0 * (1.0 + nu_val));
        
        Real diag = (3.0 * lambda + 2.0 * mu) * thermalStrain;
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
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
        
        for (int i = 0; i < totalDofs; ++i) {
            elvec(i) -= w * (B(0, i) + B(1, i) + B(2, i)) * diag;
        }
    }
}

}  // namespace mpfem
