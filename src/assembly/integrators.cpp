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
        Matrix3 D;
        coef_->eval(trans, D);
        
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
        Real coef;
        coef_->eval(trans, coef);
        
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
        Real f;
        coef_->eval(trans, f);
        
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
        Real g;
        coef_->eval(trans, g);
        
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
        Real h;
        coef_->eval(trans, h);
        
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
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        Real h;
        coef_->eval(trans, h);
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

namespace {
// Helper function to compute strain-displacement B matrix from shape function gradients
template <typename BMatrix>
inline void computeStrainDispMatrix(
    BMatrix& B,
    const ReferenceElement& ref,
    ElementTransform& trans,
    int nd,
    int vdim,
    int q)
{
    const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
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
}

// Helper function to fill elasticity tensor C from Lamé parameters
inline void fillElasticityTensorC(Eigen::Matrix<Real, 6, 6>& C, Real lambda, Real mu) {
    C.setZero();
    C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2.0 * mu;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
    C(3, 3) = C(4, 4) = C(5, 5) = mu;
}

} // anonymous namespace

void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    const int totalDofs = nd * vdim_;
    
    elmat.setZero(totalDofs, totalDofs);
    
    Real E_val, nu_val;
    E_->eval(trans, E_val);
    nu_->eval(trans, nu_val);
    
    Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
    Real mu = E_val / (2.0 * (1.0 + nu_val));
    
    Eigen::Matrix<Real, 6, 6> C;
    fillElasticityTensorC(C, lambda, mu);
    
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> CB_full;
    
    auto B = B_full.leftCols(totalDofs);
    auto CB = CB_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        B.setZero();
        
        computeStrainDispMatrix(B, ref, trans, nd, vdim_, q);
        
        CB.noalias() = C * B;
        elmat.noalias() += w * (B.transpose() * CB);
    }
}

void StrainLoadIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Vector& elvec) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    const int totalDofs = nd * vdim_;
    
    elvec.setZero(totalDofs);
    
    Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
    Eigen::Matrix<Real, 6, 1> sigmaVoigt;
    Matrix3 stress;
    auto B = B_full.leftCols(totalDofs);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        stress_->eval(trans, stress);

        if (stress.norm() < 1e-20) continue;

        sigmaVoigt(0) = stress(0, 0);
        sigmaVoigt(1) = stress(1, 1);
        sigmaVoigt(2) = stress(2, 2);
        sigmaVoigt(3) = stress(1, 2);
        sigmaVoigt(4) = stress(0, 2);
        sigmaVoigt(5) = stress(0, 1);
        
        B.setZero();
        
        computeStrainDispMatrix(B, ref, trans, nd, vdim_, q);
        
        elvec.noalias() += w * (B.transpose() * sigmaVoigt);
    }
}

}  // namespace mpfem
