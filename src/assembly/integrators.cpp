#include "integrators.hpp"

namespace mpfem {

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

void ConvectionBoundaryIntegrator::assembleFaceMatrix(const ReferenceElement& ref,
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

void ConvectionBoundaryIntegrator::assembleFaceVector(const ReferenceElement& ref,
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

}  // namespace mpfem
