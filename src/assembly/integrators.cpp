#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include <Eigen/Dense>

namespace mpfem {

// =============================================================================
// DiffusionIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                 ElementTransform& trans,
                                                 Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int dim = refElem.dim();
    const int nq = refElem.numQuadraturePoints();
    
    // Use passed-in elmat directly (no allocation here)
    elmat.setZero(nd, nd);
    
    // Temporary gradient matrix - stack allocated for small sizes
    // Use Eigen's fixed-size matrices for better performance
    Eigen::MatrixXd gradMat(nd, dim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi_vals = refElem.shapeValuesAtQuad(q);
        const Vector3* refGrads = refElem.shapeGradientsAtQuad(q);
        
        // Build gradient matrix (nd x dim)
        // physGrad[i] = J^{-T} * refGrad[i]
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            for (int d = 0; d < dim; ++d) {
                gradMat(i, d) = physGrad[d];
            }
        }
        
        const Real coeff = evalCoefficient(trans);
        
        // Vectorized: elmat += w * coeff * (gradMat * gradMat^T)
        elmat.noalias() += w * coeff * (gradMat * gradMat.transpose());
    }
}

// =============================================================================
// MassIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void MassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                            ElementTransform& trans,
                                            Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nq = refElem.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real coeff = evalCoefficient(trans);
        
        // Vectorized using Eigen::Map - phi is stored contiguously
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        elmat.noalias() += w * coeff * (phiMap * phiMap.transpose());
    }
}

// =============================================================================
// DomainLFIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void DomainLFIntegrator::assembleElementVector(const ReferenceElement& refElem,
                                                ElementTransform& trans,
                                                Vector& elvec) const {
    const int nd = refElem.numDofs();
    const int nq = refElem.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real f = evalCoefficient(trans);
        
        // Vectorized using Eigen::Map
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        elvec.noalias() += w * f * phiMap;
    }
}

// =============================================================================
// BoundaryLFIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                               FacetElementTransform& trans,
                                               Vector& elvec) const {
    const int nd = refElem.numDofs();
    const int nq = refElem.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real g = evalCoefficient(trans);
        
        // Vectorized using Eigen::Map
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        elvec.noalias() += w * g * phiMap;
    }
}

// =============================================================================
// ConvectionBoundaryIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void ConvectionBoundaryIntegrator::assembleFaceMatrix(const ReferenceElement& refElem,
                                                       FacetElementTransform& trans,
                                                       Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nq = refElem.numQuadraturePoints();
    
    elmat.setZero(nd, nd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real h = evalCoefficient(trans);
        
        // Vectorized using Eigen::Map
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        elmat.noalias() += w * h * (phiMap * phiMap.transpose());
    }
}

void ConvectionBoundaryIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                       FacetElementTransform& trans,
                                                       Vector& elvec) const {
    const int nd = refElem.numDofs();
    const int nq = refElem.numQuadraturePoints();
    
    elvec.setZero(nd);
    
    if (!Tinf_) return;
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real h = evalCoefficient(trans);
        const Real Tinf = Tinf_->eval(trans);
        
        // Vectorized using Eigen::Map
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        elvec.noalias() += w * h * Tinf * phiMap;
    }
}

// =============================================================================
// VectorMassIntegrator - Optimized with Precomputed Shape Values
// =============================================================================

void VectorMassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                  ElementTransform& trans,
                                                  int vdim,
                                                  Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nvd = nd * vdim;
    const int nq = refElem.numQuadraturePoints();
    
    elmat.setZero(nvd, nvd);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = refElem.integrationPoint(q);
        trans.setIntegrationPoint(ip);
        
        const Real w = ip.weight * trans.weight();
        
        // Use precomputed shape values - NO ALLOCATION
        const Real* phi = refElem.shapeValuesAtQuad(q);
        
        const Real rho = evalCoefficient(trans);
        
        // Vectorized using Eigen::Map
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        // Mass matrix for each component (block diagonal)
        const Eigen::MatrixXd massBlock = w * rho * (phiMap * phiMap.transpose());
        
        for (int c = 0; c < vdim; ++c) {
            elmat.block(c * nd, c * nd, nd, nd) += massBlock;
        }
    }
}

}  // namespace mpfem
