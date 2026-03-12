#include "assembly/integrators.hpp"
#include <Eigen/Dense>

namespace mpfem {

// =============================================================================
// DiffusionIntegrator - Vectorized Implementation
// =============================================================================

void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                 ElementTransform& trans,
                                                 Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int dim = refElem.dim();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate gradient matrix for vectorized computation
    Eigen::MatrixXd gradMat(nd, dim);
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build gradient matrix (nd x dim)
        // physGrad[i] = J^{-T} * refGrad[i]
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(sv.gradients[i], physGrad);
            for (int d = 0; d < dim; ++d) {
                gradMat(i, d) = physGrad[d];
            }
            phi(i) = sv.values[i];
        }
        
        Real coeff = evalCoefficient(trans);
        
        // Vectorized: elmat += w * coeff * (gradMat * gradMat^T)
        elmat.noalias() += w * coeff * (gradMat * gradMat.transpose());
    }
}

// =============================================================================
// MassIntegrator - Vectorized Implementation
// =============================================================================

void MassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                            ElementTransform& trans,
                                            Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real coeff = evalCoefficient(trans);
        
        // Vectorized: elmat += w * coeff * (phi * phi^T)
        elmat.noalias() += w * coeff * (phi * phi.transpose());
    }
}

// =============================================================================
// DomainLFIntegrator - Vectorized Implementation
// =============================================================================

void DomainLFIntegrator::assembleElementVector(const ReferenceElement& refElem,
                                                ElementTransform& trans,
                                                Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real f = evalCoefficient(trans);
        
        // Vectorized: elvec += w * f * phi
        elvec.noalias() += w * f * phi;
    }
}

// =============================================================================
// BoundaryLFIntegrator - Vectorized Implementation
// =============================================================================

void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                               FacetElementTransform& trans,
                                               Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real g = evalCoefficient(trans);
        
        // Vectorized: elvec += w * g * phi
        elvec.noalias() += w * g * phi;
    }
}

// =============================================================================
// ConvectionBoundaryIntegrator - Vectorized Implementation
// =============================================================================

void ConvectionBoundaryIntegrator::assembleFaceMatrix(const ReferenceElement& refElem,
                                                       FacetElementTransform& trans,
                                                       Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real h = evalCoefficient(trans);
        
        // Vectorized: elmat += w * h * (phi * phi^T)
        elmat.noalias() += w * h * (phi * phi.transpose());
    }
}

void ConvectionBoundaryIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                       FacetElementTransform& trans,
                                                       Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    if (!Tinf_) return;
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real h = evalCoefficient(trans);
        Real Tinf = Tinf_->eval(trans);
        
        // Vectorized: elvec += w * h * Tinf * phi
        elvec.noalias() += w * h * Tinf * phi;
    }
}

// =============================================================================
// VectorMassIntegrator - Vectorized Implementation
// =============================================================================

void VectorMassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                  ElementTransform& trans,
                                                  int vdim,
                                                  Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nvd = nd * vdim;
    
    elmat.setZero(nvd, nvd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    // Pre-allocate shape function vector
    Eigen::VectorXd phi(nd);
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        // Build shape function vector
        for (int i = 0; i < nd; ++i) {
            phi(i) = sv.values[i];
        }
        
        Real rho = evalCoefficient(trans);
        
        // Mass matrix for each component (block diagonal)
        // Using block operations for vectorization
        Eigen::MatrixXd massBlock = w * rho * (phi * phi.transpose());
        
        for (int c = 0; c < vdim; ++c) {
            elmat.block(c * nd, c * nd, nd, nd) += massBlock;
        }
    }
}

}  // namespace mpfem
