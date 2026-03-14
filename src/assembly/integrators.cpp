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

void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Matrix& elmat) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    const int vdim = 3;  // 3D displacement
    
    elmat.setZero(nd * vdim, nd * vdim);
    
    // Get material properties
    Real E = E_ ? E_->eval(trans) : 1.0;
    Real nu = nu_ ? nu_->eval(trans) : 0.3;
    
    // Lame parameters
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // Compute physical gradients for all shape functions
        Eigen::MatrixXd B(dim * vdim, nd * vdim);
        B.setZero();
        
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            
            // Strain-displacement matrix B
            // epsilon = [du/dx, dv/dy, dw/dz, dv/dz + dw/dy, du/dz + dw/dx, du/dy + dv/dx]
            // For simplicity, we use the simplified form
            for (int d = 0; d < dim; ++d) {
                B(d, i * vdim + d) = physGrad[d];  // diagonal terms
            }
            // Shear terms (simplified for now)
            if (dim >= 2) {
                B(3, i * vdim + 1) = physGrad[2]; B(3, i * vdim + 2) = physGrad[1];
                B(5, i * vdim + 0) = physGrad[1]; B(5, i * vdim + 1) = physGrad[0];
            }
            if (dim >= 3) {
                B(4, i * vdim + 0) = physGrad[2]; B(4, i * vdim + 2) = physGrad[0];
            }
        }
        
        // Constitutive matrix C (isotropic)
        Eigen::Matrix<Real, 6, 6> C;
        C.setZero();
        C(0,0) = C(1,1) = C(2,2) = lambda + 2*mu;
        C(0,1) = C(0,2) = C(1,0) = C(1,2) = C(2,0) = C(2,1) = lambda;
        C(3,3) = C(4,4) = C(5,5) = mu;
        
        // Simplified: use only the diagonal terms for now
        // K = B^T * C * B
        // This is a simplified implementation
        Eigen::MatrixXd gradMat(nd, dim);
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            for (int d = 0; d < dim; ++d) {
                gradMat(i, d) = physGrad[d];
            }
        }
        
        // Simplified stiffness matrix (ignoring Poisson effect for initial implementation)
        for (int a = 0; a < nd; ++a) {
            for (int b = 0; b < nd; ++b) {
                Real divTerm = 0;
                for (int d = 0; d < dim; ++d) {
                    divTerm += gradMat(a, d) * gradMat(b, d);
                }
                
                for (int i = 0; i < vdim; ++i) {
                    for (int j = 0; j < vdim; ++j) {
                        int row = a * vdim + i;
                        int col = b * vdim + j;
                        
                        if (i == j) {
                            // Diagonal: lambda * div * div + 2 * mu * grad_i * grad_i
                            elmat(row, col) += w * (lambda * gradMat(a, i) * gradMat(b, i) + 
                                                    2 * mu * gradMat(a, i) * gradMat(b, i));
                        } else {
                            // Off-diagonal: lambda * grad_i * grad_j + mu * grad_j * grad_i
                            elmat(row, col) += w * (lambda * gradMat(a, i) * gradMat(b, j) + 
                                                    mu * gradMat(a, j) * gradMat(b, i));
                        }
                    }
                }
            }
        }
    }
}

void ThermalLoadIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                   ElementTransform& trans,
                                                   Vector& elvec) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    const int vdim = 3;  // 3D displacement
    
    elvec.setZero(nd * vdim);
    
    if (!T_ || !alphaT_) return;
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        // Get material properties
        Real E = E_ ? E_->eval(trans) : 1.0;
        Real nu = nu_ ? nu_->eval(trans) : 0.3;
        Real alpha = alphaT_->eval(trans);
        
        // Get temperature
        Real T_val = T_->eval(trans.elementIndex(), xi);
        Real dT = T_val - Tref_;
        
        // Bulk modulus K = E / (3 * (1 - 2*nu))
        Real K = E / (3.0 * (1.0 - 2.0 * nu));
        
        // Thermal strain: epsilon_th = alpha * dT * I
        // Thermal stress: sigma_th = -3 * K * alpha * dT * I
        // Thermal load: f = div(sigma_th) = 0 (constant stress)
        // But we need to integrate: f = B^T * sigma_th
        
        Real thermalStress = 3.0 * K * alpha * dT;
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // Thermal load vector: thermal expansion creates initial strain
        // This contributes to the RHS as: f = integral of (B^T * C * epsilon_th)
        // For isotropic material: C * epsilon_th = 3 * K * alpha * dT * I
        
        for (int a = 0; a < nd; ++a) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[a].data(), physGrad.data());
            
            // Divergence of shape function
            Real divPhi = 0;
            for (int d = 0; d < dim; ++d) {
                divPhi += physGrad[d];
            }
            
            // Thermal load: -3 * K * alpha * dT * div(phi)
            for (int i = 0; i < vdim; ++i) {
                elvec(a * vdim + i) -= w * thermalStress * physGrad[i];
            }
        }
    }
}

}  // namespace mpfem
