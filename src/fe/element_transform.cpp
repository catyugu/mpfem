#include "fe/element_transform.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// ElementTransform Implementation
// =============================================================================

Index ElementTransform::elementAttribute() const {
    if (!mesh_) return 0;
    
    if (elemIdx_ < mesh_->numElements()) {
        return mesh_->element(elemIdx_).attribute();
    }
    return 0;
}

void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (elemIdx_ < mesh_->numElements()) {
        elem = &mesh_->element(elemIdx_);
    }
    
    if (!elem) return;
    
    geometry_ = elem->geometry();
    dim_ = geom::dim(geometry_);
    
    // Get vertex coordinates
    vertexIndices_ = elem->vertices();
    vertices_.resize(vertexIndices_.size());
    for (size_t i = 0; i < vertexIndices_.size(); ++i) {
        vertices_[i] = mesh_->vertex(vertexIndices_[i]).toVector();
    }
    
    // Pre-allocate matrices
    jacobian_.setZero(dim_, dim_);
    invJacobian_.setZero(dim_, dim_);
    adjJacobian_.setZero(dim_, dim_);
}

// =============================================================================
// Jacobian evaluation
// =============================================================================

void ElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    // For linear elements, Jacobian is constant
    // J_ij = dx_i / dxi_j
    
    switch (geometry_) {
        case Geometry::Segment: {
            // dphi/dxi = (-0.5, 0.5)
            // J = 0.5 * (x1 - x0)
            jacobian_(0, 0) = 0.5 * (vertices_[1][0] - vertices_[0][0]);
            break;
        }
        
        case Geometry::Triangle: {
            // Shape function gradients in reference coords (constant for linear)
            // phi0 = 1 - xi - eta, phi1 = xi, phi2 = eta
            // grad(phi0) = (-1, -1), grad(phi1) = (1, 0), grad(phi2) = (0, 1)
            // J = [x1-x0, x2-x0; y1-y0, y2-y0]
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Square: {
            // Bilinear mapping: phi = 0.25 * (1 +/- xi) * (1 +/- eta)
            Real xi = ip_.xi, eta = ip_.eta;
            
            // dphi/dxi
            Real dphi0_dxi = -0.25 * (1.0 - eta);
            Real dphi1_dxi =  0.25 * (1.0 - eta);
            Real dphi2_dxi =  0.25 * (1.0 + eta);
            Real dphi3_dxi = -0.25 * (1.0 + eta);
            
            // dphi/deta
            Real dphi0_deta = -0.25 * (1.0 - xi);
            Real dphi1_deta = -0.25 * (1.0 + xi);
            Real dphi2_deta =  0.25 * (1.0 + xi);
            Real dphi3_deta =  0.25 * (1.0 - xi);
            
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = dphi0_dxi * vertices_[0][i] + dphi1_dxi * vertices_[1][i] +
                                  dphi2_dxi * vertices_[2][i] + dphi3_dxi * vertices_[3][i];
                jacobian_(i, 1) = dphi0_deta * vertices_[0][i] + dphi1_deta * vertices_[1][i] +
                                  dphi2_deta * vertices_[2][i] + dphi3_deta * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Tetrahedron: {
            // Linear tetrahedron: constant Jacobian
            // phi0 = 1 - xi - eta - zeta, phi1 = xi, phi2 = eta, phi3 = zeta
            // J_ij = x_{j+1,i} - x_{0,i} for j = 0,1,2
            for (int i = 0; i < 3; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
                jacobian_(i, 2) = vertices_[3][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Cube: {
            // Trilinear mapping for hexahedron
            Real xi = ip_.xi, eta = ip_.eta, zeta = ip_.zeta;
            
            // Derivatives of shape functions
            // phi_k = 0.125 * (1 +/- xi) * (1 +/- eta) * (1 +/- zeta)
            Real dphi[8][3];  // dphi/dxi, dphi/deta, dphi/dzeta
            
            // Vertex ordering: (following geometry.hpp face_table::Cube)
            // 0: (-1,-1,-1), 1: (1,-1,-1), 2: (1,1,-1), 3: (-1,1,-1)
            // 4: (-1,-1,1),  5: (1,-1,1),  6: (1,1,1),  7: (-1,1,1)
            
            Real t1 = 0.125 * (1.0 - eta) * (1.0 - zeta);
            Real t2 = 0.125 * (1.0 + eta) * (1.0 - zeta);
            Real t3 = 0.125 * (1.0 - eta) * (1.0 + zeta);
            Real t4 = 0.125 * (1.0 + eta) * (1.0 + zeta);
            
            Real s1 = 0.125 * (1.0 - xi) * (1.0 - zeta);
            Real s2 = 0.125 * (1.0 + xi) * (1.0 - zeta);
            Real s3 = 0.125 * (1.0 - xi) * (1.0 + zeta);
            Real s4 = 0.125 * (1.0 + xi) * (1.0 + zeta);
            
            Real r1 = 0.125 * (1.0 - xi) * (1.0 - eta);
            Real r2 = 0.125 * (1.0 + xi) * (1.0 - eta);
            Real r3 = 0.125 * (1.0 + xi) * (1.0 + eta);
            Real r4 = 0.125 * (1.0 - xi) * (1.0 + eta);
            
            // dphi/dxi
            dphi[0][0] = -t1; dphi[1][0] = t1;
            dphi[2][0] = t2;  dphi[3][0] = -t2;
            dphi[4][0] = -t3; dphi[5][0] = t3;
            dphi[6][0] = t4;  dphi[7][0] = -t4;
            
            // dphi/deta
            dphi[0][1] = -s1; dphi[1][1] = -s2;
            dphi[2][1] = s2;  dphi[3][1] = s1;
            dphi[4][1] = -s3; dphi[5][1] = -s4;
            dphi[6][1] = s4;  dphi[7][1] = s3;
            
            // dphi/dzeta
            dphi[0][2] = -r1; dphi[1][2] = -r2;
            dphi[2][2] = -r3; dphi[3][2] = -r4;
            dphi[4][2] = r1;  dphi[5][2] = r2;
            dphi[6][2] = r3;  dphi[7][2] = r4;
            
            // J = sum_k (x_k * grad(phi_k))
            for (int i = 0; i < 3; ++i) {
                jacobian_(i, 0) = 0;
                jacobian_(i, 1) = 0;
                jacobian_(i, 2) = 0;
                for (int k = 0; k < 8; ++k) {
                    jacobian_(i, 0) += vertices_[k][i] * dphi[k][0];
                    jacobian_(i, 1) += vertices_[k][i] * dphi[k][1];
                    jacobian_(i, 2) += vertices_[k][i] * dphi[k][2];
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    evalState_ |= JACOBIAN_MASK;
}

void ElementTransform::evalWeight() const {
    if (evalState_ & WEIGHT_MASK) return;
    
    // Ensure Jacobian is computed
    evalJacobian();
    
    // Compute determinant based on dimension
    if (dim_ == 1) {
        detJ_ = jacobian_(0, 0);
    } else if (dim_ == 2) {
        detJ_ = jacobian_(0, 0) * jacobian_(1, 1) - jacobian_(0, 1) * jacobian_(1, 0);
    } else if (dim_ == 3) {
        detJ_ = jacobian_(0, 0) * (jacobian_(1, 1) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 1))
              - jacobian_(0, 1) * (jacobian_(1, 0) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 0))
              + jacobian_(0, 2) * (jacobian_(1, 0) * jacobian_(2, 1) - jacobian_(1, 1) * jacobian_(2, 0));
    }
    
    weight_ = std::abs(detJ_);
    evalState_ |= WEIGHT_MASK;
}

void ElementTransform::evalAdjugate() const {
    if (evalState_ & ADJUGATE_MASK) return;
    
    // Ensure Jacobian and weight are computed
    evalWeight();
    
    if (dim_ == 1) {
        adjJacobian_(0, 0) = 1.0;
    } else if (dim_ == 2) {
        adjJacobian_(0, 0) = jacobian_(1, 1);
        adjJacobian_(0, 1) = -jacobian_(0, 1);
        adjJacobian_(1, 0) = -jacobian_(1, 0);
        adjJacobian_(1, 1) = jacobian_(0, 0);
    } else if (dim_ == 3) {
        // adj(J) = det(J) * J^{-1}, computed after inverse
        evalInverse();
        adjJacobian_ = detJ_ * invJacobian_;
    }
    
    evalState_ |= ADJUGATE_MASK;
}

void ElementTransform::evalInverse() const {
    if (evalState_ & INVERSE_MASK) return;
    
    // Ensure Jacobian is computed
    evalJacobian();
    
    if (dim_ == 1) {
        invJacobian_(0, 0) = 1.0 / jacobian_(0, 0);
    } else if (dim_ == 2) {
        evalWeight();
        Real invDet = 1.0 / detJ_;
        invJacobian_(0, 0) = jacobian_(1, 1) * invDet;
        invJacobian_(0, 1) = -jacobian_(0, 1) * invDet;
        invJacobian_(1, 0) = -jacobian_(1, 0) * invDet;
        invJacobian_(1, 1) = jacobian_(0, 0) * invDet;
    } else if (dim_ == 3) {
        invJacobian_ = jacobian_.inverse();
    }
    
    evalState_ |= INVERSE_MASK;
}

void ElementTransform::evalInvJacobianT() const {
    if (evalState_ & INV_JACOBIAN_T_MASK) return;
    
    // Ensure inverse Jacobian is computed
    evalInverse();
    
    // invJacobianT_ = invJacobian_^T
    invJacobianT_ = invJacobian_.transpose();
    
    evalState_ |= INV_JACOBIAN_T_MASK;
}

// =============================================================================
// Transform implementations
// =============================================================================

void ElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    switch (geometry_) {
        case Geometry::Segment: {
            // Reference: [-1, 1]
            // phi0 = 0.5*(1-xi), phi1 = 0.5*(1+xi)
            Real phi0 = 0.5 * (1.0 - xi[0]);
            Real phi1 = 0.5 * (1.0 + xi[0]);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi0 * vertices_[0][i] + phi1 * vertices_[1][i];
            }
            break;
        }
        
        case Geometry::Triangle: {
            // Reference: (0,0), (1,0), (0,1)
            // Barycentric: (1-xi-eta, xi, eta)
            Real l1 = 1.0 - xi[0] - xi[1];
            Real l2 = xi[0];
            Real l3 = xi[1];
            for (int i = 0; i < 3; ++i) {
                x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] + l3 * vertices_[2][i];
            }
            break;
        }
        
        case Geometry::Square: {
            // Reference: [-1,1] x [-1,1]
            Real xi1 = xi[0], xi2 = xi[1];
            Real phi[4];
            phi[0] = 0.25 * (1.0 - xi1) * (1.0 - xi2);
            phi[1] = 0.25 * (1.0 + xi1) * (1.0 - xi2);
            phi[2] = 0.25 * (1.0 + xi1) * (1.0 + xi2);
            phi[3] = 0.25 * (1.0 - xi1) * (1.0 + xi2);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi[0] * vertices_[0][i] + phi[1] * vertices_[1][i] +
                       phi[2] * vertices_[2][i] + phi[3] * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Tetrahedron: {
            // Reference: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            // Barycentric: (1-xi-eta-zeta, xi, eta, zeta)
            Real l1 = 1.0 - xi[0] - xi[1] - xi[2];
            Real l2 = xi[0];
            Real l3 = xi[1];
            Real l4 = xi[2];
            for (int i = 0; i < 3; ++i) {
                x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] +
                       l3 * vertices_[2][i] + l4 * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Cube: {
            // Reference: [-1,1]^3
            Real xi1 = xi[0], xi2 = xi[1], xi3 = xi[2];
            Real phi[8];
            phi[0] = 0.125 * (1.0 - xi1) * (1.0 - xi2) * (1.0 - xi3);
            phi[1] = 0.125 * (1.0 + xi1) * (1.0 - xi2) * (1.0 - xi3);
            phi[2] = 0.125 * (1.0 + xi1) * (1.0 + xi2) * (1.0 - xi3);
            phi[3] = 0.125 * (1.0 - xi1) * (1.0 + xi2) * (1.0 - xi3);
            phi[4] = 0.125 * (1.0 - xi1) * (1.0 - xi2) * (1.0 + xi3);
            phi[5] = 0.125 * (1.0 + xi1) * (1.0 - xi2) * (1.0 + xi3);
            phi[6] = 0.125 * (1.0 + xi1) * (1.0 + xi2) * (1.0 + xi3);
            phi[7] = 0.125 * (1.0 - xi1) * (1.0 + xi2) * (1.0 + xi3);
            for (int i = 0; i < 3; ++i) {
                x[i] = 0;
                for (int j = 0; j < 8; ++j) {
                    x[i] += phi[j] * vertices_[j][i];
                }
            }
            break;
        }
        
        default:
            break;
    }
}

void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    // grad_x(phi) = J^{-T} * grad_xi(phi)
    // Using cached invJacobianT_ for efficiency
    
    const Matrix& invJT = invJacobianT();
    
    for (int i = 0; i < dim_; ++i) {
        physGrad[i] = 0.0;
        for (int j = 0; j < dim_; ++j) {
            physGrad[i] += invJT(i, j) * refGrad[j];
        }
    }
}

}  // namespace mpfem
