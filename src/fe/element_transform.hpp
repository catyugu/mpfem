#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>

namespace mpfem {

/**
 * @brief Element transformation from reference to physical coordinates.
 * 
 * Provides mapping between reference element and physical element,
 * including Jacobian computation and coordinate transformation.
 */
class ElementTransform {
public:
    /// Default constructor
    ElementTransform() = default;
    
    /// Construct with mesh and element index
    ElementTransform(const Mesh* mesh, Index elemIdx, bool isBoundary = false);
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    /// Set the mesh
    void setMesh(const Mesh* mesh) { mesh_ = mesh; }
    
    /// Set the element index
    void setElement(Index elemIdx, bool isBoundary = false);
    
    /// Set integration point (triggers Jacobian computation)
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Real* xi);
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension
    int dim() const { return dim_; }
    
    /// Get element index
    Index elementIndex() const { return elemIdx_; }
    
    /// Is this a boundary element?
    bool isBoundary() const { return isBoundary_; }
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    /// Transform reference coordinates to physical coordinates
    void transform(const Real* xi, Real* x) const;
    void transform(const Real* xi, Vector3& x) const;
    
    // Note: Use transform(const Real* xi, Vector3& x) to get physical coordinates
    // IntegrationPoint does not store physical coordinates by default
    
    // -------------------------------------------------------------------------
    // Jacobian
    // -------------------------------------------------------------------------
    
    /// Get the Jacobian matrix (dim x dim)
    /// J_ij = dx_i / dξ_j
    const Matrix& jacobian() const { return jacobian_; }
    
    /// Get the inverse Jacobian matrix
    const Matrix& invJacobian() const { return invJacobian_; }
    
    /// Get the Jacobian determinant
    Real detJ() const { return detJ_; }
    
    /// Get the weight = |det(J)|
    Real weight() const { return weight_; }
    
    /// Get the adjugate of the Jacobian (for gradient transformation)
    const Matrix& adjJacobian() const { return adjJacobian_; }
    
    // -------------------------------------------------------------------------
    // Gradient transformation
    // -------------------------------------------------------------------------
    
    /**
     * @brief Transform gradient from reference to physical coordinates.
     * 
     * Given gradient in reference coordinates: ∇_ξ φ
     * Returns gradient in physical coordinates: ∇_x φ
     * 
     * ∇_x φ = J^{-T} ∇_ξ φ
     * 
     * @param refGrad Gradient in reference coordinates (dim components)
     * @param physGrad Gradient in physical coordinates (output)
     */
    void transformGradient(const Real* refGrad, Real* physGrad) const;
    void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;
    
    // -------------------------------------------------------------------------
    // Element vertices (convenience)
    // -------------------------------------------------------------------------
    
    /// Get vertex coordinates
    const std::vector<Vector3>& vertices() const { return vertices_; }
    
    /// Get number of vertices
    int numVertices() const { return static_cast<int>(vertices_.size()); }
    
private:
    void computeJacobian();
    void computeGeometryInfo();
    
    const Mesh* mesh_ = nullptr;
    Index elemIdx_ = 0;
    bool isBoundary_ = false;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;
    std::vector<Vector3> vertices_;
    std::vector<Index> vertexIndices_;
    
    IntegrationPoint ip_;
    Matrix jacobian_;
    Matrix invJacobian_;
    Matrix adjJacobian_;
    Real detJ_ = 0.0;
    Real weight_ = 0.0;
    
    // State flags
    mutable bool jacobianComputed_ = false;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline ElementTransform::ElementTransform(const Mesh* mesh, Index elemIdx, bool isBoundary)
    : mesh_(mesh), elemIdx_(elemIdx), isBoundary_(isBoundary) {
    computeGeometryInfo();
}

inline void ElementTransform::setElement(Index elemIdx, bool isBoundary) {
    elemIdx_ = elemIdx;
    isBoundary_ = isBoundary;
    jacobianComputed_ = false;
    computeGeometryInfo();
}

inline void ElementTransform::setIntegrationPoint(const IntegrationPoint& ip) {
    ip_ = ip;
    jacobianComputed_ = false;
}

inline void ElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    jacobianComputed_ = false;
}

inline void ElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    // Evaluate using barycentric or tensor product coordinates
    switch (geometry_) {
        case Geometry::Triangle: {
            // Reference coords: (xi, eta), barycentric: (1-xi-eta, xi, eta)
            Real l1 = 1.0 - xi[0] - xi[1];
            Real l2 = xi[0];
            Real l3 = xi[1];
            for (int i = 0; i < 3; ++i) {
                x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] + l3 * vertices_[2][i];
            }
            break;
        }
        case Geometry::Square: {
            // Tensor product: vertices are (-1,-1), (1,-1), (1,1), (-1,1)
            Real xi1 = xi[0], xi2 = xi[1];
            Real phi[4];
            phi[0] = 0.25 * (1 - xi1) * (1 - xi2);
            phi[1] = 0.25 * (1 + xi1) * (1 - xi2);
            phi[2] = 0.25 * (1 + xi1) * (1 + xi2);
            phi[3] = 0.25 * (1 - xi1) * (1 + xi2);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi[0] * vertices_[0][i] + phi[1] * vertices_[1][i] +
                       phi[2] * vertices_[2][i] + phi[3] * vertices_[3][i];
            }
            break;
        }
        case Geometry::Tetrahedron: {
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
            // Tensor product 3D
            Real xi1 = xi[0], xi2 = xi[1], xi3 = xi[2];
            Real phi[8];
            phi[0] = 0.125 * (1 - xi1) * (1 - xi2) * (1 - xi3);
            phi[1] = 0.125 * (1 + xi1) * (1 - xi2) * (1 - xi3);
            phi[2] = 0.125 * (1 + xi1) * (1 + xi2) * (1 - xi3);
            phi[3] = 0.125 * (1 - xi1) * (1 + xi2) * (1 - xi3);
            phi[4] = 0.125 * (1 - xi1) * (1 - xi2) * (1 + xi3);
            phi[5] = 0.125 * (1 + xi1) * (1 - xi2) * (1 + xi3);
            phi[6] = 0.125 * (1 + xi1) * (1 + xi2) * (1 + xi3);
            phi[7] = 0.125 * (1 - xi1) * (1 + xi2) * (1 + xi3);
            for (int i = 0; i < 3; ++i) {
                x[i] = 0;
                for (int j = 0; j < 8; ++j) {
                    x[i] += phi[j] * vertices_[j][i];
                }
            }
            break;
        }
        case Geometry::Segment: {
            Real phi0 = 0.5 * (1 - xi[0]);
            Real phi1 = 0.5 * (1 + xi[0]);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi0 * vertices_[0][i] + phi1 * vertices_[1][i];
            }
            break;
        }
        default:
            break;
    }
}

inline void ElementTransform::transform(const Real* xi, Vector3& x) const {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
}

inline void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (isBoundary_) {
        if (elemIdx_ < mesh_->numBdrElements()) {
            elem = &mesh_->bdrElement(elemIdx_);
        }
    } else {
        if (elemIdx_ < mesh_->numElements()) {
            elem = &mesh_->element(elemIdx_);
        }
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
    
    // Initialize Jacobian matrix
    jacobian_.setZero(dim_, dim_);
    invJacobian_.setZero(dim_, dim_);
    adjJacobian_.setZero(dim_, dim_);
}

inline void ElementTransform::computeJacobian() {
    if (jacobianComputed_) return;
    
    // Compute Jacobian using shape function gradients
    // J_ij = d(x_i)/d(ξ_j) = Σ_k (dφ_k/dξ_j) * x_k_i
    
    switch (geometry_) {
        case Geometry::Segment: {
            // dφ/dξ = (-0.5, 0.5)
            jacobian_(0, 0) = 0.5 * (vertices_[1][0] - vertices_[0][0]);
            detJ_ = jacobian_(0, 0);
            weight_ = std::abs(detJ_);
            invJacobian_(0, 0) = 1.0 / detJ_;
            break;
        }
        case Geometry::Triangle: {
            // Shape function gradients (constant for linear triangle)
            // φ0 = 1 - ξ - η, φ1 = ξ, φ2 = η
            // ∇φ0 = (-1, -1), ∇φ1 = (1, 0), ∇φ2 = (0, 1)
            // J = [x1-x0, y1-y0; x2-x0, y2-y0]
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
            }
            detJ_ = jacobian_(0, 0) * jacobian_(1, 1) - jacobian_(0, 1) * jacobian_(1, 0);
            weight_ = std::abs(detJ_);
            invJacobian_(0, 0) = jacobian_(1, 1) / detJ_;
            invJacobian_(0, 1) = -jacobian_(0, 1) / detJ_;
            invJacobian_(1, 0) = -jacobian_(1, 0) / detJ_;
            invJacobian_(1, 1) = jacobian_(0, 0) / detJ_;
            break;
        }
        case Geometry::Square: {
            // φ = 0.25 * (1 ± ξ) * (1 ± η)
            Real xi = ip_.xi, eta = ip_.eta;
            
            // dφ/dξ
            Real dphi0_dxi = -0.25 * (1 - eta);
            Real dphi1_dxi = 0.25 * (1 - eta);
            Real dphi2_dxi = 0.25 * (1 + eta);
            Real dphi3_dxi = -0.25 * (1 + eta);
            
            // dφ/dη
            Real dphi0_deta = -0.25 * (1 - xi);
            Real dphi1_deta = -0.25 * (1 + xi);
            Real dphi2_deta = 0.25 * (1 + xi);
            Real dphi3_deta = 0.25 * (1 - xi);
            
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = dphi0_dxi * vertices_[0][i] + dphi1_dxi * vertices_[1][i] +
                                  dphi2_dxi * vertices_[2][i] + dphi3_dxi * vertices_[3][i];
                jacobian_(i, 1) = dphi0_deta * vertices_[0][i] + dphi1_deta * vertices_[1][i] +
                                  dphi2_deta * vertices_[2][i] + dphi3_deta * vertices_[3][i];
            }
            
            detJ_ = jacobian_(0, 0) * jacobian_(1, 1) - jacobian_(0, 1) * jacobian_(1, 0);
            weight_ = std::abs(detJ_);
            invJacobian_(0, 0) = jacobian_(1, 1) / detJ_;
            invJacobian_(0, 1) = -jacobian_(0, 1) / detJ_;
            invJacobian_(1, 0) = -jacobian_(1, 0) / detJ_;
            invJacobian_(1, 1) = jacobian_(0, 0) / detJ_;
            break;
        }
        case Geometry::Tetrahedron: {
            // Shape function gradients (constant for linear tet)
            // φ0 = 1 - ξ - η - ζ, φ1 = ξ, φ2 = η, φ3 = ζ
            // J_ij = x_{i,j+1} - x_{i,0} for j = 0,1,2
            for (int i = 0; i < 3; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
                jacobian_(i, 2) = vertices_[3][i] - vertices_[0][i];
            }
            
            // 3x3 determinant
            detJ_ = jacobian_(0, 0) * (jacobian_(1, 1) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 1))
                  - jacobian_(0, 1) * (jacobian_(1, 0) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 0))
                  + jacobian_(0, 2) * (jacobian_(1, 0) * jacobian_(2, 1) - jacobian_(1, 1) * jacobian_(2, 0));
            weight_ = std::abs(detJ_);
            
            // Inverse
            invJacobian_ = jacobian_.inverse();
            break;
        }
        case Geometry::Cube: {
            // Tensor product shape functions
            Real xi = ip_.xi, eta = ip_.eta, zeta = ip_.zeta;
            
            // Derivatives at quadrature point
            Real dphi[8][3];  // dφ/dξ, dφ/dη, dφ/dζ
            // ...
            // TODO: Implement for hex
            break;
        }
        default:
            break;
    }
    
    // Compute adjugate
    if (dim_ == 2) {
        adjJacobian_(0, 0) = jacobian_(1, 1);
        adjJacobian_(0, 1) = -jacobian_(0, 1);
        adjJacobian_(1, 0) = -jacobian_(1, 0);
        adjJacobian_(1, 1) = jacobian_(0, 0);
    } else if (dim_ == 3) {
        adjJacobian_ = detJ_ * invJacobian_;
    }
    
    jacobianComputed_ = true;
}

inline void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    // Need to compute Jacobian first
    const_cast<ElementTransform*>(this)->computeJacobian();
    
    // ∇_x φ = J^{-T} ∇_ξ φ
    for (int i = 0; i < dim_; ++i) {
        physGrad[i] = 0.0;
        for (int j = 0; j < dim_; ++j) {
            physGrad[i] += invJacobian_(j, i) * refGrad[j];
        }
    }
}

inline void ElementTransform::transformGradient(const Vector3& refGrad, Vector3& physGrad) const {
    Real rg[3] = {refGrad.x(), refGrad.y(), refGrad.z()};
    Real pg[3];
    transformGradient(rg, pg);
    physGrad = Vector3(pg[0], pg[1], pg[2]);
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_TRANSFORM_HPP
