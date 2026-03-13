#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/shape_function.hpp"
#include "fe/quadrature.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <memory>

namespace mpfem {

/**
 * @file element_transform.hpp
 * @brief Element transformation from reference to physical coordinates.
 * 
 * Design inspired by MFEM's ElementTransformation class.
 */

/**
 * @brief Base class for element transformations.
 * 
 * Provides mapping between reference element and physical element,
 * including Jacobian computation and coordinate transformation.
 * 
 * Element types:
 * - VOLUME: Interior elements (tetrahedra, hexahedra, triangles, quads)
 * - BOUNDARY: Boundary elements (triangles, quads in 3D; segments in 2D)
 * 
 * Thread safety: Each thread should have its own transform instance.
 */
class ElementTransform {
public:
    /// Element type enumeration
    enum ElementType {
        VOLUME = 1,     ///< Interior/volume element
        BOUNDARY = 2    ///< Boundary/facet element
    };
    
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    ElementTransform() = default;
    explicit ElementTransform(const Mesh* mesh, Index elemIdx, 
                              ElementType type = VOLUME);
    virtual ~ElementTransform() = default;
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    virtual void setMesh(const Mesh* mesh);
    virtual void setElement(Index elemIdx);
    
    /// Set integration point (invalidates cached data)
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Real* xi);
    
    /// Reset evaluation state (force recompute on next access)
    void reset() { evalState_ = 0; }
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension of reference element
    int dim() const { return dim_; }
    
    /// Get spatial dimension of physical space
    int spaceDim() const { return spaceDim_; }
    
    /// Get element index
    Index elementIndex() const { return elemIdx_; }
    
    /// Get element type
    ElementType elementType() const { return elemType_; }
    
    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }
    
    /// Get the element attribute (domain/boundary ID)
    Index attribute() const;
    
    /// Get geometric order
    int geometricOrder() const { return geomOrder_; }
    
    /// Check if element is curved (geometric order >= 2)
    bool isCurved() const { return geomOrder_ >= 2; }
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    /// Transform reference coordinates to physical coordinates
    virtual void transform(const Real* xi, Real* x) const;
    void transform(const Real* xi, Vector3& x) const;
    void transform(const IntegrationPoint& ip, Vector3& x) const;
    
    // -------------------------------------------------------------------------
    // Jacobian and related quantities
    // -------------------------------------------------------------------------
    
    /// Get the Jacobian matrix (spaceDim x dim)
    const Matrix& jacobian() const;
    
    /// Get the inverse Jacobian matrix
    const Matrix& invJacobian() const;
    
    /// Get the transpose of inverse Jacobian (J^{-T})
    const Matrix& invJacobianT() const;
    
    /// Get the Jacobian determinant
    Real detJ() const;
    
    /// Get the weight = |det(J)| for volume elements, or sqrt(det(J^T J)) for surface elements
    virtual Real weight() const;
    
    /// Get the adjugate of the Jacobian
    const Matrix& adjJacobian() const;
    
    // -------------------------------------------------------------------------
    // Gradient transformation
    // -------------------------------------------------------------------------
    
    /// Transform gradient from reference to physical coordinates
    void transformGradient(const Real* refGrad, Real* physGrad) const;
    void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;
    
    // -------------------------------------------------------------------------
    // Element nodes
    // -------------------------------------------------------------------------
    
    /// Get geometric node coordinates
    const std::vector<Vector3>& nodes() const { return nodes_; }
    
    /// Get number of geometric nodes
    int numNodes() const { return static_cast<int>(nodes_.size()); }
    
    /// Alias for compatibility
    const std::vector<Vector3>& vertices() const { return nodes_; }
    int numVertices() const { return numNodes(); }
    
    /// Get the currently set integration point
    const IntegrationPoint& integrationPoint() const { return ip_; }
    
protected:
    // -------------------------------------------------------------------------
    // Evaluation state management
    // -------------------------------------------------------------------------
    enum EvalMask {
        JACOBIAN_MASK = 1,
        WEIGHT_MASK   = 2,
        ADJUGATE_MASK = 4,
        INVERSE_MASK  = 8,
        INV_JACOBIAN_T_MASK = 16
    };
    
    virtual void computeGeometryInfo();
    void evalJacobian() const;
    void evalWeight() const;
    void evalAdjugate() const;
    void evalInverse() const;
    void evalInvJacobianT() const;
    
    void initGeometricShapeFunction();
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    const Mesh* mesh_ = nullptr;
    Index elemIdx_ = InvalidIndex;
    ElementType elemType_ = VOLUME;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;        ///< Reference dimension
    int spaceDim_ = 0;   ///< Physical space dimension
    int geomOrder_ = 1;
    
    std::unique_ptr<ShapeFunction> shapeFunc_; 
    std::vector<Vector3> nodes_;
    std::vector<Index> nodeIndices_;
    
    IntegrationPoint ip_;
    
    // Mutable for lazy evaluation
    mutable Matrix jacobian_;       ///< spaceDim x dim
    mutable Matrix invJacobian_;    ///< dim x spaceDim
    mutable Matrix invJacobianT_;   ///< spaceDim x dim
    mutable Matrix adjJacobian_;    ///< spaceDim x dim
    mutable Real detJ_ = 0.0;
    mutable Real weight_ = 0.0;
    mutable int evalState_ = 0;
    mutable ShapeValues shapeValues_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline ElementTransform::ElementTransform(const Mesh* mesh, Index elemIdx, ElementType type)
    : mesh_(mesh), elemIdx_(elemIdx), elemType_(type) {
    computeGeometryInfo();
}

inline void ElementTransform::setIntegrationPoint(const IntegrationPoint& ip) {
    ip_ = ip;
    evalState_ = 0;
    // Don't compute shape values here - delay until needed in evalJacobian()
    // This avoids memory allocation when setting integration points
    shapeValues_ = ShapeValues();  // Clear cached values
}

inline void ElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    evalState_ = 0;
    // Don't compute shape values here - delay until needed in evalJacobian()
    shapeValues_ = ShapeValues();  // Clear cached values
}

inline void ElementTransform::transform(const Real* xi, Vector3& x) const {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
}

inline void ElementTransform::transform(const IntegrationPoint& ip, Vector3& x) const {
    transform(&ip.xi, x);
}

inline void ElementTransform::transformGradient(const Vector3& refGrad, Vector3& physGrad) const {
    Real rg[3] = {refGrad.x(), refGrad.y(), refGrad.z()};
    Real pg[3];
    transformGradient(rg, pg);
    physGrad = Vector3(pg[0], pg[1], pg[2]);
}

inline const Matrix& ElementTransform::jacobian() const {
    evalJacobian();
    return jacobian_;
}

inline const Matrix& ElementTransform::invJacobian() const {
    evalInverse();
    return invJacobian_;
}

inline const Matrix& ElementTransform::invJacobianT() const {
    evalInvJacobianT();
    return invJacobianT_;
}

inline Real ElementTransform::detJ() const {
    evalWeight();
    return detJ_;
}

inline const Matrix& ElementTransform::adjJacobian() const {
    evalAdjugate();
    return adjJacobian_;
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_TRANSFORM_HPP
