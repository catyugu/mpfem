#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/shape_function.hpp"
#include "fe/quadrature.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>

namespace mpfem {

/**
 * @brief Element transformation from reference to physical coordinates.
 * 
 * Provides mapping between reference element and physical element,
 * including Jacobian computation and coordinate transformation.
 * 
 * This class is designed for volume elements (tetrahedra, hexahedra in 3D;
 * triangles, quadrilaterals in 2D). For boundary elements, use
 * FacetElementTransform instead.
 * 
 * **Geometric Order vs Field Order**:
 * - Geometric order (geomOrder_): Determines the accuracy of coordinate mapping.
 *   Derived from Element::order(). For curved (second-order) meshes, this is 2.
 * - Field order: Determined by the FE space, not stored here.
 * 
 * The geometric shape function (geomShapeFunc_) is used for coordinate transformation
 * and Jacobian computation. For linear elements, it's a linear Lagrange shape function.
 * For quadratic elements, it's a quadratic Lagrange shape function that accounts for
 * edge midpoints, enabling proper curved element representation.
 * 
 * Thread safety: Each thread should have its own ElementTransform instance.
 * The cached Jacobian data uses mutable for lazy evaluation but is not
 * thread-safe when shared between threads.
 * 
 * Design inspired by MFEM's ElementTransformation class.
 */
class ElementTransform {
public:
    /// Default constructor
    ElementTransform() = default;
    
    /// Construct with mesh and element index
    explicit ElementTransform(const Mesh* mesh, Index elemIdx);
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    /// Set the mesh
    void setMesh(const Mesh* mesh) { mesh_ = mesh; }
    
    /// Set the element index
    void setElement(Index elemIdx);
    
    /// Set integration point (invalidates cached Jacobian data)
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Real* xi);
    
    /// Reset evaluation state (force recompute on next access)
    void reset() { evalState_ = 0; }
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension
    int dim() const { return dim_; }
    
    /// Get element index
    Index elementIndex() const { return elemIdx_; }
    
    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }
    
    /// Get the element attribute (domain ID)
    Index elementAttribute() const;
    
    /// Get geometric order (from Element::order())
    /// This determines the accuracy of coordinate mapping.
    /// For curved elements, this is >= 2.
    int geometricOrder() const { return geomOrder_; }
    
    /// Get geometric shape function (for coordinate mapping)
    /// This is used for transform() and Jacobian computation.
    const ShapeFunction* geometricShapeFunction() const { 
        return geomShapeFunc_.get(); 
    }
    
    /// Get number of geometric nodes (nodes used for coordinate mapping)
    int numGeometricNodes() const { 
        return geomShapeFunc_ ? geomShapeFunc_->numDofs() : 0; 
    }
    
    /// Check if element is curved (geometric order >= 2)
    bool isCurved() const { return geomOrder_ >= 2; }
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    /// Transform reference coordinates to physical coordinates
    void transform(const Real* xi, Real* x) const;
    void transform(const Real* xi, Vector3& x) const;
    
    /// Transform integration point to physical coordinates
    void transform(const IntegrationPoint& ip, Vector3& x) const {
        transform(&ip.xi, x);
    }
    
    // -------------------------------------------------------------------------
    // Jacobian and related quantities
    // -------------------------------------------------------------------------
    
    /// Get the Jacobian matrix (dim x dim)
    /// J_ij = dx_i / dxi_j
    const Matrix& jacobian() const;
    
    /// Get the inverse Jacobian matrix
    const Matrix& invJacobian() const;
    
    /// Get the transpose of inverse Jacobian matrix (J^{-T}) for gradient transformation
    const Matrix& invJacobianT() const;
    
    /// Get the Jacobian determinant
    Real detJ() const;
    
    /// Get the weight = |det(J)|
    Real weight() const;
    
    /// Get the adjugate of the Jacobian (for gradient transformation)
    const Matrix& adjJacobian() const;
    
    // -------------------------------------------------------------------------
    // Gradient transformation
    // -------------------------------------------------------------------------
    
    /**
     * @brief Transform gradient from reference to physical coordinates.
     * 
     * Given gradient in reference coordinates: grad_xi(phi)
     * Returns gradient in physical coordinates: grad_x(phi)
     * 
     * grad_x(phi) = J^{-T} * grad_xi(phi)
     */
    void transformGradient(const Real* refGrad, Real* physGrad) const;
    void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;
    
    // -------------------------------------------------------------------------
    // Element vertices (convenience)
    // -------------------------------------------------------------------------
    
    /// Get geometric node coordinates
    const std::vector<Vector3>& geomNodes() const { return geomNodes_; }
    
    /// Get number of geometric nodes
    int numGeomNodes() const { return static_cast<int>(geomNodes_.size()); }
    
    /// Get vertex coordinates (alias for geometric nodes)
    const std::vector<Vector3>& vertices() const { return geomNodes_; }
    
    /// Get number of vertices (alias for numGeomNodes)
    int numVertices() const { return numGeomNodes(); }
    
    /// Get the currently set integration point
    const IntegrationPoint& integrationPoint() const { return ip_; }
    
private:
    // -------------------------------------------------------------------------
    // Evaluation state management (MFEM-style)
    // -------------------------------------------------------------------------
    enum EvalMask {
        JACOBIAN_MASK    = 1,
        WEIGHT_MASK      = 2,
        ADJUGATE_MASK    = 4,
        INVERSE_MASK     = 8,
        INV_JACOBIAN_T_MASK = 16
    };
    
    void computeGeometryInfo();
    void evalJacobian() const;
    void evalWeight() const;
    void evalAdjugate() const;
    void evalInverse() const;
    void evalInvJacobianT() const;
    
    /// Initialize geometric shape function based on geometry type and order
    void initGeometricShapeFunction();
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    const Mesh* mesh_ = nullptr;
    Index elemIdx_ = 0;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;
    
    /// Geometric order (from Element::order())
    /// This determines the coordinate mapping accuracy.
    int geomOrder_ = 1;
    
    /// Geometric shape function for coordinate transformation.
    /// For linear elements: order 1 Lagrange shape function.
    /// For curved elements: order 2+ Lagrange shape function.
    std::unique_ptr<ShapeFunction> geomShapeFunc_;
    
    /// Geometric node coordinates (all nodes used for coordinate mapping).
    /// For linear elements: corner vertices only.
    /// For curved elements: corner vertices + edge midpoints + ...
    std::vector<Vector3> geomNodes_;
    
    /// Geometric node indices in the mesh vertex array.
    /// For curved elements, this includes edge midpoint node indices.
    std::vector<Index> geomNodeIndices_;
    
    IntegrationPoint ip_;
    
    // Mutable for lazy evaluation
    mutable Matrix jacobian_;
    mutable Matrix invJacobian_;
    mutable Matrix invJacobianT_;  // J^{-T} for gradient transformation
    mutable Matrix adjJacobian_;
    mutable Real detJ_ = 0.0;
    mutable Real weight_ = 0.0;
    mutable int evalState_ = 0;
    
    // Shape function values cache for Jacobian computation
    mutable ShapeValues geomShapeValues_;
};

// =============================================================================
// Inline implementations for simple getters/setters
// =============================================================================

inline ElementTransform::ElementTransform(const Mesh* mesh, Index elemIdx)
    : mesh_(mesh), elemIdx_(elemIdx) {
    computeGeometryInfo();
}

inline void ElementTransform::setElement(Index elemIdx) {
    elemIdx_ = elemIdx;
    evalState_ = 0;
    computeGeometryInfo();
}

inline void ElementTransform::setIntegrationPoint(const IntegrationPoint& ip) {
    ip_ = ip;
    evalState_ = 0;
}

inline void ElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    evalState_ = 0;
}

inline void ElementTransform::transform(const Real* xi, Vector3& x) const {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
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

inline Real ElementTransform::weight() const {
    evalWeight();
    return weight_;
}

inline const Matrix& ElementTransform::adjJacobian() const {
    evalAdjugate();
    return adjJacobian_;
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_TRANSFORM_HPP