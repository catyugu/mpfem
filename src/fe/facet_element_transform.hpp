#ifndef MPFEM_FACET_ELEMENT_TRANSFORM_HPP
#define MPFEM_FACET_ELEMENT_TRANSFORM_HPP

#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include "shape_function.hpp"

namespace mpfem {

// Forward declaration
class MeshTopology;

/**
 * @brief Transformation for boundary facet elements.
 * 
 * FacetElementTransform handles coordinate transformation for boundary 
 * elements (triangles, quadrilaterals in 3D; segments in 2D). It provides:
 * - Normal vector computation for boundary faces
 * - Jacobian and weight computation for surface integrals
 * - Access to adjacent volume element information
 * 
 * Thread safety: Each thread should have its own FacetElementTransform instance.
 * The cached Jacobian data uses mutable for lazy evaluation but is not thread-safe
 * when shared between threads.
 * 
 * Usage:
 *   FacetElementTransform trans(&mesh, bdrElemIdx);
 *   for (const auto& ip : quadrature) {
 *       trans.setIntegrationPoint(ip);
 *       Vector3 n = trans.normal();
 *       Real w = trans.weight() * ip.weight;
 *       // ... surface integral
 *   }
 */
class FacetElementTransform {
public:
    /// Default constructor
    FacetElementTransform() = default;
    
    /// Construct with mesh and boundary element index
    FacetElementTransform(const Mesh* mesh, Index bdrElemIdx);
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    /// Set the mesh
    void setMesh(const Mesh* mesh) { mesh_ = mesh; }
    
    /// Set the boundary element index
    void setBoundaryElement(Index bdrElemIdx);
    
    /// Set integration point (invalidates cached Jacobian data)
    void setIntegrationPoint(const Real* xi);
    
    /// Reset evaluation state (force recompute on next access)
    void reset() { evalState_ = 0; }
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension of the boundary element (one less than mesh dim)
    int dim() const { return dim_; }
    
    /// Get spatial dimension of the parent mesh
    int spaceDim() const { return spaceDim_; }
    
    /// Get boundary element index
    Index boundaryElementIndex() const { return bdrElemIdx_; }
    
    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }
    
    /// Get the boundary attribute (boundary ID)
    Index boundaryAttribute() const;
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    /// Transform reference coordinates to physical coordinates
    void transform(const Real* xi, Real* x) const;
    void transform(const Real* xi, Vector3& x) const;
    
    // -------------------------------------------------------------------------
    // Jacobian and related quantities
    // -------------------------------------------------------------------------
    
    /// Get the Jacobian matrix (spaceDim x dim)
    /// For a triangle in 3D: J = [dF/dxi, dF/deta] (3x2 matrix)
    const Matrix& jacobian() const;
    
    /// Get the Jacobian determinant (area scaling factor)
    /// For surface elements: |det(J)| = |dF/dxi x dF/deta|
    Real detJ() const;
    
    /// Get the weight = |det(J)| for surface integrals
    Real weight() const;
    
    // -------------------------------------------------------------------------
    // Normal vector
    // -------------------------------------------------------------------------
    
    /**
     * @brief Get outward unit normal vector at current integration point.
     * 
     * For 2D surface in 3D: n = (dF/dxi x dF/deta) / |dF/dxi x dF/deta|
     * For 1D curve in 2D: n = tangent rotated 90 degrees
     * 
     * The normal direction follows the right-hand rule with the boundary
     * element's vertex ordering (outward from the domain).
     */
    Vector3 normal() const;
    
    // -------------------------------------------------------------------------
    // Element vertices (convenience)
    // -------------------------------------------------------------------------
    
    /// Get vertex coordinates
    const std::vector<Vector3>& vertices() const { return vertices_; }
    
    /// Get number of vertices
    int numVertices() const { return static_cast<int>(vertices_.size()); }
    
private:
    // -------------------------------------------------------------------------
    // Evaluation state management
    // -------------------------------------------------------------------------
    enum EvalMask {
        JACOBIAN_MASK = 1,
        WEIGHT_MASK   = 2
    };
    
    void computeGeometryInfo();
    void createGeomShapeFunction();
    void evalJacobian() const;
    void evalJacobianLinear() const;
    void evalWeight() const;
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    const Mesh* mesh_ = nullptr;
    Index bdrElemIdx_ = 0;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;        // Dimension of boundary element (1 for segment, 2 for triangle/quad)
    int spaceDim_ = 0;   // Dimension of parent mesh
    int geomOrder_ = 1;  // Geometric order (from element)
    std::vector<Vector3> vertices_;
    std::vector<Index> vertexIndices_;
    
    IntegrationPoint ip_;
    
    // Geometric shape function for curved boundary elements
    std::unique_ptr<ShapeFunction> geomShapeFunc_;
    
    // Mutable for lazy evaluation
    mutable Matrix jacobian_;  // spaceDim_ x dim_
    mutable Real detJ_ = 0.0;
    mutable Real weight_ = 0.0;
    mutable int evalState_ = 0;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline FacetElementTransform::FacetElementTransform(const Mesh* mesh, Index bdrElemIdx)
    : mesh_(mesh), bdrElemIdx_(bdrElemIdx) {
    computeGeometryInfo();
}

inline void FacetElementTransform::setBoundaryElement(Index bdrElemIdx) {
    bdrElemIdx_ = bdrElemIdx;
    evalState_ = 0;
    computeGeometryInfo();
}

inline void FacetElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    evalState_ = 0;
}

inline void FacetElementTransform::transform(const Real* xi, Vector3& x) const {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
}

inline const Matrix& FacetElementTransform::jacobian() const {
    evalJacobian();
    return jacobian_;
}

inline Real FacetElementTransform::detJ() const {
    evalWeight();
    return detJ_;
}

inline Real FacetElementTransform::weight() const {
    evalWeight();
    return weight_;
}

}  // namespace mpfem

#endif  // MPFEM_FACET_ELEMENT_TRANSFORM_HPP
