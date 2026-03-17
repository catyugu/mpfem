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
 * Design: NO mutable members. All geometric quantities are computed
 * eagerly when setElement() or setIntegrationPoint() is called.
 * 
 * Thread safety: Each thread should have its own transform instance.
 */

class ElementTransform {
public:
    enum ElementType { VOLUME = 1, BOUNDARY = 2 };
    
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    ElementTransform() = default;
    
    explicit ElementTransform(const Mesh* mesh, Index elemIdx, 
                              ElementType type = VOLUME);
    
    virtual ~ElementTransform() = default;
    
    // -------------------------------------------------------------------------
    // Setup (computes geometry info eagerly)
    // -------------------------------------------------------------------------
    
    virtual void setMesh(const Mesh* mesh);
    virtual void setElement(Index elemIdx);
    
    /// Set integration point (recomputes Jacobian and weight)
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Real* xi);
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    Geometry geometry() const { return geometry_; }
    int dim() const { return dim_; }
    int spaceDim() const { return spaceDim_; }
    Index elementIndex() const { return elemIdx_; }
    ElementType elementType() const { return elemType_; }
    const Mesh* mesh() const { return mesh_; }
    Index attribute() const;
    int geometricOrder() const { return geomOrder_; }
    bool isCurved() const { return geomOrder_ >= 2; }
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    virtual void transform(const Real* xi, Real* x);
    void transform(const Real* xi, Vector3& x);
    void transform(const IntegrationPoint& ip, Vector3& x);
    
    // -------------------------------------------------------------------------
    // Jacobian and related quantities (computed in setIntegrationPoint)
    // -------------------------------------------------------------------------
    
    const Matrix& jacobian() const { return jacobian_; }
    const Matrix& invJacobian() const { return invJacobian_; }
    const Matrix& invJacobianT() const { return invJacobianT_; }
    Real detJ() const { return detJ_; }
    virtual Real weight() const { return weight_; }
    
    // -------------------------------------------------------------------------
    // Gradient transformation
    // -------------------------------------------------------------------------
    
    void transformGradient(const Real* refGrad, Real* physGrad) const;
    void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;
    
    // -------------------------------------------------------------------------
    // Element nodes
    // -------------------------------------------------------------------------
    
    const std::vector<Vector3>& nodes() const { return nodes_; }
    int numNodes() const { return static_cast<int>(nodes_.size()); }
    
    const IntegrationPoint& integrationPoint() const { return ip_; }
    
protected:
    void computeGeometryInfo();
    void computeJacobianAtIP();
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    const Mesh* mesh_ = nullptr;
    Index elemIdx_ = InvalidIndex;
    ElementType elemType_ = VOLUME;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;
    int spaceDim_ = 0;
    int geomOrder_ = 1;
    
    std::unique_ptr<ShapeFunction> shapeFunc_;
    std::vector<Vector3> nodes_;
    std::vector<Index> nodeIndices_;
    
    IntegrationPoint ip_;
    
    // All computed eagerly (no lazy evaluation)
    Matrix jacobian_;       // spaceDim x dim
    Matrix invJacobian_;    // dim x spaceDim
    Matrix invJacobianT_;   // spaceDim x dim
    Real detJ_ = 0.0;
    Real weight_ = 0.0;
    
    // Pre-allocated buffers for shape function evaluation
    std::vector<Real> shapeValuesBuf_;
    std::vector<Vector3> shapeGradsBuf_;
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
    computeJacobianAtIP();
}

inline void ElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    computeJacobianAtIP();
}

inline void ElementTransform::transform(const Real* xi, Vector3& x) {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
}

inline void ElementTransform::transform(const IntegrationPoint& ip, Vector3& x) {
    transform(&ip.xi, x);
}

inline void ElementTransform::transformGradient(const Vector3& refGrad, Vector3& physGrad) const {
    Real rg[3] = {refGrad.x(), refGrad.y(), refGrad.z()};
    Real pg[3];
    transformGradient(rg, pg);
    physGrad = Vector3(pg[0], pg[1], pg[2]);
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_TRANSFORM_HPP