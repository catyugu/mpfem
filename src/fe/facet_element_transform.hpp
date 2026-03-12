#ifndef MPFEM_FACET_ELEMENT_TRANSFORM_HPP
#define MPFEM_FACET_ELEMENT_TRANSFORM_HPP

#include "fe/element_transform.hpp"
#include <cmath>

namespace mpfem {

// Forward declaration
class MeshTopology;

/**
 * @file facet_element_transform.hpp
 * @brief Transformation for boundary facet elements.
 * 
 * FacetElementTransform inherits from ElementTransform and provides
 * additional functionality for boundary elements:
 * - Normal vector computation
 * - Access to adjacent volume element
 * - Integration point mapping from boundary to volume element
 * 
 * Design inspired by MFEM's FaceElementTransformations class.
 */
class FacetElementTransform : public ElementTransform {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    FacetElementTransform() : ElementTransform(nullptr, InvalidIndex, BOUNDARY) {}
    
    FacetElementTransform(const Mesh* mesh, Index bdrElemIdx)
        : ElementTransform(mesh, bdrElemIdx, BOUNDARY) {}
    
    FacetElementTransform(const Mesh* mesh, const MeshTopology* topo, Index bdrElemIdx);
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    void setMesh(const Mesh* mesh) override;
    void setElement(Index bdrElemIdx) override;
    
    /// Set boundary element index (alias for setElement)
    void setBoundaryElement(Index bdrElemIdx) { setElement(bdrElemIdx); }
    
    /// Set mesh topology (required for adjacent element access)
    void setTopology(const MeshTopology* topo);
    
    // -------------------------------------------------------------------------
    // Boundary-specific properties
    // -------------------------------------------------------------------------
    
    /// Get the boundary attribute (boundary condition ID)
    Index boundaryAttribute() const;
    
    /**
     * @brief Get outward unit normal vector at current integration point.
     * 
     * For 2D surface in 3D: n = (dF/dxi x dF/deta) / |dF/dxi x dF/deta|
     * For 1D curve in 2D: n = tangent rotated 90 degrees
     */
    Vector3 normal() const;
    
    // -------------------------------------------------------------------------
    // Adjacent element access
    // -------------------------------------------------------------------------
    
    /// Check if topology information is available
    bool hasTopology() const { return topo_ != nullptr; }
    
    /// Get adjacent volume element index
    Index adjacentElementIndex() const;
    
    /// Get local face index in the adjacent element
    int localFaceIndex() const;
    
    /// Get ElementTransform for the adjacent volume element
    bool getAdjacentElementTransform(ElementTransform& trans) const;
    
    /// Map integration point from boundary to volume element coordinates
    bool mapToVolumeElement(const Real* bdrXi, Real* volXi) const;
    
private:
    void computeGeometryInfo() override;
    void computeAdjacentElementInfo() const;
    
    const MeshTopology* topo_ = nullptr;
    
    // Cached adjacent element info
    mutable Index adjElemIdx_ = InvalidIndex;
    mutable int localFaceIdx_ = -1;
    mutable bool adjElemComputed_ = false;
};

}  // namespace mpfem

#endif  // MPFEM_FACET_ELEMENT_TRANSFORM_HPP