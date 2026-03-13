#ifndef MPFEM_FACET_ELEMENT_TRANSFORM_HPP
#define MPFEM_FACET_ELEMENT_TRANSFORM_HPP

#include "fe/element_transform.hpp"
#include <cmath>

namespace mpfem {

class MeshTopology;

/**
 * @brief Transformation for boundary facet elements.
 */
class FacetElementTransform : public ElementTransform {
public:
    FacetElementTransform() : ElementTransform(nullptr, InvalidIndex, BOUNDARY) {}
    
    FacetElementTransform(const Mesh* mesh, Index bdrElemIdx)
        : ElementTransform(mesh, bdrElemIdx, BOUNDARY) {
        computeAdjacentElementInfo();
    }
    
    FacetElementTransform(const Mesh* mesh, const MeshTopology* topo, Index bdrElemIdx);
    
    void setMesh(const Mesh* mesh) override;
    void setElement(Index bdrElemIdx) override;
    void setBoundaryElement(Index bdrElemIdx) { setElement(bdrElemIdx); }
    void setTopology(const MeshTopology* topo);
    
    Index boundaryAttribute() const;
    Vector3 normal() const;
    
    bool hasTopology() const { return topo_ != nullptr; }
    Index adjacentElementIndex() const { return adjElemIdx_; }
    int localFaceIndex() const { return localFaceIdx_; }
    bool getAdjacentElementTransform(ElementTransform& trans) const;
    bool mapToVolumeElement(const Real* bdrXi, Real* volXi) const;
    
private:
    void computeAdjacentElementInfo();
    
    const MeshTopology* topo_ = nullptr;
    Index adjElemIdx_ = InvalidIndex;
    int localFaceIdx_ = -1;
};

}  // namespace mpfem

#endif  // MPFEM_FACET_ELEMENT_TRANSFORM_HPP
