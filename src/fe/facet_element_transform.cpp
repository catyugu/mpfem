#include "fe/facet_element_transform.hpp"
#include "mesh/mesh_topology.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// FacetElementTransform Implementation
// =============================================================================

FacetElementTransform::FacetElementTransform(const Mesh* mesh, 
                                             const MeshTopology* topo, 
                                             Index bdrElemIdx)
    : ElementTransform(mesh, bdrElemIdx, BOUNDARY), topo_(topo) {
}

void FacetElementTransform::setMesh(const Mesh* mesh) {
    ElementTransform::setMesh(mesh);
    adjElemComputed_ = false;
}

void FacetElementTransform::setElement(Index bdrElemIdx) {
    elemIdx_ = bdrElemIdx;
    elemType_ = BOUNDARY;
    evalState_ = 0;
    adjElemComputed_ = false;
    computeGeometryInfo();
}

void FacetElementTransform::setTopology(const MeshTopology* topo) {
    topo_ = topo;
    adjElemComputed_ = false;
}

void FacetElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    if (elemIdx_ >= mesh_->numBdrElements()) return;
    
    const Element& elem = mesh_->bdrElement(elemIdx_);
    geometry_ = elem.geometry();
    spaceDim_ = mesh_->dim();
    dim_ = geom::dim(geometry_);
    geomOrder_ = elem.order();
    
    // Get node coordinates
    nodeIndices_ = elem.vertices();
    nodes_.resize(nodeIndices_.size());
    for (size_t i = 0; i < nodeIndices_.size(); ++i) {
        nodes_[i] = mesh_->vertex(nodeIndices_[i]).toVector();
    }
    
    // Pre-allocate matrices
    jacobian_.setZero(spaceDim_, dim_);
    invJacobian_.setZero(dim_, spaceDim_);
    invJacobianT_.setZero(spaceDim_, dim_);
    adjJacobian_.setZero(spaceDim_, dim_);
    
    initGeometricShapeFunction();
    
    // Pre-allocate storage for shape function evaluation (avoids runtime allocation during assembly)
    if (shapeFunc_) {
        const int numDofs = shapeFunc_->numDofs();
        shapeValuesOnly_.resize(numDofs);
        shapeGradsOnly_.resize(numDofs);
    }
    
    // Reset cached adjacent element info
    adjElemComputed_ = false;
    adjElemIdx_ = InvalidIndex;
    localFaceIdx_ = -1;
}

Index FacetElementTransform::boundaryAttribute() const {
    return attribute();  // Already handled in base class
}

Index FacetElementTransform::adjacentElementIndex() const {
    computeAdjacentElementInfo();
    return adjElemIdx_;
}

int FacetElementTransform::localFaceIndex() const {
    computeAdjacentElementInfo();
    return localFaceIdx_;
}

bool FacetElementTransform::getAdjacentElementTransform(ElementTransform& trans) const {
    computeAdjacentElementInfo();
    if (adjElemIdx_ == InvalidIndex) return false;
    trans.setMesh(mesh_);
    trans.setElement(adjElemIdx_);
    return true;
}

Vector3 FacetElementTransform::normal() const {
    evalJacobian();
    
    Vector3 n(0.0, 0.0, 0.0);
    
    if (dim_ == 2 && spaceDim_ == 3) {
        // Surface in 3D: n = (dF/dxi x dF/deta) / |dF/dxi x dF/deta|
        Vector3 t1(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
        Vector3 t2(jacobian_(0, 1), jacobian_(1, 1), jacobian_(2, 1));
        n = t1.cross(t2);
        n.normalize();
    }
    else if (dim_ == 1 && spaceDim_ == 2) {
        // Curve in 2D: n = tangent rotated 90 degrees
        Vector3 t(jacobian_(0, 0), jacobian_(1, 0), 0.0);
        // Rotate 90 degrees counterclockwise (outward normal convention)
        n = Vector3(-t.y(), t.x(), 0.0);
        n.normalize();
    }
    else if (dim_ == 1 && spaceDim_ == 3) {
        // Curve in 3D: need additional information for normal
        // Use cross product with a reference direction
        Vector3 t(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
        Vector3 ref(0, 0, 1);
        if (std::abs(t.dot(ref)) > 0.9) {
            ref = Vector3(1, 0, 0);
        }
        n = t.cross(ref);
        n.normalize();
    }
    
    return n;
}

void FacetElementTransform::computeAdjacentElementInfo() const {
    if (adjElemComputed_) return;
    
    adjElemIdx_ = InvalidIndex;
    localFaceIdx_ = -1;
    
    if (!mesh_ || !topo_) {
        adjElemComputed_ = true;
        return;
    }
    
    // Get the topology face index for this boundary element
    Index faceIdx = topo_->getBoundaryFaceIndex(elemIdx_);
    if (faceIdx == InvalidIndex) {
        adjElemComputed_ = true;
        return;
    }
    
    // Get the face info
    const auto& faceInfo = topo_->getFaceInfo(faceIdx);
    
    // For boundary faces, elem1 is the interior element
    adjElemIdx_ = faceInfo.elem1;
    localFaceIdx_ = faceInfo.localFace1;
    
    adjElemComputed_ = true;
}

bool FacetElementTransform::mapToVolumeElement(const Real* bdrXi, Real* volXi) const {
    computeAdjacentElementInfo();
    
    if (adjElemIdx_ == InvalidIndex || localFaceIdx_ < 0) {
        return false;
    }
    
    const Element& volElem = mesh_->element(adjElemIdx_);
    Geometry volGeom = volElem.geometry();
    
    // Tetrahedron mapping
    if (volGeom == Geometry::Tetrahedron) {
        const Real xi = bdrXi[0];
        const Real eta = bdrXi[1];
        
        switch (localFaceIdx_) {
            case 0:  volXi[0] = xi; volXi[1] = eta; volXi[2] = 1.0 - xi - eta; break;
            case 1:  volXi[0] = 0.0; volXi[1] = eta; volXi[2] = 1.0 - xi - eta; break;
            case 2:  volXi[0] = xi; volXi[1] = 0.0; volXi[2] = 1.0 - xi - eta; break;
            case 3:  volXi[0] = xi; volXi[1] = eta; volXi[2] = 0.0; break;
            default: return false;
        }
        return true;
    }
    
    // Hexahedron mapping
    if (volGeom == Geometry::Cube) {
        const Real xi = bdrXi[0];
        const Real eta = bdrXi[1];
        
        switch (localFaceIdx_) {
            case 0:  volXi[0] = xi; volXi[1] = eta; volXi[2] = -1.0; break;
            case 1:  volXi[0] = xi; volXi[1] = eta; volXi[2] = 1.0; break;
            case 2:  volXi[0] = xi; volXi[1] = -1.0; volXi[2] = eta; break;
            case 3:  volXi[0] = xi; volXi[1] = 1.0; volXi[2] = eta; break;
            case 4:  volXi[0] = -1.0; volXi[1] = xi; volXi[2] = eta; break;
            case 5:  volXi[0] = 1.0; volXi[1] = xi; volXi[2] = eta; break;
            default: return false;
        }
        return true;
    }
    
    // Triangle mapping (2D)
    if (volGeom == Geometry::Triangle) {
        const Real xi = bdrXi[0];
        
        switch (localFaceIdx_) {
            case 0:  volXi[0] = xi; volXi[1] = 1.0 - xi; break;
            case 1:  volXi[0] = 0.0; volXi[1] = xi; break;
            case 2:  volXi[0] = xi; volXi[1] = 0.0; break;
            default: return false;
        }
        return true;
    }
    
    // Square mapping (2D)
    if (volGeom == Geometry::Square) {
        const Real xi = bdrXi[0];
        
        switch (localFaceIdx_) {
            case 0:  volXi[0] = xi; volXi[1] = -1.0; break;
            case 1:  volXi[0] = 1.0; volXi[1] = xi; break;
            case 2:  volXi[0] = xi; volXi[1] = 1.0; break;
            case 3:  volXi[0] = -1.0; volXi[1] = xi; break;
            default: return false;
        }
        return true;
    }
    
    return false;
}

}  // namespace mpfem