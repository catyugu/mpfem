#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "mesh/mesh.hpp"
#include "fe_collection.hpp"
#include "core/types.hpp"
#include <vector>
#include <memory>

namespace mpfem {

/**
 * @brief Finite element space managing degrees of freedom on a mesh.
 * 
 * FESpace provides:
 * - Mapping from element local dofs to global dof indices
 * - Total number of degrees of freedom
 * - Support for scalar and vector fields
 */
class FESpace {
public:
    /// Default constructor
    FESpace() = default;
    
    /// Construct from mesh and FE collection (does not take ownership)
    FESpace(const Mesh* mesh, const FECollection* fec, int vdim = 1)
        : mesh_(mesh), fecRef_(fec), vdim_(vdim) {
        buildDofTable();
    }
    
    /// Construct and take ownership of fec
    FESpace(const Mesh* mesh, std::unique_ptr<FECollection> fec, int vdim = 1)
        : mesh_(mesh), fec_(std::move(fec)), vdim_(vdim) {
        buildDofTable();
    }
    
    // -------------------------------------------------------------------------
    // Mesh and FE info
    // -------------------------------------------------------------------------
    
    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }
    
    /// Get the FE collection
    const FECollection* fec() const { 
        return fec_ ? fec_.get() : fecRef_; 
    }
    
    /// Get polynomial order
    int order() const { 
        const FECollection* f = fec(); 
        return f ? f->order() : 0; 
    }
    
    /// Get vector dimension (1 = scalar, 2/3 = vector)
    int vdim() const { return vdim_; }
    
    /// Get spatial dimension
    int dim() const { return mesh_ ? mesh_->dim() : 0; }
    
    // -------------------------------------------------------------------------
    // DOF access
    // -------------------------------------------------------------------------
    
    /// Get total number of degrees of freedom
    Index numDofs() const { return numDofs_; }
    
    /// Get number of true (unconstrained) dofs
    Index numTrueDofs() const { return numTrueDofs_; }
    
    /// Get local to global dof mapping for an element
    void getElementDofs(Index elemIdx, std::vector<Index>& dofs) const;
    
    /// Get local to global dof mapping for a boundary element
    void getBdrElementDofs(Index bdrIdx, std::vector<Index>& dofs) const;
    
    /// Get local to global dof mapping (return vector)
    std::vector<Index> elementDofs(Index elemIdx) const {
        std::vector<Index> dofs;
        getElementDofs(elemIdx, dofs);
        return dofs;
    }
    
    /// Get number of dofs for an element
    int numElementDofs(Index elemIdx) const;
    
    /// Get number of dofs for a boundary element
    int numBdrElementDofs(Index bdrIdx) const;
    
    // -------------------------------------------------------------------------
    // Reference element access
    // -------------------------------------------------------------------------
    
    /// Get reference element for a given geometry
    const ReferenceElement* refElement(Geometry geom) const {
        const FECollection* f = fec();
        return f ? f->get(geom) : nullptr;
    }
    
    /// Get reference element for an element
    const ReferenceElement* elementRefElement(Index elemIdx) const {
        if (!mesh_ || elemIdx >= mesh_->numElements()) return nullptr;
        return refElement(mesh_->element(elemIdx).geometry());
    }
    
    /// Get reference element for a boundary element
    const ReferenceElement* bdrElementRefElement(Index bdrIdx) const {
        if (!mesh_ || bdrIdx >= mesh_->numBdrElements()) return nullptr;
        return refElement(mesh_->bdrElement(bdrIdx).geometry());
    }
    
    // -------------------------------------------------------------------------
    // Dof ordering
    // -------------------------------------------------------------------------
    
    /// Ordering type
    enum class Ordering {
        byNodes,   ///< All components at each node
        byVDim     ///< All nodes for each component
    };
    
    /// Get ordering type
    Ordering ordering() const { return ordering_; }
    
    /// Set ordering type
    void setOrdering(Ordering ord) { ordering_ = ord; }
    
private:
    void buildDofTable();
    
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;   ///< Owned FE collection
    const FECollection* fecRef_ = nullptr; ///< Non-owning reference
    int vdim_ = 1;
    Ordering ordering_ = Ordering::byVDim;
    
    Index numDofs_ = 0;
    Index numTrueDofs_ = 0;
    
    // Element to dof mapping: elemDofs_[elemIdx * maxDofsPerElem + localDof]
    std::vector<Index> elemDofs_;
    std::vector<Index> bdrElemDofs_;
    int maxDofsPerElem_ = 0;
    int maxDofsPerBdrElem_ = 0;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline void FESpace::getElementDofs(Index elemIdx, std::vector<Index>& dofs) const {
    const FECollection* f = fec();
    if (!mesh_ || !f || elemIdx >= mesh_->numElements()) {
        dofs.clear();
        return;
    }
    
    const Element& elem = mesh_->element(elemIdx);
    const ReferenceElement* refElem = f->get(elem.geometry());
    if (!refElem) {
        dofs.clear();
        return;
    }
    
    int nd = refElem->numDofs();
    dofs.resize(nd * vdim_);
    
    const Index base = elemIdx * maxDofsPerElem_;
    for (int i = 0; i < nd; ++i) {
        Index globalDof = elemDofs_[base + i];
        
        if (ordering_ == Ordering::byVDim) {
            for (int c = 0; c < vdim_; ++c) {
                dofs[c * nd + i] = globalDof + c * (numDofs_ / vdim_);
            }
        } else {
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }
}

inline void FESpace::getBdrElementDofs(Index bdrIdx, std::vector<Index>& dofs) const {
    const FECollection* f = fec();
    if (!mesh_ || !f || bdrIdx >= mesh_->numBdrElements()) {
        dofs.clear();
        return;
    }
    
    const Element& bdrElem = mesh_->bdrElement(bdrIdx);
    const ReferenceElement* refElem = f->get(bdrElem.geometry());
    if (!refElem) {
        dofs.clear();
        return;
    }
    
    int nd = refElem->numDofs();
    dofs.resize(nd * vdim_);
    
    const Index base = bdrIdx * maxDofsPerBdrElem_;
    for (int i = 0; i < nd; ++i) {
        Index globalDof = bdrElemDofs_[base + i];
        
        if (ordering_ == Ordering::byVDim) {
            for (int c = 0; c < vdim_; ++c) {
                dofs[c * nd + i] = globalDof + c * (numDofs_ / vdim_);
            }
        } else {
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }
}

inline int FESpace::numElementDofs(Index elemIdx) const {
    const FECollection* f = fec();
    if (!mesh_ || !f || elemIdx >= mesh_->numElements()) return 0;
    
    const Element& elem = mesh_->element(elemIdx);
    const ReferenceElement* refElem = f->get(elem.geometry());
    return refElem ? refElem->numDofs() * vdim_ : 0;
}

inline int FESpace::numBdrElementDofs(Index bdrIdx) const {
    const FECollection* f = fec();
    if (!mesh_ || !f || bdrIdx >= mesh_->numBdrElements()) return 0;
    
    const Element& bdrElem = mesh_->bdrElement(bdrIdx);
    const ReferenceElement* refElem = f->get(bdrElem.geometry());
    return refElem ? refElem->numDofs() * vdim_ : 0;
}

inline void FESpace::buildDofTable() {
    const FECollection* f = fec();
    if (!mesh_ || !f) return;
    
    // Find max dofs per element
    maxDofsPerElem_ = 0;
    for (Index i = 0; i < mesh_->numElements(); ++i) {
        const Element& elem = mesh_->element(i);
        const ReferenceElement* refElem = f->get(elem.geometry());
        if (refElem) {
            maxDofsPerElem_ = std::max(maxDofsPerElem_, refElem->numDofs());
        }
    }
    
    maxDofsPerBdrElem_ = 0;
    for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
        const Element& bdrElem = mesh_->bdrElement(i);
        const ReferenceElement* refElem = f->get(bdrElem.geometry());
        if (refElem) {
            maxDofsPerBdrElem_ = std::max(maxDofsPerBdrElem_, refElem->numDofs());
        }
    }
    
    // Build element dof table
    // For now, use simple node-based dof numbering
    // This assumes each vertex has one dof
    Index dofCounter = 0;
    std::vector<Index> vertexDofs(mesh_->numVertices(), InvalidIndex);
    
    elemDofs_.resize(mesh_->numElements() * maxDofsPerElem_);
    
    for (Index i = 0; i < mesh_->numElements(); ++i) {
        const Element& elem = mesh_->element(i);
        const ReferenceElement* refElem = f->get(elem.geometry());
        const Index base = i * maxDofsPerElem_;
        
        if (refElem) {
            int nd = refElem->numDofs();
            const auto& elemVertices = elem.vertices();
            
            for (int j = 0; j < nd; ++j) {
                // Map local dof to vertex
                Index vertexIdx = elemVertices[j];  // Simplified: assumes dofs = vertices
                if (vertexIdx >= 0 && vertexIdx < mesh_->numVertices()) {
                    if (vertexDofs[vertexIdx] == InvalidIndex) {
                        vertexDofs[vertexIdx] = dofCounter++;
                    }
                    elemDofs_[base + j] = vertexDofs[vertexIdx];
                } else {
                    elemDofs_[base + j] = InvalidIndex;
                }
            }
        }
    }
    
    // Build boundary element dof table
    bdrElemDofs_.resize(mesh_->numBdrElements() * maxDofsPerBdrElem_);
    
    for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
        const Element& bdrElem = mesh_->bdrElement(i);
        const ReferenceElement* refElem = f->get(bdrElem.geometry());
        const Index base = i * maxDofsPerBdrElem_;
        
        if (refElem) {
            int nd = refElem->numDofs();
            const auto& elemVertices = bdrElem.vertices();
            
            for (int j = 0; j < nd; ++j) {
                Index vertexIdx = elemVertices[j];
                if (vertexIdx >= 0 && vertexIdx < mesh_->numVertices() &&
                    vertexDofs[vertexIdx] != InvalidIndex) {
                    bdrElemDofs_[base + j] = vertexDofs[vertexIdx];
                } else {
                    bdrElemDofs_[base + j] = InvalidIndex;
                }
            }
        }
    }
    
    numDofs_ = dofCounter * vdim_;
    numTrueDofs_ = numDofs_;  // No constraints for now
}

}  // namespace mpfem

#endif  // MPFEM_FE_SPACE_HPP