#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "mesh/mesh.hpp"
#include "mesh/mesh_topology.hpp"
#include "fe_collection.hpp"
#include "core/types.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>

namespace mpfem {

// Custom hash function for std::pair
struct PairHash {
    std::size_t operator()(const std::pair<Index, Index>& p) const {
        // Combine hashes using a common technique
        std::size_t h1 = std::hash<Index>()(p.first);
        std::size_t h2 = std::hash<Index>()(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

/**
 * @brief Finite element space managing degrees of freedom on a mesh.
 * 
 * FESpace provides:
 * - Mapping from element local dofs to global dof indices
 * - Total number of degrees of freedom
 * - Support for scalar and vector fields
 * 
 * DOF ordering for higher-order elements:
 * - First N vertex dofs (N = number of vertices)
 * - Then edge dofs (one per edge for order >= 2)
 * - Then face dofs (if needed for order >= 3)
 * - Then interior dofs (if needed)
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
    
    /// Build edge-to-dof mapping for higher-order elements
    void buildEdgeDofMap(Index& dofCounter);
    
    /// Build face-to-dof mapping for higher-order elements
    void buildFaceDofMap(Index& dofCounter);
    
    /// Get edge key (sorted vertex pair)
    static std::pair<Index, Index> makeEdgeKey(Index v1, Index v2) {
        return v1 < v2 ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
    }
    
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
    
    // Edge to DOF mapping (for higher-order elements)
    std::unordered_map<std::pair<Index, Index>, Index, PairHash> edgeDofMap_;
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

inline void FESpace::buildEdgeDofMap(Index& dofCounter) {
    // Build edge-to-dof mapping by processing all elements
    edgeDofMap_.clear();
    
    for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
        const Element& elem = mesh_->element(elemIdx);
        
        // Process each edge of the element
        for (int e = 0; e < elem.numEdges(); ++e) {
            auto [v1, v2] = elem.edgeVertices(e);
            auto key = makeEdgeKey(v1, v2);
            
            if (edgeDofMap_.find(key) == edgeDofMap_.end()) {
                edgeDofMap_[key] = dofCounter++;
            }
        }
    }
    
    // Also process boundary elements
    for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
        const Element& bdrElem = mesh_->bdrElement(bdrIdx);
        
        for (int e = 0; e < bdrElem.numEdges(); ++e) {
            auto [v1, v2] = bdrElem.edgeVertices(e);
            auto key = makeEdgeKey(v1, v2);
            
            if (edgeDofMap_.find(key) == edgeDofMap_.end()) {
                edgeDofMap_[key] = dofCounter++;
            }
        }
    }
}

inline void FESpace::buildDofTable() {
    const FECollection* f = fec();
    if (!mesh_ || !f) return;
    
    int feOrder = f->order();
    
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
    
    // Assign DOFs
    Index dofCounter = 0;
    
    // Step 1: Assign vertex DOFs
    std::vector<Index> vertexDofs(mesh_->numVertices(), InvalidIndex);
    for (Index v = 0; v < mesh_->numVertices(); ++v) {
        vertexDofs[v] = dofCounter++;
    }
    
    // Step 2: Build edge DOF map for order >= 2
    if (feOrder >= 2) {
        buildEdgeDofMap(dofCounter);
    }
    
    // Step 3: Build element DOF table
    elemDofs_.resize(mesh_->numElements() * maxDofsPerElem_, InvalidIndex);
    
    for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
        const Element& elem = mesh_->element(elemIdx);
        const ReferenceElement* refElem = f->get(elem.geometry());
        const Index base = elemIdx * maxDofsPerElem_;
        
        if (!refElem) continue;
        
        int nd = refElem->numDofs();
        int numCorners = elem.numCorners();
        int numEdges = elem.numEdges();
        
        // Assign vertex DOFs (first numCorners dofs)
        for (int j = 0; j < numCorners && j < nd; ++j) {
            Index vertexIdx = elem.vertex(j);
            elemDofs_[base + j] = vertexDofs[vertexIdx];
        }
        
        // Assign edge DOFs for order >= 2
        if (feOrder >= 2 && nd > numCorners) {
            for (int e = 0; e < numEdges && (numCorners + e) < nd; ++e) {
                auto [v1, v2] = elem.edgeVertices(e);
                auto key = makeEdgeKey(v1, v2);
                auto it = edgeDofMap_.find(key);
                if (it != edgeDofMap_.end()) {
                    elemDofs_[base + numCorners + e] = it->second;
                }
            }
        }
    }
    
    // Step 4: Build boundary element DOF table
    bdrElemDofs_.resize(mesh_->numBdrElements() * maxDofsPerBdrElem_, InvalidIndex);
    
    for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
        const Element& bdrElem = mesh_->bdrElement(bdrIdx);
        const ReferenceElement* refElem = f->get(bdrElem.geometry());
        const Index base = bdrIdx * maxDofsPerBdrElem_;
        
        if (!refElem) continue;
        
        int nd = refElem->numDofs();
        int numCorners = bdrElem.numCorners();
        int numEdges = bdrElem.numEdges();
        
        // Assign vertex DOFs
        for (int j = 0; j < numCorners && j < nd; ++j) {
            Index vertexIdx = bdrElem.vertex(j);
            bdrElemDofs_[base + j] = vertexDofs[vertexIdx];
        }
        
        // Assign edge DOFs for order >= 2
        if (feOrder >= 2 && nd > numCorners) {
            for (int e = 0; e < numEdges && (numCorners + e) < nd; ++e) {
                auto [v1, v2] = bdrElem.edgeVertices(e);
                auto key = makeEdgeKey(v1, v2);
                auto it = edgeDofMap_.find(key);
                if (it != edgeDofMap_.end()) {
                    bdrElemDofs_[base + numCorners + e] = it->second;
                }
            }
        }
    }
    
    numDofs_ = dofCounter * vdim_;
    numTrueDofs_ = numDofs_;  // No constraints for now
}

}  // namespace mpfem

#endif  // MPFEM_FE_SPACE_HPP
