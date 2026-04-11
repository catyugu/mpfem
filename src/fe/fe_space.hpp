#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "fe_collection.hpp"
#include "mesh/mesh.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

namespace mpfem {

    /**
     * @brief Finite element space managing degrees of freedom on a mesh.
     *
     * **Geometric Order vs Field Order**:
     * - Geometric order: From mesh element, used for coordinate transformation (ElementTransform)
    * - Field order: From FECollection, used for FiniteElement basis and DOF management
     *
     * **Isoparametric, Subparametric, Superparametric**:
     * - Isoparametric: geo_order == field_order
     * - Subparametric: geo_order < field_order (linear mesh, quadratic field)
     * - Superparametric: geo_order > field_order (curved mesh, linear field)
     *
    * **DOF Mapping**:
    * - DOFs are attached to topology entities (vertex/edge/face/cell)
    * - Global numbering is contiguous by entity dimension
    * - Element local-to-global mapping is generated from mesh topology
     */
    class FESpace {
    public:
        FESpace() = default;

        FESpace(const Mesh* mesh, std::unique_ptr<FECollection> fec, int vdim = 1)
            : mesh_(mesh), fec_(std::move(fec)), vdim_(vdim)
        {
            buildDofTable();
        }

        // -------------------------------------------------------------------------
        // Mesh and FE info
        // -------------------------------------------------------------------------

        const Mesh* mesh() const { return mesh_; }
        const FECollection* fec() const { return fec_.get(); }
        int order() const { return fec_ ? fec_->order() : 0; }
        int vdim() const { return vdim_; }
        int dim() const { return mesh_ ? mesh_->dim() : 0; }

        bool isExternalBoundary(Index bdrElemIdx) const
        {
            return mesh_ ? mesh_->isExternalBoundary(bdrElemIdx) : true;
        }
        bool isExternalBoundaryId(Index bdrId) const
        {
            return mesh_ ? mesh_->isExternalBoundaryId(bdrId) : true;
        }

        // -------------------------------------------------------------------------
        // DOF access
        // -------------------------------------------------------------------------

        Index numDofs() const { return numDofs_; }
        Index scalarNumDofs() const { return scalarNumDofs_; }

        void getElementDofs(Index elemIdx, std::span<Index> dofs) const;
        void getBdrElementDofs(Index bdrIdx, std::span<Index> dofs) const;
        int numElementDofs(Index elemIdx) const;
        int numBdrElementDofs(Index bdrIdx) const;
        Index vertexScalarDof(Index vertexIdx, int localVertexDof = 0) const;
        Index vertexDof(Index vertexIdx, int component = 0, int localVertexDof = 0) const;

        // -------------------------------------------------------------------------
        // Reference element access
        // -------------------------------------------------------------------------

        const ReferenceElement* refElement(Geometry geom) const
        {
            return fec_ ? fec_->get(geom) : nullptr;
        }

        const ReferenceElement* elementRefElement(Index elemIdx) const
        {
            if (!mesh_)
                MPFEM_THROW(Exception, "mesh not set");
            if (elemIdx >= mesh_->numElements())
                MPFEM_THROW(RangeException, "invalid element index");
            return refElement(mesh_->element(elemIdx).geometry());
        }

        const ReferenceElement* bdrElementRefElement(Index bdrIdx) const
        {
            if (!mesh_)
                MPFEM_THROW(Exception, "mesh not set");
            if (bdrIdx >= mesh_->numBdrElements())
                MPFEM_THROW(RangeException, "invalid boundary element index");
            return refElement(mesh_->bdrElement(bdrIdx).geometry());
        }

        // -------------------------------------------------------------------------
        // Geometric order queries
        // -------------------------------------------------------------------------

        /// Get geometric order of an element
        int elementGeoOrder(Index elemIdx) const
        {
            return mesh_ ? mesh_->element(elemIdx).order() : 1;
        }

        /// Get geometric order of a boundary element
        int bdrElementGeoOrder(Index bdrIdx) const
        {
            return mesh_ ? mesh_->bdrElement(bdrIdx).order() : 1;
        }

        // -------------------------------------------------------------------------
        // Subparametric/Superparametric classification
        // -------------------------------------------------------------------------

        /// Check if element uses subparametric formulation (field_order > geo_order)
        bool isSubparametric(Index elemIdx) const
        {
            return order() > elementGeoOrder(elemIdx);
        }

        /// Check if element uses superparametric formulation (field_order < geo_order)
        bool isSuperparametric(Index elemIdx) const
        {
            return order() < elementGeoOrder(elemIdx);
        }

    private:
        void buildDofTable();

        const Mesh* mesh_ = nullptr;
        std::unique_ptr<FECollection> fec_;
        int vdim_ = 1;

        Index numDofs_ = 0;
        Index scalarNumDofs_ = 0;

        // Element DOF table: [elemIdx * maxDofsPerElem + localDof]
        std::vector<Index> elemDofs_;
        std::vector<Index> bdrElemDofs_;
        std::vector<Index> vertexDofBase_;
        int maxDofsPerElem_ = 0;
        int maxDofsPerBdrElem_ = 0;
    };

    // =============================================================================
    // Inline implementations
    // =============================================================================

    inline void FESpace::getElementDofs(Index elemIdx, std::span<Index> dofs) const
    {
        if (!mesh_ || !fec_ || elemIdx >= mesh_->numElements())
            return;

        const Element& elem = mesh_->element(elemIdx);
        const ReferenceElement* refElem = fec_->get(elem.geometry());
        if (!refElem)
            return;

        const int nd = refElem->numDofs();
        const int totalDofs = nd * vdim_;
        if (static_cast<int>(dofs.size()) < totalDofs)
            return;

        const Index base = elemIdx * maxDofsPerElem_;
        for (int i = 0; i < nd; ++i) {
            Index globalDof = elemDofs_[base + i];
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }

    inline void FESpace::getBdrElementDofs(Index bdrIdx, std::span<Index> dofs) const
    {
        if (!mesh_ || !fec_ || bdrIdx >= mesh_->numBdrElements())
            return;

        const Element& bdrElem = mesh_->bdrElement(bdrIdx);
        const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
        if (!refElem)
            return;

        const int nd = refElem->numDofs();
        const int totalDofs = nd * vdim_;
        if (static_cast<int>(dofs.size()) < totalDofs)
            return;

        const Index base = bdrIdx * maxDofsPerBdrElem_;
        for (int i = 0; i < nd; ++i) {
            Index globalDof = bdrElemDofs_[base + i];
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }

    inline int FESpace::numElementDofs(Index elemIdx) const
    {
        if (!mesh_ || !fec_ || elemIdx >= mesh_->numElements())
            return 0;
        const Element& elem = mesh_->element(elemIdx);
        const ReferenceElement* refElem = fec_->get(elem.geometry());
        return refElem ? refElem->numDofs() * vdim_ : 0;
    }

    inline int FESpace::numBdrElementDofs(Index bdrIdx) const
    {
        if (!mesh_ || !fec_ || bdrIdx >= mesh_->numBdrElements())
            return 0;
        const Element& bdrElem = mesh_->bdrElement(bdrIdx);
        const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
        return refElem ? refElem->numDofs() * vdim_ : 0;
    }

    inline Index FESpace::vertexScalarDof(Index vertexIdx, int localVertexDof) const
    {
        if (!mesh_ || vertexIdx >= mesh_->numVertices() || localVertexDof < 0) {
            return InvalidIndex;
        }
        if (vertexIdx >= static_cast<Index>(vertexDofBase_.size())) {
            return InvalidIndex;
        }
        const Index base = vertexDofBase_[vertexIdx];
        if (base == InvalidIndex) {
            return InvalidIndex;
        }
        return base + localVertexDof;
    }

    inline Index FESpace::vertexDof(Index vertexIdx, int component, int localVertexDof) const
    {
        if (component < 0 || component >= vdim_) {
            return InvalidIndex;
        }
        const Index scalarDof = vertexScalarDof(vertexIdx, localVertexDof);
        if (scalarDof == InvalidIndex) {
            return InvalidIndex;
        }
        return scalarDof * vdim_ + component;
    }

} // namespace mpfem

#endif // MPFEM_FE_SPACE_HPP
