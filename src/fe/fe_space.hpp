#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "fe_collection.hpp"
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
    class Mesh;

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
        int dim() const;

        bool isExternalBoundary(Index bdrElemIdx) const;
        bool isExternalBoundaryId(Index bdrId) const;

        // -------------------------------------------------------------------------
        // DOF access
        // -------------------------------------------------------------------------

        Index numDofs() const { return numDofs_; }
        Index scalarNumDofs() const { return scalarNumDofs_; }

        void getElementDofs(Index elemIdx, std::span<Index> dofs) const;
        void getBdrElementDofs(Index bdrIdx, std::span<Index> dofs) const;
        int numElementDofs(Index elemIdx) const;
        int numBdrElementDofs(Index bdrIdx) const;

        // -------------------------------------------------------------------------
        // Reference element access
        // -------------------------------------------------------------------------

        const ReferenceElement* refElement(Geometry geom) const;
        const ReferenceElement* elementRefElement(Index elemIdx) const;
        const ReferenceElement* bdrElementRefElement(Index bdrIdx) const;

        // -------------------------------------------------------------------------
        // Geometric order queries
        // -------------------------------------------------------------------------

        /// Get geometric order of an element
        int elementGeoOrder(Index elemIdx) const;

        /// Get geometric order of a boundary element
        int bdrElementGeoOrder(Index bdrIdx) const;

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
        int maxDofsPerElem_ = 0;
        int maxDofsPerBdrElem_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_FE_SPACE_HPP
