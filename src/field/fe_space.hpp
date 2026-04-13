#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "fe/fe_collection.hpp"
#include <algorithm>
#include <memory>
#include <span>
#include <vector>

namespace mpfem {

    class Mesh;

    class FESpace {
    public:
        FESpace() = default;

        FESpace(const Mesh* mesh, std::unique_ptr<FECollection> fec)
            : mesh_(mesh), fec_(std::move(fec))
        {
            buildDofTable();
        }

        const Mesh* mesh() const { return mesh_; }
        const FECollection* fec() const { return fec_.get(); }
        int order() const { return fec_ ? fec_->order() : 0; }
        int vdim() const { return fec_ ? fec_->vdim() : 1; }
        int dim() const;

        bool isExternalBoundary(Index bdrElemIdx) const;
        bool isExternalBoundaryId(Index bdrId) const;

        Index numDofs() const { return numDofs_; }

        void getElementDofs(Index elemIdx, std::span<Index> dofs) const;
        std::span<const int> getElementOrientations(Index elemIdx) const;
        void getBdrElementDofs(Index bdrIdx, std::span<Index> dofs) const;
        int numElementDofs(Index elemIdx) const;
        int numBdrElementDofs(Index bdrIdx) const;

        const ReferenceElement* refElement(Geometry geom) const;
        const ReferenceElement* elementRefElement(Index elemIdx) const;
        const ReferenceElement* bdrElementRefElement(Index bdrIdx) const;

        int elementGeoOrder(Index elemIdx) const;
        int bdrElementGeoOrder(Index bdrIdx) const;

    private:
        void buildDofTable();

        const Mesh* mesh_ = nullptr;
        std::unique_ptr<FECollection> fec_;

        Index numDofs_ = 0;

        std::vector<Index> elemDofs_;
        std::vector<int> elemOrientations_;
        std::vector<Index> bdrElemDofs_;
        int maxDofsPerElem_ = 0;
        int maxDofsPerBdrElem_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_FE_SPACE_HPP
