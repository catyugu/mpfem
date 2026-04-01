#include "fe/fe_space.hpp"
#include "mesh/element.hpp"
#include "mesh/mesh.hpp"

namespace mpfem {

    void FESpace::buildDofTable()
    {
        if (!mesh_ || !fec_)
            return;

        const int fieldOrder = fec_->order();

        // Find max dofs per element based on field order
        maxDofsPerElem_ = 0;
        maxDofsPerBdrElem_ = 0;

        for (Index i = 0; i < mesh_->numElements(); ++i) {
            const Element& elem = mesh_->element(i);
            const ReferenceElement* refElem = fec_->get(elem.geometry());
            if (refElem) {
                maxDofsPerElem_ = std::max(maxDofsPerElem_, refElem->numDofs());
            }
        }

        for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
            const Element& bdrElem = mesh_->bdrElement(i);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
            if (refElem) {
                maxDofsPerBdrElem_ = std::max(maxDofsPerBdrElem_, refElem->numDofs());
            }
        }

        // For H1 Lagrange elements:
        // - Each corner vertex has exactly 1 DOF
        // - Additional DOFs exist only if mesh has higher-order vertices
        //
        // DOF allocation strategy:
        // - Isoparametric: field_order == geo_order -> DOFs from mesh vertices
        // - Superparametric: field_order < geo_order -> subset of mesh vertices
        // - Subparametric: field_order > geo_order -> only valid DOFs are mesh vertices,
        //                                                  DOFs beyond mesh vertices are InvalidIndex
        //
        // Total DOFs = mesh vertices (one per vertex for H1)

        // Allocate element DOF table
        elemDofs_.resize(mesh_->numElements() * maxDofsPerElem_, InvalidIndex);
        bdrElemDofs_.resize(mesh_->numBdrElements() * maxDofsPerBdrElem_, InvalidIndex);

        // Build element DOF mapping
        for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
            const Element& elem = mesh_->element(elemIdx);
            const ReferenceElement* refElem = fec_->get(elem.geometry());
            if (!refElem)
                continue;

            const int fieldDofs = refElem->numDofs();
            const int geoNodes = static_cast<int>(elem.vertices().size());
            const int geoOrder = elem.order();

            const Index base = elemIdx * maxDofsPerElem_;

            if (fieldOrder == geoOrder) {
                // Isoparametric: DOFs exactly match mesh vertices
                for (int j = 0; j < fieldDofs; ++j) {
                    elemDofs_[base + j] = elem.vertex(j);
                }
            }
            else if (fieldOrder > geoOrder) {
                // Subparametric: field order > geo order
                // Mesh has fewer nodes than field DOFs
                // Only map DOFs that exist in mesh - rest remain InvalidIndex
                for (int j = 0; j < geoNodes; ++j) {
                    elemDofs_[base + j] = elem.vertex(j);
                }
                // DOFs beyond geoNodes remain InvalidIndex
            }
            else {
                // Superparametric: field order < geo order
                // Mesh has more vertices than field DOFs
                // DOFs are only the corner vertices (first numDofs of them)
                for (int j = 0; j < fieldDofs; ++j) {
                    elemDofs_[base + j] = elem.vertex(j);
                }
            }
        }

        // Build boundary element DOF mapping
        for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
            const Element& bdrElem = mesh_->bdrElement(bdrIdx);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
            if (!refElem)
                continue;

            const int fieldDofs = refElem->numDofs();
            const int geoOrder = bdrElem.order();

            const Index base = bdrIdx * maxDofsPerBdrElem_;

            if (fieldOrder == geoOrder) {
                for (int j = 0; j < fieldDofs; ++j) {
                    bdrElemDofs_[base + j] = bdrElem.vertex(j);
                }
            }
            else if (fieldOrder > geoOrder) {
                int geoNodes = static_cast<int>(bdrElem.vertices().size());
                for (int j = 0; j < geoNodes; ++j) {
                    bdrElemDofs_[base + j] = bdrElem.vertex(j);
                }
            }
            else {
                for (int j = 0; j < fieldDofs; ++j) {
                    bdrElemDofs_[base + j] = bdrElem.vertex(j);
                }
            }
        }

        // Total DOFs = mesh vertices (one per vertex for H1)
        numDofs_ = mesh_->numVertices();

        // Multiply by vdim for vector fields
        numDofs_ = numDofs_ * vdim_;
    }

} // namespace mpfem
