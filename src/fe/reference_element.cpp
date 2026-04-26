#include "fe/reference_element.hpp"
#include "core/exception.hpp"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <iostream> // TODO: remove debug

namespace mpfem {

    namespace {

        basix::cell::type toBasixCell(Geometry geom)
        {
            switch (geom) {
            case Geometry::Segment:
                return basix::cell::type::interval;
            case Geometry::Triangle:
                return basix::cell::type::triangle;
            case Geometry::Square:
                return basix::cell::type::quadrilateral;
            case Geometry::Tetrahedron:
                return basix::cell::type::tetrahedron;
            case Geometry::Cube:
                return basix::cell::type::hexahedron;
            default:
                MPFEM_THROW(Exception, "Unsupported geometry for ReferenceElement");
            }
        }

        basix::element::family toBasixFamily(BasisType bt)
        {
            switch (bt) {
            case BasisType::H1:
                return basix::element::family::P;
            case BasisType::ND:
                return basix::element::family::N1E;
            case BasisType::RT:
                return basix::element::family::RT;
            default:
                MPFEM_THROW(Exception, "Unsupported BasisType for ReferenceElement");
            }
        }

    } // anonymous namespace

    // =============================================================================
    // DofLayout implementation
    // =============================================================================

    // =============================================================================
    // ReferenceElement implementation
    // =============================================================================

    ReferenceElement::ReferenceElement(Geometry geom, int order, BasisType basisType, int vdim)
        : geometry_(geom), order_(order), basisType_(basisType), vdim_(vdim)
    {
        initialize();
    }

    void ReferenceElement::initialize()
    {
        auto cell = toBasixCell(geometry_);
        auto family = toBasixFamily(basisType_);

        basixElement_ = std::make_unique<basix::FiniteElement<double>>(
            basix::create_element<double>(
                family,
                cell,
                order_,
                basix::element::lagrange_variant::equispaced,
                basix::element::dpc_variant::unset,
                false));

        // Build permutation for tensor-product elements
        buildPermutation();

        // Create quadrature rule with order 2*order for exact integration
        quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));

        // Precompute basis values and derivatives at all quadrature points
        precomputeShapeValues();
    }

    void ReferenceElement::buildPermutation()
    {
        const auto& entity_dofs = basixElement_->entity_dofs();

        // Extract DOF layout from entity_dofs (needed for ALL element types)
        // entity_dofs is indexed as: entity_dofs[dim][entity_index][dof_index]
        // dofLayout stores DOFs PER entity, not total DOFs
        dofLayout_ = DofLayout {};
        if (entity_dofs.size() > 0 && !entity_dofs[0].empty()) {
            // DOFs per vertex (not total vertex DOFs)
            dofLayout_.numVertexDofs = static_cast<int>(entity_dofs[0][0].size());
        }
        if (entity_dofs.size() > 1 && !entity_dofs[1].empty()) {
            // DOFs per edge (not total edge DOFs)
            dofLayout_.numEdgeDofs = static_cast<int>(entity_dofs[1][0].size());
        }
        if (entity_dofs.size() > 2 && !entity_dofs[2].empty()) {
            // DOFs per face (not total face DOFs)
            dofLayout_.numFaceDofs = static_cast<int>(entity_dofs[2][0].size());
        }
        if (entity_dofs.size() > 3 && !entity_dofs[3].empty()) {
            // DOFs per cell (not total cell DOFs)
            dofLayout_.numVolumeDofs = static_cast<int>(entity_dofs[3][0].size());
        }

        // Simplices use CCW ordering already (identity permutation)
        if (geom::isSimplex(geometry_)) {
            int ndofs = basixElement_->dim();
            ccwToBasix_.resize(ndofs);
            basixToCcw_.resize(ndofs);
            for (int i = 0; i < ndofs; ++i) {
                ccwToBasix_[i] = i;
                basixToCcw_[i] = i;
            }
            return;
        }

        // For tensor-product elements, we need to map BASIX → CCW ordering
        // Use entity_dofs to determine the mapping

        if (geometry_ == Geometry::Square) {
            // Square vertex ordering:
            // CCW: 0(0,0), 1(1,0), 2(1,1), 3(0,1)
            // BASIX: 0(0,0), 1(1,0), 2(0,1), 3(1,1)
            // Permutation: BASIX[2] ↔ CCW[2], BASIX[3] ↔ CCW[3]

            if (order_ == 1) {
                // Only 4 vertex DOFs
                basixToCcw_ = {0, 1, 3, 2}; // Swap indices 2 and 3
                ccwToBasix_ = {0, 1, 3, 2};
            }
            else if (order_ == 2) {
                // 9 DOFs: 4 vertices + 4 edge midpoints + 1 center
                // Use identity for now
                basixToCcw_.resize(9);
                ccwToBasix_.resize(9);
                for (int i = 0; i < 9; ++i) {
                    basixToCcw_[i] = i;
                    ccwToBasix_[i] = i;
                }
            }
        }
        else if (geometry_ == Geometry::Cube) {
            // Cube vertex ordering:
            // CCW (from geom.hpp):
            //   0(0,0,0), 1(1,0,0), 2(1,1,0), 3(0,1,0), 4(0,0,1), 5(1,0,1), 6(1,1,1), 7(0,1,1)
            // BASIX:
            //   0(0,0,0), 1(1,0,0), 2(0,1,0), 3(1,1,0), 4(0,0,1), 5(1,0,1), 6(0,1,1), 7(1,1,1)
            // Permutation: BASIX[2] ↔ CCW[2], BASIX[3] ↔ CCW[3], BASIX[6] ↔ CCW[6], BASIX[7] ↔ CCW[7]

            if (order_ == 1) {
                basixToCcw_ = {0, 1, 3, 2, 4, 5, 7, 6};
                ccwToBasix_ = {0, 1, 3, 2, 4, 5, 7, 6};
            }
            else if (order_ == 2) {
                // For order 2, we have 27 DOFs
                // Vertices: 0-7 (permuted)
                // Edges: 8-19 (12 edge midpoints)
                // Faces: 20-25 (6 face centers)
                // Cell: 26 (center)

                // For now, use identity for non-vertices
                int ndofs = basixElement_->dim();
                basixToCcw_.resize(ndofs);
                ccwToBasix_.resize(ndofs);
                for (int i = 0; i < 8; ++i) {
                    basixToCcw_[i] = i;
                    ccwToBasix_[i] = i;
                }
                // Apply vertex permutation to first 8
                basixToCcw_[0] = 0;
                basixToCcw_[1] = 1;
                basixToCcw_[2] = 3;
                basixToCcw_[3] = 2;
                basixToCcw_[4] = 4;
                basixToCcw_[5] = 5;
                basixToCcw_[6] = 7;
                basixToCcw_[7] = 6;
                ccwToBasix_[0] = 0;
                ccwToBasix_[1] = 1;
                ccwToBasix_[2] = 3;
                ccwToBasix_[3] = 2;
                ccwToBasix_[4] = 4;
                ccwToBasix_[5] = 5;
                ccwToBasix_[6] = 7;
                ccwToBasix_[7] = 6;
                for (int i = 8; i < ndofs; ++i) {
                    basixToCcw_[i] = i;
                    ccwToBasix_[i] = i;
                }
            }
        }
        else if (geometry_ == Geometry::Segment) {
            // Segment: vertices are 0 and 1 in both orderings
            int ndofs = basixElement_->dim();
            basixToCcw_.resize(ndofs);
            ccwToBasix_.resize(ndofs);
            for (int i = 0; i < ndofs; ++i) {
                basixToCcw_[i] = i;
                ccwToBasix_[i] = i;
            }
        }
    }

    void ReferenceElement::precomputeShapeValues()
    {
        const int nq = quadrature_.size();
        if (nq == 0)
            return;

        const int d = dim();
        cachedShapeValues_.resize(nq);
        cachedDerivatives_.resize(nq);

        // Extract quadrature points to flat array
        std::vector<double> points_data(static_cast<std::size_t>(nq) * d);
        for (int q = 0; q < nq; ++q) {
            auto xi = quadrature_[q].getXi();
            points_data[q * d + 0] = xi.x();
            if (d > 1)
                points_data[q * d + 1] = xi.y();
            if (d > 2)
                points_data[q * d + 2] = xi.z();
        }

        // BASIX tabulate: returns pair (data, shape)
        auto [tab_data, tab_shape] = basixElement_->tabulate(
            1, points_data, {static_cast<std::size_t>(nq), static_cast<std::size_t>(d)});

        int ndofs = basixElement_->dim();

        // Compute value size from value_shape - scalar elements have empty shape (size 1)
        const auto& value_shape = basixElement_->value_shape();
        int vs = 1;
        for (auto s : value_shape) {
            vs *= static_cast<int>(s);
        }

        // BASIX returns data in (deriv, pt, dof, val) order, flattened
        // Deriv 0 = shape values, Deriv 1+ = derivatives in each dimension

        for (int q = 0; q < nq; ++q) {
            cachedShapeValues_[q].resize(ndofs, vs);
            cachedDerivatives_[q].resize(ndofs, d * vs);

            for (int b_dof = 0; b_dof < ndofs; ++b_dof) {
                int ccw_dof = basixToCcw_[b_dof]; // Apply permutation!

                // Shape values (deriv=0)
                for (int v = 0; v < vs; ++v) {
                    std::size_t idx = 0 * nq * ndofs * vs + q * ndofs * vs + b_dof * vs + v;
                    cachedShapeValues_[q](ccw_dof, v) = tab_data[idx];
                }

                // Derivatives (deriv=1..d)
                for (int deriv = 0; deriv < d; ++deriv) {
                    for (int v = 0; v < vs; ++v) {
                        std::size_t idx = (deriv + 1) * nq * ndofs * vs + q * ndofs * vs + b_dof * vs + v;
                        cachedDerivatives_[q](ccw_dof, deriv * vs + v) = tab_data[idx];
                    }
                }
            }
        }
    }

    int ReferenceElement::numDofs() const
    {
        return basixElement_->dim();
    }

    DofLayout ReferenceElement::dofLayout() const
    {
        return dofLayout_;
    }

    std::vector<int> ReferenceElement::faceDofs(int faceIdx) const
    {
        const auto& entity_dofs = basixElement_->entity_dofs();
        // entity_dofs[2] = face DOFs
        if (entity_dofs.size() <= 2) {
            return {};
        }
        if (faceIdx < 0 || faceIdx >= static_cast<int>(entity_dofs[2].size())) {
            return {};
        }
        const auto& b_dofs = entity_dofs[2][faceIdx];

        std::vector<int> result(b_dofs.size());
        for (size_t i = 0; i < b_dofs.size(); ++i) {
            result[i] = basixToCcw_[b_dofs[i]];
        }
        return result;
    }

    std::vector<int> ReferenceElement::facetDofs(int facetIdx) const
    {
        const int d = dim();
        if (d == 1)
            return {facetIdx}; // Point
        if (d == 2)
            return edgeDofs(facetIdx);
        if (d == 3)
            return faceDofs(facetIdx);
        return {};
    }

    std::vector<int> ReferenceElement::edgeDofs(int edgeIdx) const
    {
        const auto& entity_dofs = basixElement_->entity_dofs();
        // entity_dofs[1] = edge DOFs
        if (entity_dofs.size() <= 1) {
            return {};
        }
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(entity_dofs[1].size())) {
            return {};
        }
        const auto& b_dofs = entity_dofs[1][edgeIdx];

        std::vector<int> result(b_dofs.size());
        for (size_t i = 0; i < b_dofs.size(); ++i) {
            result[i] = basixToCcw_[b_dofs[i]];
        }
        return result;
    }

    std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const
    {
        return geom::edgeVertices(geometry_, edgeIdx);
    }

    void ReferenceElement::evalShape(const Vector3& xi, ShapeMatrix& shape) const
    {
        const int d = dim();
        const int nd = numDofs();

        // Compute value size from value_shape - scalar elements have empty shape (size 1)
        const auto& value_shape = basixElement_->value_shape();
        int vs = 1;
        for (auto s : value_shape) {
            vs *= static_cast<int>(s);
        }

        // BASIX tabulate at single point
        std::vector<double> point_data(static_cast<std::size_t>(d));
        point_data[0] = xi.x();
        if (d > 1)
            point_data[1] = xi.y();
        if (d > 2)
            point_data[2] = xi.z();

        auto [tab_data, tab_shape] = basixElement_->tabulate(
            0, point_data, {1, static_cast<std::size_t>(d)});

        shape.resize(nd, vs);

        // Apply permutation ccwToBasix_ and extract shape values
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = ccwToBasix_[b_dof];
            for (int v = 0; v < vs; ++v) {
                std::size_t idx = b_dof * vs + v;
                shape(ccw_dof, v) = tab_data[idx];
            }
        }
    }

    void ReferenceElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        const int d = dim();
        const int nd = numDofs();

        // Compute value size from value_shape - scalar elements have empty shape (size 1)
        const auto& value_shape = basixElement_->value_shape();
        int vs = 1;
        for (auto s : value_shape) {
            vs *= static_cast<int>(s);
        }

        // BASIX tabulate at single point
        std::vector<double> point_data(static_cast<std::size_t>(d));
        point_data[0] = xi.x();
        if (d > 1)
            point_data[1] = xi.y();
        if (d > 2)
            point_data[2] = xi.z();

        auto [tab_data, tab_shape] = basixElement_->tabulate(
            1, point_data, {1, static_cast<std::size_t>(d)});

        derivatives.resize(nd, d * vs);

        // BASIX returns data in (deriv, pt, dof, val) order, flattened
        // Deriv 0 = shape values, Deriv 1+ = derivatives in each dimension
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = ccwToBasix_[b_dof];
            for (int deriv = 0; deriv < d; ++deriv) {
                for (int v = 0; v < vs; ++v) {
                    std::size_t idx = (deriv + 1) * nd * vs + b_dof * vs + v;
                    derivatives(ccw_dof, deriv * vs + v) = tab_data[idx];
                }
            }
        }
    }

} // namespace mpfem
