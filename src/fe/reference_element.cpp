#include "fe/reference_element.hpp"
#include "core/exception.hpp"
#include <algorithm>
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <set>

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
    }

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
                family, cell, order_,
                basix::element::lagrange_variant::equispaced,
                basix::element::dpc_variant::unset,
                false));

        buildPermutation();
        quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));
        precomputeShapeValues();
    }

    void ReferenceElement::buildPermutation()
    {
        auto cell_type = basixElement_->cell_type();
        auto topo = basix::cell::topology(cell_type);
        const auto& entity_dofs = basixElement_->entity_dofs();

        dofLayout_ = DofLayout {};
        if (entity_dofs.size() > 0 && !entity_dofs[0].empty())
            dofLayout_.numVertexDofs = entity_dofs[0][0].size();
        if (entity_dofs.size() > 1 && !entity_dofs[1].empty())
            dofLayout_.numEdgeDofs = entity_dofs[1][0].size();
        if (entity_dofs.size() > 2 && !entity_dofs[2].empty())
            dofLayout_.numFaceDofs = entity_dofs[2][0].size();
        if (entity_dofs.size() > 3 && !entity_dofs[3].empty())
            dofLayout_.numVolumeDofs = entity_dofs[3][0].size();

        int ndofs = basixElement_->dim();
        ccwToBasix_.assign(ndofs, -1);
        basixToCcw_.assign(ndofs, -1);

        int current_ccw_dof = 0;

        // =======================================================================
        // Step 1: Base Vertex Mapping (CCW -> BASIX)
        // =======================================================================
        std::vector<int> v_map;
        switch (geometry_) {
        case Geometry::Segment:
            v_map = {0, 1};
            break;
        case Geometry::Triangle:
            v_map = {0, 1, 2};
            break;
        case Geometry::Tetrahedron:
            v_map = {0, 1, 2, 3};
            break;
        case Geometry::Square:
            v_map = {0, 1, 3, 2};
            break; // Swap for tensor-product
        case Geometry::Cube:
            v_map = {0, 1, 3, 2, 4, 5, 7, 6};
            break;
        default:
            MPFEM_THROW(Exception, "Unsupported geometry for Mapping");
        }

        // Map Vertex DOFs
        int num_verts = geom::numVertices(geometry_);
        for (int i = 0; i < num_verts; ++i) {
            int b_v = v_map[i];
            for (int b_dof : entity_dofs[0][b_v]) {
                basixToCcw_[b_dof] = current_ccw_dof;
                ccwToBasix_[current_ccw_dof++] = b_dof;
            }
        }

        // =======================================================================
        // Step 2: Dynamically Map Edges & Handle Reverse Orientations
        // =======================================================================
        int num_edges = geom::numEdges(geometry_);
        edgeDofs_.resize(num_edges);
        if (entity_dofs.size() > 1) {
            for (int e_ccw = 0; e_ccw < num_edges; ++e_ccw) {
                auto [v0_ccw, v1_ccw] = geom::edgeVertices(geometry_, e_ccw);
                int b_v0 = v_map[v0_ccw];
                int b_v1 = v_map[v1_ccw];

                int match_b = -1;
                bool reversed = false;

                // Search for matching edge in BASIX topology
                for (size_t e_b = 0; e_b < topo[1].size(); ++e_b) {
                    if (topo[1][e_b][0] == b_v0 && topo[1][e_b][1] == b_v1) {
                        match_b = e_b;
                        reversed = false;
                        break;
                    }
                    if (topo[1][e_b][0] == b_v1 && topo[1][e_b][1] == b_v0) {
                        match_b = e_b;
                        reversed = true;
                        break;
                    }
                }

                if (match_b != -1) {
                    const auto& edofs = entity_dofs[1][match_b];
                    // CRITICAL FIX: Reverse high-order edge DOFs if edge traversal directions misalign
                    if (reversed) {
                        for (auto it = edofs.rbegin(); it != edofs.rend(); ++it) {
                            basixToCcw_[*it] = current_ccw_dof;
                            ccwToBasix_[current_ccw_dof] = *it;
                            edgeDofs_[e_ccw].push_back(current_ccw_dof++);
                        }
                    }
                    else {
                        for (int d : edofs) {
                            basixToCcw_[d] = current_ccw_dof;
                            ccwToBasix_[current_ccw_dof] = d;
                            edgeDofs_[e_ccw].push_back(current_ccw_dof++);
                        }
                    }
                }
            }
        }

        // =======================================================================
        // Step 3: Dynamically Map Faces
        // =======================================================================
        int num_faces = geom::numFaces(geometry_);
        faceDofs_.resize(num_faces);
        if (entity_dofs.size() > 2) {
            for (int f_ccw = 0; f_ccw < num_faces; ++f_ccw) {
                auto f_verts_ccw = geom::faceVertices(geometry_, f_ccw);
                std::set<int> f_verts_b_set;
                for (int v : f_verts_ccw)
                    f_verts_b_set.insert(v_map[v]);

                int match_b = -1;
                for (size_t f_b = 0; f_b < topo[2].size(); ++f_b) {
                    std::set<int> b_set(topo[2][f_b].begin(), topo[2][f_b].end());
                    if (f_verts_b_set == b_set) {
                        match_b = f_b;
                        break;
                    }
                }

                if (match_b != -1) {
                    for (int d : entity_dofs[2][match_b]) {
                        basixToCcw_[d] = current_ccw_dof;
                        ccwToBasix_[current_ccw_dof] = d;
                        faceDofs_[f_ccw].push_back(current_ccw_dof++);
                    }
                }
            }
        }

        // =======================================================================
        // Step 4: Map Volume DOFs
        // =======================================================================
        if (entity_dofs.size() > 3 && !entity_dofs[3].empty()) {
            for (int d : entity_dofs[3][0]) {
                basixToCcw_[d] = current_ccw_dof;
                ccwToBasix_[current_ccw_dof++] = d;
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

        std::vector<double> points_data(nq * d);
        for (int q = 0; q < nq; ++q) {
            auto xi = quadrature_[q].getXi();
            points_data[q * d + 0] = xi.x();
            if (d > 1)
                points_data[q * d + 1] = xi.y();
            if (d > 2)
                points_data[q * d + 2] = xi.z();
        }

        auto [tab_data, tab_shape] = basixElement_->tabulate(1, points_data, {static_cast<std::size_t>(nq), static_cast<std::size_t>(d)});

        int ndofs = basixElement_->dim();
        const auto& value_shape = basixElement_->value_shape();
        int vs = 1;
        for (auto s : value_shape)
            vs *= static_cast<int>(s);

        for (int q = 0; q < nq; ++q) {
            cachedShapeValues_[q].resize(ndofs, vs);
            cachedDerivatives_[q].resize(ndofs, d * vs);

            for (int b_dof = 0; b_dof < ndofs; ++b_dof) {
                int ccw_dof = basixToCcw_[b_dof];

                for (int v = 0; v < vs; ++v) {
                    std::size_t idx = 0 * nq * ndofs * vs + q * ndofs * vs + b_dof * vs + v;
                    cachedShapeValues_[q](ccw_dof, v) = tab_data[idx];
                }

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
        if (faceIdx < 0 || faceIdx >= static_cast<int>(faceDofs_.size()))
            return {};
        return faceDofs_[faceIdx];
    }

    std::vector<int> ReferenceElement::facetDofs(int facetIdx) const
    {
        const int d = dim();
        if (d == 1)
            return {facetIdx};
        if (d == 2)
            return edgeDofs(facetIdx);
        if (d == 3)
            return faceDofs(facetIdx);
        return {};
    }

    std::vector<int> ReferenceElement::edgeDofs(int edgeIdx) const
    {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(edgeDofs_.size()))
            return {};
        return edgeDofs_[edgeIdx];
    }

    std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const
    {
        return geom::edgeVertices(geometry_, edgeIdx);
    }

    void ReferenceElement::evalShape(const Vector3& xi, ShapeMatrix& shape) const
    {
        const int d = dim();
        const int nd = numDofs();

        int vs = 1;
        for (auto s : basixElement_->value_shape())
            vs *= static_cast<int>(s);

        std::vector<double> point_data(d);
        point_data[0] = xi.x();
        if (d > 1)
            point_data[1] = xi.y();
        if (d > 2)
            point_data[2] = xi.z();

        auto [tab_data, tab_shape] = basixElement_->tabulate(0, point_data, {1, static_cast<std::size_t>(d)});

        shape.resize(nd, vs);
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = basixToCcw_[b_dof];
            for (int v = 0; v < vs; ++v) {
                shape(ccw_dof, v) = tab_data[b_dof * vs + v];
            }
        }
    }

    void ReferenceElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        const int d = dim();
        const int nd = numDofs();

        int vs = 1;
        for (auto s : basixElement_->value_shape())
            vs *= static_cast<int>(s);

        std::vector<double> point_data(d);
        point_data[0] = xi.x();
        if (d > 1)
            point_data[1] = xi.y();
        if (d > 2)
            point_data[2] = xi.z();

        auto [tab_data, tab_shape] = basixElement_->tabulate(1, point_data, {1, static_cast<std::size_t>(d)});

        derivatives.resize(nd, d * vs);
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = basixToCcw_[b_dof];
            for (int deriv = 0; deriv < d; ++deriv) {
                for (int v = 0; v < vs; ++v) {
                    std::size_t idx = (deriv + 1) * nd * vs + b_dof * vs + v;
                    derivatives(ccw_dof, deriv * vs + v) = tab_data[idx];
                }
            }
        }
    }

    std::vector<Vector3> ReferenceElement::interpolationPoints() const
    {
        // 彻底丢弃繁杂的几何硬编码点！直接委托给 BASIX 计算坐标，然后按CCW排列输出
        auto [b_pts, b_shape] = basixElement_->points();

        std::vector<Vector3> ccw_points(numDofs());
        int d = dim();

        for (int ccw_dof = 0; ccw_dof < numDofs(); ++ccw_dof) {
            int b_dof = ccwToBasix_[ccw_dof];
            double x = b_pts[b_dof * d + 0];
            double y = (d > 1) ? b_pts[b_dof * d + 1] : 0.0;
            double z = (d > 2) ? b_pts[b_dof * d + 2] : 0.0;
            ccw_points[ccw_dof] = Vector3(x, y, z);
        }

        return ccw_points;
    }

} // namespace mpfem