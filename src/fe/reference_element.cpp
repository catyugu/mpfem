#include "fe/reference_element.hpp"
#include "core/exception.hpp"
#include <algorithm>
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

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

        // Centralized definition of CCW to Tensor-Product vertex permutation
        std::vector<int> getBaseVertexPermutation(Geometry geom)
        {
            switch (geom) {
            case Geometry::Segment:
                return {0, 1};
            case Geometry::Triangle:
                return {0, 1, 2};
            case Geometry::Tetrahedron:
                return {0, 1, 2, 3};
            case Geometry::Square:
                return {0, 1, 3, 2};
            case Geometry::Cube:
                return {0, 1, 3, 2, 4, 5, 7, 6};
            default:
                MPFEM_THROW(Exception, "Unsupported geometry for permutation");
            }
        }
    }

    // =======================================================================
    // Thread-Safe Global Flyweight Cache
    // =======================================================================
    const ReferenceElement* ReferenceElement::get(Geometry geom, int order, BasisType basisType, int vdim)
    {
        using Key = std::tuple<Geometry, int, BasisType, int>;
        static std::map<Key, std::unique_ptr<ReferenceElement>> cache;
        static std::mutex cache_mutex;

        Key key {geom, order, basisType, vdim};
        std::lock_guard<std::mutex> lock(cache_mutex);

        auto it = cache.find(key);
        if (it == cache.end()) {
            auto ptr = std::unique_ptr<ReferenceElement>(
                new ReferenceElement(geom, order, basisType, vdim));
            it = cache.emplace(key, std::move(ptr)).first;
        }
        return it->second.get();
    }

    ReferenceElement::ReferenceElement(Geometry geom, int order, BasisType basisType, int vdim)
        : geometry_(geom), order_(order), basisType_(basisType), vdim_(vdim)
    {
        initialize();
    }

    void ReferenceElement::initialize()
    {
        basixElement_ = std::make_unique<basix::FiniteElement<double>>(
            basix::create_element<double>(
                toBasixFamily(basisType_), toBasixCell(geometry_), order_,
                basix::element::lagrange_variant::equispaced,
                basix::element::dpc_variant::unset, false));

        buildPermutation();
        quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));
        precomputeShapeValues();
    }

    void ReferenceElement::buildPermutation()
    {
        const int ndofs = basixElement_->dim();
        const auto& entity_dofs = basixElement_->entity_dofs();
        const auto topo = basix::cell::topology(basixElement_->cell_type());

        // Configure Layout Properties
        dofLayout_ = DofLayout {};
        if (entity_dofs.size() > 0 && !entity_dofs[0].empty())
            dofLayout_.numVertexDofs = entity_dofs[0][0].size();
        if (entity_dofs.size() > 1 && !entity_dofs[1].empty())
            dofLayout_.numEdgeDofs = entity_dofs[1][0].size();
        if (entity_dofs.size() > 2 && !entity_dofs[2].empty())
            dofLayout_.numFaceDofs = entity_dofs[2][0].size();
        if (entity_dofs.size() > 3 && !entity_dofs[3].empty())
            dofLayout_.numVolumeDofs = entity_dofs[3][0].size();

        ccwToBasix_.assign(ndofs, -1);
        basixToCcw_.assign(ndofs, -1);
        int current_ccw_dof = 0;

        // --- Core Utility 1: Sequential DOF Allocator ---
        // Registers internal DOFs and handles directional reversing gracefully
        auto register_dofs = [&](const std::vector<int>& b_dofs, bool reversed = false) {
            std::vector<int> mapped_interior;
            auto consume = [&](int d) {
                if (basixToCcw_[d] == -1) {
                    basixToCcw_[d] = current_ccw_dof;
                    ccwToBasix_[current_ccw_dof] = d;
                    mapped_interior.push_back(current_ccw_dof++);
                }
            };
            if (reversed) {
                for (auto it = b_dofs.rbegin(); it != b_dofs.rend(); ++it)
                    consume(*it);
            }
            else {
                for (int d : b_dofs)
                    consume(d);
            }
            return mapped_interior;
        };

        const std::vector<int> v_map = getBaseVertexPermutation(geometry_);
        const int num_verts = geom::numVertices(geometry_);
        const int num_edges = geom::numEdges(geometry_);
        const int num_faces = geom::numFaces(geometry_);

        std::vector<std::vector<int>> v_interior(num_verts);
        std::vector<std::vector<int>> e_interior(num_edges);
        std::vector<std::vector<int>> f_interior(std::max(1, num_faces));

        // --- Core Utility 2: Robust Topological Matcher ---
        // Uses set-intersection to guarantee shape matching regardless of vertex sequence.
        // Returns {matched_basix_index, needs_reverse_flag}
        auto match_entity = [&](int dim, const std::vector<int>& ccw_verts, std::vector<bool>& consumed) -> std::pair<int, bool> {
            if (dim >= topo.size() || topo[dim].empty())
                return {-1, false};

            std::set<int> b_target;
            for (int v : ccw_verts)
                b_target.insert(v_map[v]);

            // Attempt Exact Topological Set Match
            for (size_t i = 0; i < topo[dim].size(); ++i) {
                if (consumed[i])
                    continue;
                std::set<int> b_candidate(topo[dim][i].begin(), topo[dim][i].end());
                if (b_target == b_candidate) {
                    consumed[i] = true;
                    bool reversed = (dim == 1 && topo[dim][i][0] != v_map[ccw_verts[0]]);
                    return {static_cast<int>(i), reversed};
                }
            }
            // Greedy Fallback for unmapped geometry quirks
            for (size_t i = 0; i < topo[dim].size(); ++i) {
                if (!consumed[i]) {
                    consumed[i] = true;
                    return {static_cast<int>(i), false};
                }
            }
            return {-1, false};
        };

        // =======================================================================
        // PHASE 1: Entity Interior DOF Extraction (UFC Standard)
        // =======================================================================

        // 0D: Vertices
        for (int i = 0; i < num_verts; ++i) {
            if (!entity_dofs.empty())
                v_interior[i] = register_dofs(entity_dofs[0][v_map[i]]);
        }

        // 1D: Edges
        if (entity_dofs.size() > 1 && num_edges > 0) {
            std::vector<bool> consumed_b_edges(topo[1].size(), false);
            for (int e = 0; e < num_edges; ++e) {
                auto [v0, v1] = geom::edgeVertices(geometry_, e);
                auto [match_b, reversed] = match_entity(1, {v0, v1}, consumed_b_edges);
                if (match_b != -1)
                    e_interior[e] = register_dofs(entity_dofs[1][match_b], reversed);
            }
        }

        // 2D: Faces
        if (entity_dofs.size() > 2) {
            if (dim() == 2 && topo[2].size() == 1) {
                f_interior[0] = register_dofs(entity_dofs[2][0]); // 2D interior acts as face 0
            }
            else if (dim() == 3 && num_faces > 0) {
                std::vector<bool> consumed_b_faces(topo[2].size(), false);
                for (int f = 0; f < num_faces; ++f) {
                    auto f_verts = geom::faceVertices(geometry_, f);
                    auto [match_b, _] = match_entity(2, f_verts, consumed_b_faces);
                    if (match_b != -1)
                        f_interior[f] = register_dofs(entity_dofs[2][match_b]);
                }
            }
        }

        // 3D/Fallback: Cell Volumes
        std::vector<int> unmapped_pool;
        for (int d = 0; d < ndofs; ++d) {
            if (basixToCcw_[d] == -1)
                unmapped_pool.push_back(d);
        }
        register_dofs(unmapped_pool);

        // =======================================================================
        // PHASE 2: Assemble Full Entity Closures
        // =======================================================================

        edgeDofs_.resize(num_edges);
        for (int e = 0; e < num_edges; ++e) {
            auto [v0, v1] = geom::edgeVertices(geometry_, e);
            auto& dofs = edgeDofs_[e];
            dofs.insert(dofs.end(), v_interior[v0].begin(), v_interior[v0].end());
            dofs.insert(dofs.end(), v_interior[v1].begin(), v_interior[v1].end());
            dofs.insert(dofs.end(), e_interior[e].begin(), e_interior[e].end());
        }

        faceDofs_.resize(num_faces);
        for (int f = 0; f < num_faces; ++f) {
            auto f_verts = geom::faceVertices(geometry_, f);
            std::set<int> f_v_set(f_verts.begin(), f_verts.end());
            auto& dofs = faceDofs_[f];

            // 1. Add Vertices on closure
            for (int v : f_verts)
                dofs.insert(dofs.end(), v_interior[v].begin(), v_interior[v].end());

            // 2. Add Edges on closure
            for (int e = 0; e < num_edges; ++e) {
                auto [v0, v1] = geom::edgeVertices(geometry_, e);
                if (f_v_set.count(v0) && f_v_set.count(v1)) {
                    dofs.insert(dofs.end(), e_interior[e].begin(), e_interior[e].end());
                }
            }
            // 3. Add Face interior
            dofs.insert(dofs.end(), f_interior[f].begin(), f_interior[f].end());
        }
    }

    void ReferenceElement::precomputeShapeValues()
    {
        const int nq = quadrature_.size();
        if (nq == 0)
            return;

        const int d = dim();
        int vs = 1;
        for (auto s : basixElement_->value_shape())
            vs *= static_cast<int>(s);
        int nd = basixElement_->dim();

        std::vector<double> points_data(nq * d);
        for (int q = 0; q < nq; ++q) {
            auto xi = quadrature_[q].getXi();
            points_data[q * d + 0] = xi.x();
            if (d > 1)
                points_data[q * d + 1] = xi.y();
            if (d > 2)
                points_data[q * d + 2] = xi.z();
        }

        auto [tab_data, _] = basixElement_->tabulate(1, points_data, {static_cast<std::size_t>(nq), static_cast<std::size_t>(d)});

        cachedShapeValues_.resize(nq);
        cachedDerivatives_.resize(nq);

        for (int q = 0; q < nq; ++q) {
            cachedShapeValues_[q].resize(nd, vs);
            cachedDerivatives_[q].resize(nd, d * vs);

            for (int b_dof = 0; b_dof < nd; ++b_dof) {
                int ccw_dof = basixToCcw_[b_dof];

                for (int v = 0; v < vs; ++v) {
                    cachedShapeValues_[q](ccw_dof, v) = tab_data[0 * nq * nd * vs + q * nd * vs + b_dof * vs + v];
                }
                for (int deriv = 0; deriv < d; ++deriv) {
                    for (int v = 0; v < vs; ++v) {
                        cachedDerivatives_[q](ccw_dof, deriv * vs + v) = tab_data[(deriv + 1) * nq * nd * vs + q * nd * vs + b_dof * vs + v];
                    }
                }
            }
        }
    }

    int ReferenceElement::numDofs() const { return basixElement_->dim(); }
    DofLayout ReferenceElement::dofLayout() const { return dofLayout_; }

    std::vector<int> ReferenceElement::faceDofs(int faceIdx) const
    {
        return (faceIdx >= 0 && faceIdx < static_cast<int>(faceDofs_.size())) ? faceDofs_[faceIdx] : std::vector<int> {};
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
        return (edgeIdx >= 0 && edgeIdx < static_cast<int>(edgeDofs_.size())) ? edgeDofs_[edgeIdx] : std::vector<int> {};
    }

    std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const { return geom::edgeVertices(geometry_, edgeIdx); }

    void ReferenceElement::evalShape(const Vector3& xi, ShapeMatrix& shape) const
    {
        const int d = dim();
        int vs = 1;
        for (auto s : basixElement_->value_shape())
            vs *= static_cast<int>(s);
        int nd = basixElement_->dim();

        std::vector<double> pt(d);
        pt[0] = xi.x();
        if (d > 1)
            pt[1] = xi.y();
        if (d > 2)
            pt[2] = xi.z();

        auto [tab_data, _] = basixElement_->tabulate(0, pt, {1, static_cast<std::size_t>(d)});

        shape.resize(nd, vs);
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = basixToCcw_[b_dof];
            for (int v = 0; v < vs; ++v)
                shape(ccw_dof, v) = tab_data[b_dof * vs + v];
        }
    }

    void ReferenceElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        const int d = dim();
        int vs = 1;
        for (auto s : basixElement_->value_shape())
            vs *= static_cast<int>(s);
        int nd = basixElement_->dim();

        std::vector<double> pt(d);
        pt[0] = xi.x();
        if (d > 1)
            pt[1] = xi.y();
        if (d > 2)
            pt[2] = xi.z();

        auto [tab_data, _] = basixElement_->tabulate(1, pt, {1, static_cast<std::size_t>(d)});

        derivatives.resize(nd, d * vs);
        for (int b_dof = 0; b_dof < nd; ++b_dof) {
            int ccw_dof = basixToCcw_[b_dof];
            for (int deriv = 0; deriv < d; ++deriv) {
                for (int v = 0; v < vs; ++v) {
                    derivatives(ccw_dof, deriv * vs + v) = tab_data[(deriv + 1) * nd * vs + b_dof * vs + v];
                }
            }
        }
    }

    std::vector<Vector3> ReferenceElement::interpolationPoints() const
    {
        auto [b_pts, _] = basixElement_->points();
        std::vector<Vector3> ccw_points(numDofs());
        int d = dim();

        for (int ccw_dof = 0; ccw_dof < numDofs(); ++ccw_dof) {
            int b_dof = ccwToBasix_[ccw_dof];
            ccw_points[ccw_dof] = Vector3(
                b_pts[b_dof * d + 0],
                (d > 1) ? b_pts[b_dof * d + 1] : 0.0,
                (d > 2) ? b_pts[b_dof * d + 2] : 0.0);
        }
        return ccw_points;
    }

} // namespace mpfem