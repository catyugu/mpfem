#include "assembler.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace {

    using DomainIntegratorMap = std::unordered_map<int, std::vector<size_t>>;

    constexpr mpfem::Real kTripletDropTol = 1e-30;

    inline bool inDomainList(const std::vector<int>& domains, int attr)
    {
        return domains.empty() || std::binary_search(domains.begin(), domains.end(), attr);
    }

    DomainIntegratorMap buildDomainIntegratorMap(const std::vector<std::vector<int>>& domainSets,
        const mpfem::Mesh& mesh)
    {
        std::unordered_set<int> uniqueAttrs;
        uniqueAttrs.reserve(static_cast<size_t>(mesh.numElements()));

        for (mpfem::Index e = 0; e < mesh.numElements(); ++e) {
            uniqueAttrs.insert(mesh.element(e).attribute());
        }

        DomainIntegratorMap attrMap;
        attrMap.reserve(uniqueAttrs.size());

        for (int attr : uniqueAttrs) {
            auto& indices = attrMap[attr];
            indices.reserve(domainSets.size());
            for (size_t k = 0; k < domainSets.size(); ++k) {
                if (inDomainList(domainSets[k], attr)) {
                    indices.push_back(k);
                }
            }
        }

        return attrMap;
    }

    inline bool keepTriplet(mpfem::Real v)
    {
        return std::abs(v) > kTripletDropTol;
    }

} // namespace

namespace mpfem {

    // =============================================================================
    // BilinearFormAssembler
    // =============================================================================

    BilinearFormAssembler::BilinearFormAssembler(const FESpace* fes) : fes_(fes)
    {
        if (fes_ && fes_->numDofs() > 0) {
            mat_.resize(fes_->numDofs(), fes_->numDofs());
        }
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#else
        int nthreads = 1;
#endif
        buffers_.resize(nthreads);

        if (fes_) {
            const Mesh* mesh = fes_->mesh();
            if (mesh) {
                triplets_.reserve(mesh->numElements() * MaxDofsPerElement * MaxDofsPerElement / 2);
            }
        }
    }

    void BilinearFormAssembler::assemble()
    {
        if (!fes_)
            return;
        const Mesh* mesh = fes_->mesh();
        if (!mesh)
            return;

        mat_.setZero();

        const Index numElements = mesh->numElements();
        const int vdim = fes_->vdim();
        const DomainIntegratorMap activeDomains = buildDomainIntegratorMap(domainSets_, *mesh);

        triplets_.clear();
        // 预估 triplet 数量：单元数 × 每单元DOF数² × 向量维度² / 2
        // 对于二阶六面体：27 × 27 × 9 ≈ 6561，除以 2 是因为对称性近似
        const size_t estimatedTriplets = static_cast<size_t>(numElements) * MaxDofsPerElement * MaxDofsPerElement * vdim * vdim / 2;
        triplets_.reserve(estimatedTriplets);

#ifdef _OPENMP
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            ThreadBuffer& buf = buffers_[tid];

            std::vector<SparseMatrix::Triplet> localTriplets;
            localTriplets.reserve(estimatedTriplets / omp_get_num_threads());

            ElementTransform trans;
            trans.setMesh(mesh);

            // Pre-size dynMatrix to maximum possible iTotalDofs once per thread
            const int maxPossibleIvdims = maxIvdim_ > 0 ? maxIvdim_ : 1;
            const int maxDynSize = MaxDofsPerElement * maxPossibleIvdims;
            buf.ensureDynMatrixSize(maxDynSize);
#else
        ThreadBuffer& buf = buffers_[0];
        ElementTransform trans;
        trans.setMesh(mesh);

        // Pre-size dynMatrix to maximum possible iTotalDofs
        const int maxPossibleIvdims = maxIvdim_ > 0 ? maxIvdim_ : 1;
        const int maxDynSize = MaxDofsPerElement * maxPossibleIvdims;
        buf.ensureDynMatrixSize(maxDynSize);
#endif

            // Unified loop body (works for both OpenMP and non-OpenMP)
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
            for (Index e = 0; e < numElements; ++e) {
                const ReferenceElement* ref = fes_->elementRefElement(e);
                if (!ref)
                    continue;
                int nd = ref->numDofs();

                trans.setElement(e);
                const int elemAttr = mesh->element(e).attribute();
                const auto domainIt = activeDomains.find(elemAttr);
                if (domainIt == activeDomains.end() || domainIt->second.empty())
                    continue;

                // Use pre-allocated DOF buffer
                buf.numDofs = nd * vdim;
                fes_->getElementDofs(e, std::span<Index> {buf.dofs.data(), static_cast<size_t>(buf.numDofs)});
                const auto& dofs = buf.dofs;

                int totalDofs = nd * vdim;
                buf.elmatVector.topLeftCorner(totalDofs, totalDofs).setZero();

                for (size_t ki : domainIt->second) {
                    const auto& integ = domainIntegs_[ki];
                    int ivdim = integ->vdim();
                    int iTotalDofs = nd * ivdim;
                    // Only resize if needed (should be rare with pre-sizing)
                    if (buf.dynMatrix.rows() < iTotalDofs) {
                        buf.dynMatrix.resize(iTotalDofs, iTotalDofs);
                    }
                    integ->assembleElementMatrix(*ref, trans, buf.dynMatrix);

                    if (ivdim == 1) {
                        // Scalar integrator: expand to vector field diagonal blocks
                        for (int c = 0; c < vdim; ++c) {
                            buf.elmatVector.block(c * nd, c * nd, nd, nd) += buf.dynMatrix;
                        }
                    }
                    else {
                        // Vector integrator: add directly
                        buf.elmatVector.topLeftCorner(iTotalDofs, iTotalDofs) += buf.dynMatrix;
                    }
                }

                // Pre-compute valid DOF indices to reduce branching in inner loop
                buf.numValidDofs = 0;
                for (int i = 0; i < totalDofs; ++i) {
                    if (dofs[i] != InvalidIndex) {
                        buf.validDofs[buf.numValidDofs++] = i;
                    }
                }

                // Branch-reduced triplet writing: iterate only valid DOFs
                for (int vi = 0; vi < buf.numValidDofs; ++vi) {
                    const int i = buf.validDofs[vi];
                    const Index di = dofs[i];
                    for (int vj = 0; vj < buf.numValidDofs; ++vj) {
                        const int j = buf.validDofs[vj];
                        const Real v = buf.elmatVector(i, j);
                        if (keepTriplet(v)) {
#ifdef _OPENMP
                            localTriplets.emplace_back(di, dofs[j], v);
#else
                        triplets_.emplace_back(di, dofs[j], v);
#endif
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp critical
            {
                triplets_.insert(triplets_.end(), localTriplets.begin(), localTriplets.end());
            }
        }
#endif

        // Boundary integrals (not parallelized - usually small)
        if (!bdrIntegs_.empty()) {
            FacetElementTransform btrans;
            btrans.setMesh(mesh);
            ThreadBuffer& bbuf = buffers_[0]; // Use first thread buffer

            for (Index b = 0; b < mesh->numBdrElements(); ++b) {
                int attr = mesh->bdrElement(b).attribute();

                if (!fes_->isExternalBoundaryId(attr))
                    continue;

                const ReferenceElement* ref = fes_->bdrElementRefElement(b);
                if (!ref)
                    continue;
                int nd = ref->numDofs();

                btrans.setBoundaryElement(b);

                // Use pre-allocated DOF buffer
                bbuf.numDofs = nd * vdim;
                fes_->getBdrElementDofs(b, std::span<Index> {bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
                const auto& dofs = bbuf.dofs;

                int totalDofs = nd * vdim;
                bbuf.elmatVector.topLeftCorner(totalDofs, totalDofs).setZero();

                for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                    if (bdrIds_[k] >= 0 && bdrIds_[k] != attr)
                        continue;
                    bbuf.dynMatrix.resize(0, 0);
                    bdrIntegs_[k]->assembleFaceMatrix(*ref, btrans, bbuf.dynMatrix);

                    if (bbuf.dynMatrix.rows() == nd && bbuf.dynMatrix.cols() == nd) {
                        for (int c = 0; c < vdim; ++c) {
                            bbuf.elmatVector.block(c * nd, c * nd, nd, nd) += bbuf.dynMatrix;
                        }
                    }
                    else if (bbuf.dynMatrix.rows() == totalDofs && bbuf.dynMatrix.cols() == totalDofs) {
                        bbuf.elmatVector.topLeftCorner(totalDofs, totalDofs) += bbuf.dynMatrix;
                    }
                }

                // Pre-compute valid DOF indices to reduce branching
                int bNumValidDofs = 0;
                for (int i = 0; i < totalDofs; ++i) {
                    if (dofs[i] != InvalidIndex) {
                        bbuf.validDofs[bNumValidDofs++] = i;
                    }
                }

                // Branch-reduced triplet writing
                for (int vi = 0; vi < bNumValidDofs; ++vi) {
                    const int i = bbuf.validDofs[vi];
                    const Index di = dofs[i];
                    for (int vj = 0; vj < bNumValidDofs; ++vj) {
                        const int j = bbuf.validDofs[vj];
                        const Real v = bbuf.elmatVector(i, j);
                        if (keepTriplet(v)) {
                            triplets_.emplace_back(di, dofs[j], v);
                        }
                    }
                }
            }
        }

        // 高效构建稀疏矩阵：Eigen 内部会排序和求和
        mat_.setFromTriplets(std::move(triplets_));
    }

    // =============================================================================
    // LinearFormAssembler
    // =============================================================================

    LinearFormAssembler::LinearFormAssembler(const FESpace* fes) : fes_(fes)
    {
        if (fes_ && fes_->numDofs() > 0) {
            vec_.setZero(fes_->numDofs());
        }
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#else
        int nthreads = 1;
#endif
        buffers_.resize(nthreads);
        threadVectors_.resize(nthreads);
        for (auto& v : threadVectors_) {
            v.setZero(fes_ ? fes_->numDofs() : 0);
        }
    }

    void LinearFormAssembler::assemble()
    {
        if (!fes_)
            return;
        vec_.setZero();

        const Mesh* mesh = fes_->mesh();
        if (!mesh)
            return;

        const int vdim = fes_->vdim();
        const DomainIntegratorMap activeDomains = buildDomainIntegratorMap(domainSets_, *mesh);

#ifdef _OPENMP
        for (auto& v : threadVectors_) {
            v.setZero();
        }
#endif

        if (!domainIntegs_.empty()) {
            const Index numElements = mesh->numElements();

#ifdef _OPENMP
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                ThreadBuffer& buf = buffers_[tid];
                Vector& localVec = threadVectors_[tid];

                ElementTransform trans;
                trans.setMesh(mesh);
#else
            ThreadBuffer& buf = buffers_[0];
            ElementTransform trans;
            trans.setMesh(mesh);
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
                for (Index e = 0; e < numElements; ++e) {
                    const ReferenceElement* ref = fes_->elementRefElement(e);
                    if (!ref)
                        continue;
                    int nd = ref->numDofs();

                    trans.setElement(e);
                    const int elemAttr = mesh->element(e).attribute();
                    const auto domainIt = activeDomains.find(elemAttr);
                    if (domainIt == activeDomains.end() || domainIt->second.empty())
                        continue;

                    // Use pre-allocated DOF buffer
                    buf.numDofs = nd * vdim;
                    fes_->getElementDofs(e, std::span<Index> {buf.dofs.data(), static_cast<size_t>(buf.numDofs)});
                    const auto& dofs = buf.dofs;

                    int totalDofs = nd * vdim;
                    buf.elvecVector.head(totalDofs).setZero();

                    for (size_t ki : domainIt->second) {
                        const auto& integ = domainIntegs_[ki];
                        int ivdim = integ->vdim();
                        int iTotalDofs = nd * ivdim;
                        buf.dynVector.resize(iTotalDofs);
                        integ->assembleElementVector(*ref, trans, buf.dynVector);

                        if (ivdim == 1) {
                            // Scalar integrator: expand to vector field segments
                            for (int c = 0; c < vdim; ++c) {
                                buf.elvecVector.segment(c * nd, nd) += buf.dynVector;
                            }
                        }
                        else {
                            // Vector integrator: add directly
                            buf.elvecVector.head(iTotalDofs) += buf.dynVector;
                        }
                    }

                    for (int i = 0; i < totalDofs; ++i) {
                        if (dofs[i] != InvalidIndex)
#ifdef _OPENMP
                            localVec(dofs[i]) += buf.elvecVector(i);
#else
                        vec_(dofs[i]) += buf.elvecVector(i);
#endif
                    }
                }

#ifdef _OPENMP
            }

            for (const auto& lv : threadVectors_) {
                vec_ += lv;
            }
#endif
        }

        // Boundary integrals (not parallelized - usually small)
        if (!bdrIntegs_.empty()) {
            FacetElementTransform btrans;
            btrans.setMesh(mesh);
            ThreadBuffer& bbuf = buffers_[0]; // Use first thread buffer

            for (Index b = 0; b < mesh->numBdrElements(); ++b) {
                int attr = mesh->bdrElement(b).attribute();

                if (!fes_->isExternalBoundaryId(attr))
                    continue;

                const ReferenceElement* ref = fes_->bdrElementRefElement(b);
                if (!ref)
                    continue;
                int nd = ref->numDofs();

                btrans.setBoundaryElement(b);

                // Use pre-allocated DOF buffer
                bbuf.numDofs = nd * vdim;
                fes_->getBdrElementDofs(b, std::span<Index> {bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
                const auto& dofs = bbuf.dofs;

                int totalDofs = nd * vdim;
                bbuf.elvecVector.head(totalDofs).setZero();

                for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                    if (bdrIds_[k] >= 0 && bdrIds_[k] != attr)
                        continue;
                    bbuf.dynVector.resize(0);
                    bdrIntegs_[k]->assembleFaceVector(*ref, btrans, bbuf.dynVector);

                    if (bbuf.dynVector.size() == nd) {
                        for (int c = 0; c < vdim; ++c) {
                            bbuf.elvecVector.segment(c * nd, nd) += bbuf.dynVector;
                        }
                    }
                    else if (bbuf.dynVector.size() == totalDofs) {
                        bbuf.elvecVector.head(totalDofs) += bbuf.dynVector;
                    }
                }

                for (int i = 0; i < totalDofs; ++i) {
                    if (dofs[i] != InvalidIndex)
                        vec_(dofs[i]) += bbuf.elvecVector(i);
                }
            }
        }
    }

} // namespace mpfem
