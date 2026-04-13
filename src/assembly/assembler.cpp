#include "assembler.hpp"
#include "assembly/element_binding.hpp"
#include "fe/element_transform.hpp"
#include "field/fe_space.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

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
            uniqueAttrs.insert(mesh.element(e).attribute);
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

    struct alignas(64) ThreadBuffer {
        Eigen::Matrix<mpfem::Real, mpfem::MaxDofsPerElement, mpfem::MaxDofsPerElement, Eigen::RowMajor> elmatVector;
        Eigen::Matrix<mpfem::Real, mpfem::MaxDofsPerElement, 1> elvecVector;
        std::array<mpfem::Index, mpfem::MaxDofsPerElement> dofs;
        mpfem::Index numDofs = 0;
        mpfem::Matrix dynMatrix;
        mpfem::Vector dynVector;
        std::array<int, mpfem::MaxDofsPerElement> validDofs;
        int numValidDofs = 0;

        void ensureDynMatrixSize(int maxTotalDofs)
        {
            if (dynMatrix.rows() < maxTotalDofs || dynMatrix.cols() < maxTotalDofs) {
                dynMatrix.resize(maxTotalDofs, maxTotalDofs);
            }
        }
    };

} // namespace

namespace mpfem {

    // =============================================================================
    // BilinearFormAssembler
    // =============================================================================

    BilinearFormAssembler::BilinearFormAssembler(const FESpace* fes)
        : fes_(fes)
    {
        if (fes_ && fes_->numDofs() > 0) {
            mat_.resize(fes_->numDofs(), fes_->numDofs());
        }

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

        // 预估 triplet 数量：
        const size_t estimatedTriplets = fes_->numDofs() * MaxDofsPerElement;
        triplets_.reserve(estimatedTriplets);

#ifdef _OPENMP
#pragma omp parallel
        {
            ThreadBuffer buf;

            std::vector<SparseMatrix::Triplet> localTriplets;
            localTriplets.reserve(estimatedTriplets / omp_get_num_threads());

            ElementTransform trans;

            const int maxDynSize = MaxDofsPerElement * vdim;
            buf.ensureDynMatrixSize(maxDynSize);
#else
        ThreadBuffer buf;
        ElementTransform trans;

        const int maxDynSize = MaxDofsPerElement * vdim;
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

                const Element elem = mesh->element(e);
                bindElementToTransform(trans, *mesh, e, false);

                const int elemAttr = elem.attribute;
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
                    if (ivdim != vdim) {
                        MPFEM_THROW(ArgumentException, "Bilinear integrator vdim must match FESpace vdim");
                    }

                    const int iTotalDofs = nd * ivdim;
                    if (buf.dynMatrix.rows() < iTotalDofs || buf.dynMatrix.cols() < iTotalDofs) {
                        buf.dynMatrix.resize(iTotalDofs, iTotalDofs);
                    }

                    integ->assembleElementMatrix(*ref, trans, buf.dynMatrix);

                    if (buf.dynMatrix.rows() != iTotalDofs || buf.dynMatrix.cols() != iTotalDofs) {
                        MPFEM_THROW(ArgumentException, "Bilinear integrator returned matrix with unexpected size");
                    }
                    buf.elmatVector.topLeftCorner(iTotalDofs, iTotalDofs) += buf.dynMatrix.topLeftCorner(iTotalDofs, iTotalDofs);
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
            ElementTransform btrans;
            ThreadBuffer bbuf;
            const int maxDynSize = MaxDofsPerElement * vdim;
            bbuf.ensureDynMatrixSize(maxDynSize);

            for (Index b = 0; b < mesh->numBdrElements(); ++b) {
                const Element belem = mesh->bdrElement(b);
                int attr = belem.attribute;

                if (!fes_->isExternalBoundaryId(attr))
                    continue;

                const ReferenceElement* ref = fes_->bdrElementRefElement(b);
                if (!ref)
                    continue;
                int nd = ref->numDofs();

                bindElementToTransform(btrans, *mesh, b, true);

                if (mesh->hasTopology()) {
                    Index faceIdx = mesh->getBoundaryFaceIndex(b);
                    if (faceIdx != InvalidIndex) {
                        const auto& faceInfo = mesh->getFaceInfo(faceIdx);
                        btrans.setFaceInfo(faceInfo.elem1, faceInfo.localFace1);
                    }
                }

                // Use pre-allocated DOF buffer
                bbuf.numDofs = nd * vdim;
                fes_->getBdrElementDofs(b, std::span<Index> {bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
                const auto& dofs = bbuf.dofs;

                int totalDofs = nd * vdim;
                bbuf.elmatVector.topLeftCorner(totalDofs, totalDofs).setZero();

                for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                    if (bdrIds_[k] >= 0 && bdrIds_[k] != attr)
                        continue;
                    bdrIntegs_[k]->assembleFacetMatrix(*ref, btrans, bbuf.dynMatrix);

                    if (bbuf.dynMatrix.rows() != totalDofs || bbuf.dynMatrix.cols() != totalDofs) {
                        MPFEM_THROW(ArgumentException, "Boundary bilinear integrator returned matrix with unexpected size");
                    }
                    bbuf.elmatVector.topLeftCorner(totalDofs, totalDofs) += bbuf.dynMatrix;
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

    LinearFormAssembler::LinearFormAssembler(const FESpace* fes)
        : fes_(fes)
    {
        if (fes_ && fes_->numDofs() > 0) {
            vec_.setZero(fes_->numDofs());
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
        // Ensure threadVectors_ is sized for current thread count
        {
            const int nthreads = omp_get_max_threads();
            if (static_cast<int>(threadVectors_.size()) < nthreads) {
                threadVectors_.resize(nthreads);
                for (auto& v : threadVectors_) {
                    v.setZero(fes_->numDofs());
                }
            }
        }
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
                ThreadBuffer buf;
                Vector& localVec = threadVectors_[tid];

                ElementTransform trans;
#else
            ThreadBuffer buf;
            ElementTransform trans;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
                for (Index e = 0; e < numElements; ++e) {
                    const ReferenceElement* ref = fes_->elementRefElement(e);
                    if (!ref)
                        continue;
                    int nd = ref->numDofs();

                    const Element elem = mesh->element(e);
                    bindElementToTransform(trans, *mesh, e, false);

                    const int elemAttr = elem.attribute;
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
                        if (ivdim != vdim) {
                            MPFEM_THROW(ArgumentException, "Linear integrator vdim must match FESpace vdim");
                        }
                        int iTotalDofs = nd * ivdim;
                        buf.dynVector.resize(iTotalDofs);
                        integ->assembleElementVector(*ref, trans, buf.dynVector);

                        if (buf.dynVector.size() != iTotalDofs) {
                            MPFEM_THROW(ArgumentException, "Linear integrator returned vector with unexpected size");
                        }
                        buf.elvecVector.head(iTotalDofs) += buf.dynVector;
                    }

                    // Pre-compute valid DOF indices to reduce branching
                    buf.numValidDofs = 0;
                    for (int i = 0; i < totalDofs; ++i) {
                        if (dofs[i] != InvalidIndex) {
                            buf.validDofs[buf.numValidDofs++] = i;
                        }
                    }

                    // Branch-reduced vector assembly: iterate only valid DOFs
                    Vector& targetVec = [&]() -> Vector& {
#ifdef _OPENMP
                        return localVec;
#else
                    return vec_;
#endif
                    }();

                    for (int vi = 0; vi < buf.numValidDofs; ++vi) {
                        const int i = buf.validDofs[vi];
                        targetVec(dofs[i]) += buf.elvecVector(i);
                    }
                }

#ifdef _OPENMP
            }

            for (auto& lv : threadVectors_) {
                vec_ += std::move(lv);
            }
#endif
        }

        // Boundary integrals (not parallelized - usually small)
        if (!bdrIntegs_.empty()) {
            ElementTransform btrans;
            ThreadBuffer bbuf;

            for (Index b = 0; b < mesh->numBdrElements(); ++b) {
                const Element belem = mesh->bdrElement(b);
                int attr = belem.attribute;

                if (!fes_->isExternalBoundaryId(attr))
                    continue;

                const ReferenceElement* ref = fes_->bdrElementRefElement(b);
                if (!ref)
                    continue;
                int nd = ref->numDofs();

                bindElementToTransform(btrans, *mesh, b, true);
                if (mesh->hasTopology()) {
                    Index faceIdx = mesh->getBoundaryFaceIndex(b);
                    if (faceIdx != InvalidIndex) {
                        const auto& faceInfo = mesh->getFaceInfo(faceIdx);
                        btrans.setFaceInfo(faceInfo.elem1, faceInfo.localFace1);
                    }
                }

                // Use pre-allocated DOF buffer
                bbuf.numDofs = nd * vdim;
                fes_->getBdrElementDofs(b, std::span<Index> {bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
                const auto& dofs = bbuf.dofs;

                int totalDofs = nd * vdim;
                bbuf.elvecVector.head(totalDofs).setZero();

                for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                    if (bdrIds_[k] >= 0 && bdrIds_[k] != attr)
                        continue;
                    bdrIntegs_[k]->assembleFacetVector(*ref, btrans, bbuf.dynVector);

                    if (bbuf.dynVector.size() != totalDofs) {
                        MPFEM_THROW(ArgumentException, "Boundary linear integrator returned vector with unexpected size");
                    }
                    bbuf.elvecVector.head(totalDofs) += bbuf.dynVector;
                }

                // Pre-compute valid DOF indices to reduce branching
                int bNumValidDofs = 0;
                for (int i = 0; i < totalDofs; ++i) {
                    if (dofs[i] != InvalidIndex) {
                        bbuf.validDofs[bNumValidDofs++] = i;
                    }
                }

                // Branch-reduced vector assembly
                for (int vi = 0; vi < bNumValidDofs; ++vi) {
                    const int i = bbuf.validDofs[vi];
                    vec_(dofs[i]) += bbuf.elvecVector(i);
                }
            }
        }
    }

} // namespace mpfem
