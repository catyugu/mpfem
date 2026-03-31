#include "assembler.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include <algorithm>

namespace mpfem {

// =============================================================================
// BilinearFormAssembler
// =============================================================================

BilinearFormAssembler::BilinearFormAssembler(const FESpace* fes) : fes_(fes) {
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

void BilinearFormAssembler::assemble() {
    if (!fes_) return;
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    mat_.setZero();
    
    const Index numElements = mesh->numElements();
    const int vdim = fes_->vdim();
    
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
#else
    ThreadBuffer& buf = buffers_[0];
    ElementTransform trans;
    trans.setMesh(mesh);
#endif
        
        // Unified loop body (works for both OpenMP and non-OpenMP)
#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
#endif
        for (Index e = 0; e < numElements; ++e) {
            const ReferenceElement* ref = fes_->elementRefElement(e);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            trans.setElement(e);
            Index elemAttr = trans.attribute();  // Get element's domain ID
            
            // Use pre-allocated DOF buffer
            buf.numDofs = nd * vdim;
            fes_->getElementDofs(e, std::span<Index>{buf.dofs.data(), static_cast<size_t>(buf.numDofs)});
            const auto& dofs = buf.dofs;
            
            int totalDofs = nd * vdim;
            buf.elmatVector.setZero();
            
            for (size_t ki = 0; ki < domainIntegs_.size(); ++ki) {
                const auto& domains = domainSets_[ki];
                if (!domains.empty() && !domains.contains(static_cast<int>(elemAttr))) continue;
                
                const auto& integ = domainIntegs_[ki];
                int ivdim = integ->vdim();
                int iTotalDofs = nd * ivdim;
                Matrix temp(iTotalDofs, iTotalDofs);
                integ->assembleElementMatrix(*ref, trans, temp);
                
                if (ivdim == 1) {
                    // Scalar integrator: expand to vector field diagonal blocks
                    for (int c = 0; c < vdim; ++c) {
                        buf.elmatVector.block(c * nd, c * nd, nd, nd) += temp;
                    }
                } else {
                    // Vector integrator: add directly
                    buf.elmatVector.topLeftCorner(iTotalDofs, iTotalDofs) += temp;
                }
            }
            
            // 直接写入 triplets
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < totalDofs; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmatVector(i, j);
#ifdef _OPENMP
                    localTriplets.emplace_back(dofs[i], dofs[j], v);
#else
                    triplets_.emplace_back(dofs[i], dofs[j], v);
#endif
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
        ThreadBuffer& bbuf = buffers_[0];  // Use first thread buffer
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            if (!fes_->isExternalBoundaryId(attr)) continue;
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            // Use pre-allocated DOF buffer
            bbuf.numDofs = nd * vdim;
            fes_->getBdrElementDofs(b, std::span<Index>{bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
            const auto& dofs = bbuf.dofs;
            
            int totalDofs = nd * vdim;
            bbuf.elmatVector.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Matrix temp;
                bdrIntegs_[k]->assembleFaceMatrix(*ref, btrans, temp);
                
                if (temp.rows() == nd && temp.cols() == nd) {
                    for (int c = 0; c < vdim; ++c) {
                        bbuf.elmatVector.block(c * nd, c * nd, nd, nd) += temp;
                    }
                } else if (temp.rows() == totalDofs && temp.cols() == totalDofs) {
                    bbuf.elmatVector.topLeftCorner(totalDofs, totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < totalDofs; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = bbuf.elmatVector(i, j);
                    triplets_.emplace_back(dofs[i], dofs[j], v);
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

LinearFormAssembler::LinearFormAssembler(const FESpace* fes) : fes_(fes) {
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

void LinearFormAssembler::assemble() {
    if (!fes_) return;
    vec_.setZero();
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    const int vdim = fes_->vdim();
    
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
                if (!ref) continue;
                int nd = ref->numDofs();
                
                trans.setElement(e);
                Index elemAttr = trans.attribute();  // Get element's domain ID
                
                // Use pre-allocated DOF buffer
                buf.numDofs = nd * vdim;
                fes_->getElementDofs(e, std::span<Index>{buf.dofs.data(), static_cast<size_t>(buf.numDofs)});
                const auto& dofs = buf.dofs;
                
                int totalDofs = nd * vdim;
                buf.elvecVector.setZero();
                
                for (size_t ki = 0; ki < domainIntegs_.size(); ++ki) {
                    const auto& domains = domainSets_[ki];
                    if (!domains.empty() && !domains.contains(static_cast<int>(elemAttr))) continue;
                    
                    const auto& integ = domainIntegs_[ki];
                    int ivdim = integ->vdim();
                    int iTotalDofs = nd * ivdim;
                    Vector temp(iTotalDofs);
                    integ->assembleElementVector(*ref, trans, temp);
                    
                    if (ivdim == 1) {
                        // Scalar integrator: expand to vector field segments
                        for (int c = 0; c < vdim; ++c) {
                            buf.elvecVector.segment(c * nd, nd) += temp;
                        }
                    } else {
                        // Vector integrator: add directly
                        buf.elvecVector.head(iTotalDofs) += temp;
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
        ThreadBuffer& bbuf = buffers_[0];  // Use first thread buffer
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            if (!fes_->isExternalBoundaryId(attr)) continue;
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            // Use pre-allocated DOF buffer
            bbuf.numDofs = nd * vdim;
            fes_->getBdrElementDofs(b, std::span<Index>{bbuf.dofs.data(), static_cast<size_t>(bbuf.numDofs)});
            const auto& dofs = bbuf.dofs;
            
            int totalDofs = nd * vdim;
            bbuf.elvecVector.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Vector temp;
                bdrIntegs_[k]->assembleFaceVector(*ref, btrans, temp);
                
                if (temp.size() == nd) {
                    for (int c = 0; c < vdim; ++c) {
                        bbuf.elvecVector.segment(c * nd, nd) += temp;
                    }
                } else if (temp.size() == totalDofs) {
                    bbuf.elvecVector.head(totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += bbuf.elvecVector(i);
            }
        }
    }
}

}  // namespace mpfem
