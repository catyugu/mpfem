#include "assembler.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include <unordered_set>

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
    buffers_.resize(nthreads);
#else
    buffers_.resize(1);
#endif
    // 预分配三元组缓冲区
    if (fes_) {
        const Mesh* mesh = fes_->mesh();
        if (mesh) {
            triplets_.reserve(mesh->numElements() * MAX_DOFS * MAX_DOFS / 2);
        }
    }
}

void BilinearFormAssembler::computeSparsityPattern() {
    if (!fes_) return;
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    Index ndofs = fes_->numDofs();
    std::vector<std::unordered_set<Index>> rowCols(ndofs);
    
    // 临时缓冲区
    std::vector<Index> dofs;
    for (Index e = 0; e < mesh->numElements(); ++e) {
        fes_->getElementDofs(e, dofs);
        for (auto i : dofs) if (i != InvalidIndex)
            for (auto j : dofs) if (j != InvalidIndex)
                rowCols[i].insert(j);
    }
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        fes_->getBdrElementDofs(b, dofs);
        for (auto i : dofs) if (i != InvalidIndex)
            for (auto j : dofs) if (j != InvalidIndex)
                rowCols[i].insert(j);
    }
    
    triplets_.clear();
    triplets_.reserve(ndofs * 27);
    for (Index i = 0; i < ndofs; ++i) {
        for (auto j : rowCols[i]) {
            triplets_.emplace_back(i, j, 0.0);
        }
    }
    mat_.setFromTriplets(std::move(triplets_));
}

void BilinearFormAssembler::assemble() {
    if (!fes_) return;
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    mat_.setZero();  // 保留稀疏结构
    
    const Index numElements = mesh->numElements();
    const int vdim = fes_->vdim();  // 向量维度
    
    // 清空并复用三元组缓冲区（避免重新分配）
    triplets_.clear();
    triplets_.reserve(numElements * MAX_DOFS * MAX_DOFS * vdim * vdim / 2);
    
#ifdef _OPENMP
    // 并行组装
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadBuffer& buf = buffers_[tid];
        
        // 线程局部三元组
        std::vector<SparseMatrix::Triplet> localTriplets;
        localTriplets.reserve(numElements * MAX_DOFS * MAX_DOFS * vdim * vdim / buffers_.size() / 2);
        
        ElementTransform trans;
        trans.setMesh(mesh);
        
        #pragma omp for schedule(dynamic, 64)
        for (Index e = 0; e < numElements; ++e) {
            const ReferenceElement* ref = fes_->elementRefElement(e);
            if (!ref) continue;
            int nd = ref->numDofs();
            buf.numDofs = nd * vdim;
            
            trans.setElement(e);
            buf.elmat.setZero();
            
            for (const auto& integ : domainIntegs_) {
                Matrix temp;
                integ->assembleElementMatrix(*ref, trans, temp, vdim);
                buf.elmat.topLeftCorner(nd * vdim, nd * vdim) += temp;
            }
            
            // 获取 DOFs 到缓冲区
            std::vector<Index> dofs;
            fes_->getElementDofs(e, dofs);
            
            for (int i = 0; i < nd * vdim; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < nd * vdim; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmat(i, j);
                    if (std::abs(v) > 1e-30)
                        localTriplets.emplace_back(dofs[i], dofs[j], v);
                }
            }
        }
        
        // 合并线程局部三元组
        #pragma omp critical
        {
            triplets_.insert(triplets_.end(), localTriplets.begin(), localTriplets.end());
        }
    }
#else
    // 串行版本
    ElementTransform trans;
    trans.setMesh(mesh);
    ThreadBuffer& buf = buffers_[0];
    
    for (Index e = 0; e < numElements; ++e) {
        const ReferenceElement* ref = fes_->elementRefElement(e);
        if (!ref) continue;
        int nd = ref->numDofs();
        
        trans.setElement(e);
        buf.elmat.setZero();
        
        for (const auto& integ : domainIntegs_) {
            Matrix temp;
            integ->assembleElementMatrix(*ref, trans, temp, vdim);
            buf.elmat.topLeftCorner(nd * vdim, nd * vdim) += temp;
        }
        
        std::vector<Index> dofs;
        fes_->getElementDofs(e, dofs);
        
        for (int i = 0; i < nd * vdim; ++i) {
            if (dofs[i] == InvalidIndex) continue;
            for (int j = 0; j < nd * vdim; ++j) {
                if (dofs[j] == InvalidIndex) continue;
                Real v = buf.elmat(i, j);
                if (std::abs(v) > 1e-30)
                    triplets_.emplace_back(dofs[i], dofs[j], v);
            }
        }
    }
#endif
    
    // 边界积分（通常较少，串行处理）
    // 注意：只对外边界应用边界条件，跳过内边界
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        ThreadBuffer& buf = buffers_[0];
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            // 跳过内边界
            if (!fes_->isExternalBoundaryId(attr)) {
                continue;
            }
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            buf.elmat.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Matrix temp;
                bdrIntegs_[k]->assembleFaceMatrix(*ref, btrans, temp, vdim);
                buf.elmat.topLeftCorner(nd * vdim, nd * vdim) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            
            for (int i = 0; i < nd * vdim; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < nd * vdim; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmat(i, j);
                    if (std::abs(v) > 1e-30)
                        triplets_.emplace_back(dofs[i], dofs[j], v);
                }
            }
        }
    }
    
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
    buffers_.resize(nthreads);
    threadVectors_.resize(nthreads);
    for (auto& v : threadVectors_) {
        v.setZero(fes_->numDofs());
    }
#else
    buffers_.resize(1);
#endif
}

void LinearFormAssembler::assemble() {
    if (!fes_) return;
    vec_.setZero();
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    const int vdim = fes_->vdim();
    
#ifdef _OPENMP
    // 重置线程局部向量
    for (auto& v : threadVectors_) {
        v.setZero();
    }
#endif
    
    // 域积分
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
            
            #pragma omp for schedule(dynamic, 64)
            for (Index e = 0; e < numElements; ++e) {
                const ReferenceElement* ref = fes_->elementRefElement(e);
                if (!ref) continue;
                int nd = ref->numDofs();
                
                trans.setElement(e);
                buf.elvec.setZero();
                
                for (const auto& integ : domainIntegs_) {
                    Vector temp(nd * vdim);
                    temp.setZero();
                    integ->assembleElementVector(*ref, trans, temp, vdim);
                    buf.elvec.head(nd * vdim) += temp;
                }
                
                std::vector<Index> dofs;
                fes_->getElementDofs(e, dofs);
                for (int i = 0; i < nd * vdim; ++i) {
                    if (dofs[i] != InvalidIndex)
                        localVec(dofs[i]) += buf.elvec(i);
                }
            }
        }
        
        // 合并线程局部向量
        for (const auto& lv : threadVectors_) {
            vec_ += lv;
        }
#else
        ElementTransform trans;
        trans.setMesh(mesh);
        ThreadBuffer& buf = buffers_[0];
        
        for (Index e = 0; e < numElements; ++e) {
            const ReferenceElement* ref = fes_->elementRefElement(e);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            trans.setElement(e);
            buf.elvec.setZero();
            
            for (const auto& integ : domainIntegs_) {
                Vector temp(nd * vdim);
                temp.setZero();
                integ->assembleElementVector(*ref, trans, temp, vdim);
                buf.elvec.head(nd * vdim) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getElementDofs(e, dofs);
            for (int i = 0; i < nd * vdim; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvec(i);
            }
        }
#endif
    }
    
    // 边界积分（通常较少，串行处理）
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        ThreadBuffer& buf = buffers_[0];
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            // 跳过内边界
            if (!fes_->isExternalBoundaryId(attr)) {
                continue;
            }
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            buf.elvec.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Vector temp(nd * vdim);
                temp.setZero();
                bdrIntegs_[k]->assembleFaceVector(*ref, btrans, temp, vdim);
                buf.elvec.head(nd * vdim) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            for (int i = 0; i < nd * vdim; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvec(i);
            }
        }
    }
}

}  // namespace mpfem
