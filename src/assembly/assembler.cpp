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

void BilinearFormAssembler::expandScalarToVector(const Matrix& scalarMat, 
                                                   Matrix& vectorMat, 
                                                   int nd, int vdim) {
    vectorMat.setZero(nd * vdim, nd * vdim);
    for (int c = 0; c < vdim; ++c) {
        vectorMat.block(c * nd, c * nd, nd, nd) = scalarMat;
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
    triplets_.reserve(numElements * MAX_DOFS * MAX_DOFS * vdim * vdim / 2);
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadBuffer& buf = buffers_[tid];
        
        std::vector<SparseMatrix::Triplet> localTriplets;
        localTriplets.reserve(numElements * MAX_DOFS * MAX_DOFS * vdim * vdim / buffers_.size() / 2);
        
        ElementTransform trans;
        trans.setMesh(mesh);
        
        #pragma omp for schedule(dynamic, 64)
        for (Index e = 0; e < numElements; ++e) {
            const ReferenceElement* ref = fes_->elementRefElement(e);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            trans.setElement(e);
            
            std::vector<Index> dofs;
            fes_->getElementDofs(e, dofs);
            
            int totalDofs = nd * vdim;
            buf.elmatVector.setZero();
            
            for (const auto& integ : domainIntegs_) {
                Matrix temp;
                integ->assembleElementMatrix(*ref, trans, temp);
                
                // 根据输出矩阵大小判断是否需要扩展
                if (temp.rows() == nd && temp.cols() == nd) {
                    // 标量积分器输出：扩展到向量场对角块
                    for (int c = 0; c < vdim; ++c) {
                        buf.elmatVector.block(c * nd, c * nd, nd, nd) += temp;
                    }
                } else if (temp.rows() == totalDofs && temp.cols() == totalDofs) {
                    // 向量积分器输出：直接使用
                    buf.elmatVector.topLeftCorner(totalDofs, totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < totalDofs; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmatVector(i, j);
                    if (std::abs(v) > 1e-30)
                        localTriplets.emplace_back(dofs[i], dofs[j], v);
                }
            }
        }
        
        #pragma omp critical
        {
            triplets_.insert(triplets_.end(), localTriplets.begin(), localTriplets.end());
        }
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
        
        std::vector<Index> dofs;
        fes_->getElementDofs(e, dofs);
        
        int totalDofs = nd * vdim;
        buf.elmatVector.setZero();
        
        for (const auto& integ : domainIntegs_) {
            Matrix temp;
            integ->assembleElementMatrix(*ref, trans, temp);
            
            if (temp.rows() == nd && temp.cols() == nd) {
                for (int c = 0; c < vdim; ++c) {
                    buf.elmatVector.block(c * nd, c * nd, nd, nd) += temp;
                }
            } else if (temp.rows() == totalDofs && temp.cols() == totalDofs) {
                buf.elmatVector.topLeftCorner(totalDofs, totalDofs) += temp;
            }
        }
        
        for (int i = 0; i < totalDofs; ++i) {
            if (dofs[i] == InvalidIndex) continue;
            for (int j = 0; j < totalDofs; ++j) {
                if (dofs[j] == InvalidIndex) continue;
                Real v = buf.elmatVector(i, j);
                if (std::abs(v) > 1e-30)
                    triplets_.emplace_back(dofs[i], dofs[j], v);
            }
        }
    }
#endif
    
    // 边界积分
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        ThreadBuffer& buf = buffers_[0];
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            if (!fes_->isExternalBoundaryId(attr)) continue;
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            
            int totalDofs = nd * vdim;
            buf.elmatVector.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Matrix temp;
                bdrIntegs_[k]->assembleFaceMatrix(*ref, btrans, temp);
                
                if (temp.rows() == nd && temp.cols() == nd) {
                    for (int c = 0; c < vdim; ++c) {
                        buf.elmatVector.block(c * nd, c * nd, nd, nd) += temp;
                    }
                } else if (temp.rows() == totalDofs && temp.cols() == totalDofs) {
                    buf.elmatVector.topLeftCorner(totalDofs, totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < totalDofs; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmatVector(i, j);
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

void LinearFormAssembler::expandScalarToVector(const Vector& scalarVec,
                                                Vector& vectorVec,
                                                int nd, int vdim) {
    vectorVec.setZero(nd * vdim);
    for (int c = 0; c < vdim; ++c) {
        vectorVec.segment(c * nd, nd) = scalarVec;
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
            
            #pragma omp for schedule(dynamic, 64)
            for (Index e = 0; e < numElements; ++e) {
                const ReferenceElement* ref = fes_->elementRefElement(e);
                if (!ref) continue;
                int nd = ref->numDofs();
                
                trans.setElement(e);
                
                std::vector<Index> dofs;
                fes_->getElementDofs(e, dofs);
                
                int totalDofs = nd * vdim;
                buf.elvecVector.setZero();
                
                for (const auto& integ : domainIntegs_) {
                    Vector temp;
                    integ->assembleElementVector(*ref, trans, temp);
                    
                    if (temp.size() == nd) {
                        for (int c = 0; c < vdim; ++c) {
                            buf.elvecVector.segment(c * nd, nd) += temp;
                        }
                    } else if (temp.size() == totalDofs) {
                        buf.elvecVector.head(totalDofs) += temp;
                    }
                }
                
                for (int i = 0; i < totalDofs; ++i) {
                    if (dofs[i] != InvalidIndex)
                        localVec(dofs[i]) += buf.elvecVector(i);
                }
            }
        }
        
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
            
            std::vector<Index> dofs;
            fes_->getElementDofs(e, dofs);
            
            int totalDofs = nd * vdim;
            buf.elvecVector.setZero();
            
            for (const auto& integ : domainIntegs_) {
                Vector temp;
                integ->assembleElementVector(*ref, trans, temp);
                
                if (temp.size() == nd) {
                    for (int c = 0; c < vdim; ++c) {
                        buf.elvecVector.segment(c * nd, nd) += temp;
                    }
                } else if (temp.size() == totalDofs) {
                    buf.elvecVector.head(totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvecVector(i);
            }
        }
#endif
    }
    
    // 边界积分
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        ThreadBuffer& buf = buffers_[0];
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            int attr = mesh->bdrElement(b).attribute();
            
            if (!fes_->isExternalBoundaryId(attr)) continue;
            
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            
            int totalDofs = nd * vdim;
            buf.elvecVector.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Vector temp;
                bdrIntegs_[k]->assembleFaceVector(*ref, btrans, temp);
                
                if (temp.size() == nd) {
                    for (int c = 0; c < vdim; ++c) {
                        buf.elvecVector.segment(c * nd, nd) += temp;
                    }
                } else if (temp.size() == totalDofs) {
                    buf.elvecVector.head(totalDofs) += temp;
                }
            }
            
            for (int i = 0; i < totalDofs; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvecVector(i);
            }
        }
    }
}

}  // namespace mpfem