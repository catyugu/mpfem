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
    buffers_.resize(omp_get_max_threads());
#else
    buffers_.resize(1);
#endif
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
    
    std::vector<SparseMatrix::Triplet> triplets;
    triplets.reserve(ndofs * 27);
    for (Index i = 0; i < ndofs; ++i) {
        for (auto j : rowCols[i]) {
            triplets.emplace_back(i, j, 0.0);  // 先填0，后续累加
        }
    }
    mat_.setFromTriplets(std::move(triplets));
}

void BilinearFormAssembler::assemble() {
    if (!fes_ || domainIntegs_.empty()) return;
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    mat_.setZero();  // 保留稀疏结构
    
    std::vector<SparseMatrix::Triplet> triplets;
    triplets.reserve(mesh->numElements() * MAX_DOFS * MAX_DOFS);
    
    ElementTransform trans;
    trans.setMesh(mesh);
    
    for (Index e = 0; e < mesh->numElements(); ++e) {
        const ReferenceElement* ref = fes_->elementRefElement(e);
        if (!ref) continue;
        int nd = ref->numDofs();
        
        trans.setElement(e);
        
        ThreadBuffer& buf = buffers_[0];
        buf.elmat.setZero();
        
        for (const auto& integ : domainIntegs_) {
            Matrix temp;
            integ->assembleElementMatrix(*ref, trans, temp);
            buf.elmat.topLeftCorner(nd, nd) += temp;
        }
        
        std::vector<Index> dofs;
        fes_->getElementDofs(e, dofs);
        
        for (int i = 0; i < nd; ++i) {
            if (dofs[i] == InvalidIndex) continue;
            for (int j = 0; j < nd; ++j) {
                if (dofs[j] == InvalidIndex) continue;
                Real v = buf.elmat(i, j);
                if (std::abs(v) > 1e-30)
                    triplets.emplace_back(dofs[i], dofs[j], v);
            }
        }
    }
    
    // 边界积分
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            int attr = mesh->bdrElement(b).attribute();
            
            ThreadBuffer& buf = buffers_[0];
            buf.elmat.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Matrix temp;
                bdrIntegs_[k]->assembleFaceMatrix(*ref, btrans, temp);
                buf.elmat.topLeftCorner(nd, nd) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            
            for (int i = 0; i < nd; ++i) {
                if (dofs[i] == InvalidIndex) continue;
                for (int j = 0; j < nd; ++j) {
                    if (dofs[j] == InvalidIndex) continue;
                    Real v = buf.elmat(i, j);
                    if (std::abs(v) > 1e-30)
                        triplets.emplace_back(dofs[i], dofs[j], v);
                }
            }
        }
    }
    
    mat_.setFromTriplets(std::move(triplets));
}

// =============================================================================
// LinearFormAssembler
// =============================================================================

LinearFormAssembler::LinearFormAssembler(const FESpace* fes) : fes_(fes) {
    if (fes_ && fes_->numDofs() > 0) {
        vec_.setZero(fes_->numDofs());
    }
#ifdef _OPENMP
    buffers_.resize(omp_get_max_threads());
#else
    buffers_.resize(1);
#endif
}

void LinearFormAssembler::assemble() {
    if (!fes_) return;
    vec_.setZero();
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    ElementTransform trans;
    trans.setMesh(mesh);
    
    // 域积分
    if (!domainIntegs_.empty()) {
        for (Index e = 0; e < mesh->numElements(); ++e) {
            const ReferenceElement* ref = fes_->elementRefElement(e);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            trans.setElement(e);
            
            ThreadBuffer& buf = buffers_[0];
            buf.elvec.setZero();
            
            for (const auto& integ : domainIntegs_) {
                Vector temp(nd);
                temp.setZero();
                integ->assembleElementVector(*ref, trans, temp);
                buf.elvec.head(nd) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getElementDofs(e, dofs);
            for (int i = 0; i < nd; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvec(i);
            }
        }
    }
    
    // 边界积分
    if (!bdrIntegs_.empty()) {
        FacetElementTransform btrans;
        btrans.setMesh(mesh);
        
        for (Index b = 0; b < mesh->numBdrElements(); ++b) {
            const ReferenceElement* ref = fes_->bdrElementRefElement(b);
            if (!ref) continue;
            int nd = ref->numDofs();
            
            btrans.setBoundaryElement(b);
            int attr = mesh->bdrElement(b).attribute();
            
            ThreadBuffer& buf = buffers_[0];
            buf.elvec.setZero();
            
            for (size_t k = 0; k < bdrIntegs_.size(); ++k) {
                if (bdrIds_[k] >= 0 && bdrIds_[k] != attr) continue;
                Vector temp(nd);
                temp.setZero();
                bdrIntegs_[k]->assembleFaceVector(*ref, btrans, temp);
                buf.elvec.head(nd) += temp;
            }
            
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            for (int i = 0; i < nd; ++i) {
                if (dofs[i] != InvalidIndex)
                    vec_(dofs[i]) += buf.elvec(i);
            }
        }
    }
}

}  // namespace mpfem