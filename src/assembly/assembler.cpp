#include "assembly/assembler.hpp"

namespace mpfem {

// =============================================================================
// BilinearFormAssembler Implementation
// =============================================================================

BilinearFormAssembler::BilinearFormAssembler(const FESpace* fes)
    : fes_(fes) {
    if (fes_) {
        Index ndofs = fes_->numDofs();
        mat_.resize(ndofs, ndofs);
        // Estimate non-zeros per row (rough estimate)
        mat_.reserve(ndofs * 27);  // For 3D with quadratic elements
    }
}

void BilinearFormAssembler::assemble() {
    assembleDomain();
    assembleBoundary();
}

void BilinearFormAssembler::assembleDomain() {
    if (!fes_) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    const Index numElements = mesh->numElements();
    
    // Pre-collect all triplets (element-level parallelization is sufficient)
    // Each element produces a small dense matrix, which is much smaller than
    // the global matrix, so we don't need thread-local storage per element
    
    std::vector<SparseMatrix::Triplet> allTriplets;
    allTriplets.reserve(numElements * 64);  // Rough estimate
    
    Matrix elmat;
    std::vector<Index> dofs;
    ElementTransform trans;
    trans.setMesh(mesh);
    
    for (Index e = 0; e < numElements; ++e) {
        const ReferenceElement* refElem = fes_->elementRefElement(e);
        if (!refElem) continue;
        
        // Setup element transform
        trans.setElement(e);
        
        // Initialize element matrix
        elmat.setZero(refElem->numDofs(), refElem->numDofs());
        
        // Apply all domain integrators
        for (const auto& integ : domainIntegs_) {
            Matrix temp;
            integ->assembleElementMatrix(*refElem, trans, temp);
            elmat += temp;
        }
        for (const auto& integ : domainIntegRefs_) {
            Matrix temp;
            integ->assembleElementMatrix(*refElem, trans, temp);
            elmat += temp;
        }
        
        // Get global DOF indices
        fes_->getElementDofs(e, dofs);
        
        // Collect triplets
        for (size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] == InvalidIndex) continue;
            for (size_t j = 0; j < dofs.size(); ++j) {
                if (dofs[j] == InvalidIndex) continue;
                allTriplets.emplace_back(dofs[i], dofs[j], elmat(i, j));
            }
        }
    }
    
    // Finalize assembly
    mat_.setFromTriplets(std::move(allTriplets));
}

void BilinearFormAssembler::assembleBoundary() {
    if (!fes_ || boundaryIntegs_.empty()) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    Matrix elmat;
    std::vector<Index> dofs;
    
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        const ReferenceElement* refElem = fes_->bdrElementRefElement(b);
        if (!refElem) continue;
        
        // Setup facet transform
        FacetElementTransform trans(mesh, b);
        
        // Initialize element matrix
        elmat.setZero(refElem->numDofs(), refElem->numDofs());
        
        // Apply all boundary integrators
        for (const auto& integ : boundaryIntegs_) {
            Matrix temp;
            integ->assembleFaceMatrix(*refElem, trans, temp);
            elmat += temp;
        }
        for (const auto& integ : boundaryIntegRefs_) {
            Matrix temp;
            integ->assembleFaceMatrix(*refElem, trans, temp);
            elmat += temp;
        }
        
        // Get global DOF indices
        fes_->getBdrElementDofs(b, dofs);
        
        // Assemble into global matrix
        for (size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] == InvalidIndex) continue;
            for (size_t j = 0; j < dofs.size(); ++j) {
                if (dofs[j] == InvalidIndex) continue;
                mat_.addTriplet(dofs[i], dofs[j], elmat(i, j));
            }
        }
    }
    
    // Finalize
    mat_.assemble();
}

void BilinearFormAssembler::assembleElement(Index elemIdx, Matrix& elmat) {
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) {
        elmat.setZero(0, 0);
        return;
    }
    
    // Setup element transform
    ElementTransform* trans = elemTrans_;
    if (!trans) {
        defaultTrans_.setMesh(fes_->mesh());
        defaultTrans_.setElement(elemIdx);
        trans = &defaultTrans_;
    }
    
    // Initialize element matrix
    elmat.setZero(refElem->numDofs(), refElem->numDofs());
    
    // Apply all domain integrators
    for (const auto& integ : domainIntegs_) {
        Matrix temp;
        integ->assembleElementMatrix(*refElem, *trans, temp);
        elmat += temp;
    }
    for (const auto& integ : domainIntegRefs_) {
        Matrix temp;
        integ->assembleElementMatrix(*refElem, *trans, temp);
        elmat += temp;
    }
}

void BilinearFormAssembler::assembleBoundaryMatrix(Index bdrIdx, Matrix& elmat) {
    const ReferenceElement* refElem = fes_->bdrElementRefElement(bdrIdx);
    if (!refElem) {
        elmat.setZero(0, 0);
        return;
    }
    
    // Setup facet transform
    FacetElementTransform trans(fes_->mesh(), bdrIdx);
    
    // Initialize element matrix
    elmat.setZero(refElem->numDofs(), refElem->numDofs());
    
    // Apply all boundary integrators
    for (const auto& integ : boundaryIntegs_) {
        Matrix temp;
        integ->assembleFaceMatrix(*refElem, trans, temp);
        elmat += temp;
    }
    for (const auto& integ : boundaryIntegRefs_) {
        Matrix temp;
        integ->assembleFaceMatrix(*refElem, trans, temp);
        elmat += temp;
    }
}

// =============================================================================
// LinearFormAssembler Implementation
// =============================================================================

LinearFormAssembler::LinearFormAssembler(const FESpace* fes)
    : fes_(fes) {
    if (fes_) {
        vec_.setZero(fes_->numDofs());
    }
}

void LinearFormAssembler::assemble() {
    assembleDomain();
    assembleBoundary();
}

void LinearFormAssembler::assembleDomain() {
    if (!fes_) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    const Index numElements = mesh->numElements();
    
    Vector elvec;
    std::vector<Index> dofs;
    ElementTransform trans;
    trans.setMesh(mesh);
    
    for (Index e = 0; e < numElements; ++e) {
        const ReferenceElement* refElem = fes_->elementRefElement(e);
        if (!refElem) continue;
        
        trans.setElement(e);
        
        elvec.setZero(refElem->numDofs());
        
        for (const auto& integ : domainIntegs_) {
            Vector temp;
            integ->assembleElementVector(*refElem, trans, temp);
            elvec += temp;
        }
        for (const auto& integ : domainIntegRefs_) {
            Vector temp;
            integ->assembleElementVector(*refElem, trans, temp);
            elvec += temp;
        }
        
        // Get global DOF indices
        fes_->getElementDofs(e, dofs);
        
        // Assemble into global vector
        for (size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] != InvalidIndex) {
                vec_(dofs[i]) += elvec(i);
            }
        }
    }
}

void LinearFormAssembler::assembleBoundary() {
    if (!fes_ || boundaryIntegs_.empty()) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    Vector elvec;
    std::vector<Index> dofs;
    
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        const ReferenceElement* refElem = fes_->bdrElementRefElement(b);
        if (!refElem) continue;
        
        bdrTrans_.setMesh(mesh);
        bdrTrans_.setBoundaryElement(b);
        
        elvec.setZero(refElem->numDofs());
        
        for (const auto& integ : boundaryIntegs_) {
            Vector temp;
            integ->assembleFaceVector(*refElem, bdrTrans_, temp);
            elvec += temp;
        }
        for (const auto& integ : boundaryIntegRefs_) {
            Vector temp;
            integ->assembleFaceVector(*refElem, bdrTrans_, temp);
            elvec += temp;
        }
        
        // Get global DOF indices
        fes_->getBdrElementDofs(b, dofs);
        
        // Assemble into global vector
        for (size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] != InvalidIndex) {
                vec_(dofs[i]) += elvec(i);
            }
        }
    }
}

void LinearFormAssembler::assembleElementVector(Index elemIdx, Vector& elvec) {
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) {
        elvec.setZero(0);
        return;
    }
    
    elemTrans_.setMesh(fes_->mesh());
    elemTrans_.setElement(elemIdx);
    
    elvec.setZero(refElem->numDofs());
    
    for (const auto& integ : domainIntegs_) {
        Vector temp;
        integ->assembleElementVector(*refElem, elemTrans_, temp);
        elvec += temp;
    }
    for (const auto& integ : domainIntegRefs_) {
        Vector temp;
        integ->assembleElementVector(*refElem, elemTrans_, temp);
        elvec += temp;
    }
}

void LinearFormAssembler::assembleBoundaryVector(Index bdrIdx, Vector& elvec) {
    const ReferenceElement* refElem = fes_->bdrElementRefElement(bdrIdx);
    if (!refElem) {
        elvec.setZero(0);
        return;
    }
    
    bdrTrans_.setMesh(fes_->mesh());
    bdrTrans_.setBoundaryElement(bdrIdx);
    
    elvec.setZero(refElem->numDofs());
    
    for (const auto& integ : boundaryIntegs_) {
        Vector temp;
        integ->assembleFaceVector(*refElem, bdrTrans_, temp);
        elvec += temp;
    }
    for (const auto& integ : boundaryIntegRefs_) {
        Vector temp;
        integ->assembleFaceVector(*refElem, bdrTrans_, temp);
        elvec += temp;
    }
}

// =============================================================================
// DirichletBC Implementation
// =============================================================================

int DirichletBC::apply(SparseMatrix& A, Vector& x, Vector& b) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    if (method_ == Method::Elimination) {
        // Build map for batch elimination
        std::map<Index, Real> dofValues;
        for (Index dof : constrainedDofs_) {
            dofValues[dof] = value_;
        }
        A.eliminateRows(dofValues, b);
        
        // Set solution values
        for (Index dof : constrainedDofs_) {
            x(dof) = value_;
        }
    }
    else if (method_ == Method::Penalty) {
        for (Index dof : constrainedDofs_) {
            x(dof) = value_;
            A.addTriplet(dof, dof, penalty_);
            b(dof) += penalty_ * value_;
        }
        A.assemble();
    }
    
    return nConstrained;
}

int DirichletBC::applyToMatrix(SparseMatrix& A) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    for (Index dof : constrainedDofs_) {
        A.addTriplet(dof, dof, penalty_);
    }
    A.assemble();
    
    return nConstrained;
}

int DirichletBC::applyToVector(Vector& x, Vector& b) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    for (Index dof : constrainedDofs_) {
        x(dof) = value_;
        b(dof) = value_;
    }
    
    return nConstrained;
}

void DirichletBC::buildConstrainedDofs() {
    constrainedDofs_.clear();
    if (!fes_) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    // Collect DOFs on specified boundaries
    std::set<Index> dofSet;
    
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        const Element& bdrElem = mesh->bdrElement(b);
        int attr = static_cast<int>(bdrElem.attribute());
        
        // Check if this boundary element is on a constrained boundary
        bool isConstrained = false;
        for (int id : boundaryIds_) {
            if (id == attr) {
                isConstrained = true;
                break;
            }
        }
        
        if (isConstrained) {
            std::vector<Index> dofs;
            fes_->getBdrElementDofs(b, dofs);
            for (Index dof : dofs) {
                if (dof != InvalidIndex) {
                    dofSet.insert(dof);
                }
            }
        }
    }
    
    constrainedDofs_.assign(dofSet.begin(), dofSet.end());
}

// =============================================================================
// SystemAssembler Implementation
// =============================================================================

SystemAssembler::SystemAssembler(const FESpace* fes)
    : fes_(fes), 
      bilinearAsm_(fes), 
      linearAsm_(fes) {
    if (fes_) {
        solution_.setZero(fes_->numDofs());
    }
}

void SystemAssembler::setFESpace(const FESpace* fes) {
    fes_ = fes;
    bilinearAsm_.setFESpace(fes);
    linearAsm_.setFESpace(fes);
    if (fes_) {
        solution_.setZero(fes_->numDofs());
    }
}

void SystemAssembler::addDirichletBC(int boundaryId, Real value) {
    DirichletBC bc(fes_);
    bc.addBoundaryId(boundaryId);
    bc.setValue(value);
    bc.setMethod(dirichletMethod_);
    dirichletBCs_.push_back(std::move(bc));
}

void SystemAssembler::addDirichletBC(int boundaryId, Coefficient* coef) {
    DirichletBC bc(fes_);
    bc.addBoundaryId(boundaryId);
    bc.setCoefficient(coef);
    bc.setMethod(dirichletMethod_);
    dirichletBCs_.push_back(std::move(bc));
}

void SystemAssembler::assemble() {
    // Assemble bilinear form
    bilinearAsm_.assemble();
    
    // Assemble linear form
    linearAsm_.assemble();
    
    // Apply Dirichlet BCs
    for (auto& bc : dirichletBCs_) {
        bc.apply(bilinearAsm_.matrix(), solution_, linearAsm_.vector());
    }
    
    // Collect constrained DOFs
    constrainedDofs_.clear();
    for (const auto& bc : dirichletBCs_) {
        const auto& dofs = bc.constrainedDofs();
        constrainedDofs_.insert(constrainedDofs_.end(), dofs.begin(), dofs.end());
    }
    
    // Finalize matrix
    bilinearAsm_.finalize();
}

bool SystemAssembler::solve() {
    if (!solver_) {
        return false;
    }
    
    return solver_->solve(bilinearAsm_.matrix(), solution_, linearAsm_.vector());
}

}  // namespace mpfem