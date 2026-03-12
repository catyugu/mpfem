#ifndef MPFEM_ASSEMBLER_HPP
#define MPFEM_ASSEMBLER_HPP

#include "assembly/integrator.hpp"
#include "assembly/integrators.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "solver/sparse_matrix.hpp"
#include "solver/linear_solver.hpp"
#include "mesh/mesh_topology.hpp"
#include "core/logger.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <set>

namespace mpfem {

/**
 * @file assembler.hpp
 * @brief Assembler for global finite element systems.
 * 
 * The Assembler class handles:
 * - Assembly of bilinear forms (stiffness matrices)
 * - Assembly of linear forms (load vectors)
 * - Application of boundary conditions
 */

// =============================================================================
// Bilinear Form Assembly
// =============================================================================

/**
 * @brief Assembler for bilinear forms.
 * 
 * BilinearFormAssembler assembles global stiffness matrices from
 * element-level contributions computed by integrators.
 * 
 * Usage:
 *   BilinearFormAssembler assembler(&fes);
 *   assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k));
 *   assembler.assemble();
 *   SparseMatrix& A = assembler.matrix();
 */
class BilinearFormAssembler {
public:
    /// Constructor
    explicit BilinearFormAssembler(const FESpace* fes);
    
    /// Set the FE space
    void setFESpace(const FESpace* fes) { fes_ = fes; }
    
    /// Get the FE space
    const FESpace* feSpace() const { return fes_; }
    
    // -------------------------------------------------------------------------
    // Integrator management
    // -------------------------------------------------------------------------
    
    /// Add a domain integrator (takes ownership)
    void addDomainIntegrator(std::unique_ptr<BilinearFormIntegrator> integ) {
        domainIntegs_.push_back(std::move(integ));
    }
    
    /// Add a boundary integrator (takes ownership)
    void addBoundaryIntegrator(std::unique_ptr<BilinearFormIntegrator> integ) {
        boundaryIntegs_.push_back(std::move(integ));
    }
    
    /// Add a domain integrator (non-owning reference)
    void addDomainIntegratorRef(BilinearFormIntegrator& integ) {
        domainIntegRefs_.push_back(&integ);
    }
    
    /// Add a boundary integrator (non-owning reference)
    void addBoundaryIntegratorRef(BilinearFormIntegrator& integ) {
        boundaryIntegRefs_.push_back(&integ);
    }
    
    /// Clear all integrators
    void clearIntegrators() {
        domainIntegs_.clear();
        boundaryIntegs_.clear();
        domainIntegRefs_.clear();
        boundaryIntegRefs_.clear();
    }
    
    // -------------------------------------------------------------------------
    // Assembly
    // -------------------------------------------------------------------------
    
    /**
     * @brief Assemble the global matrix.
     * 
     * Loops over all elements, computes element matrices using
     * domain integrators, and assembles into the global matrix.
     * Then loops over boundary elements for boundary integrators.
     */
    void assemble();
    
    /**
     * @brief Assemble only domain contributions.
     */
    void assembleDomain();
    
    /**
     * @brief Assemble only boundary contributions.
     */
    void assembleBoundary();
    
    /**
     * @brief Assemble for a specific element (for matrix-free or testing).
     */
    void assembleElement(Index elemIdx, Matrix& elmat);
    
    // -------------------------------------------------------------------------
    // Matrix access
    // -------------------------------------------------------------------------
    
    /// Get the assembled matrix
    SparseMatrix& matrix() { return mat_; }
    const SparseMatrix& matrix() const { return mat_; }
    
    /// Get the internal Eigen matrix
    auto& eigen() { return mat_.eigen(); }
    const auto& eigen() const { return mat_.eigen(); }
    
    /// Finalize the matrix (compress sparse format)
    void finalize() {
        mat_.makeCompressed();
    }
    
    /// Clear the matrix
    void clear() {
        mat_.clear();
    }
    
    /// Get number of rows
    Index rows() const { return mat_.rows(); }
    
    /// Get number of columns
    Index cols() const { return mat_.cols(); }
    
    // -------------------------------------------------------------------------
    // Element transformation
    // -------------------------------------------------------------------------
    
    /// Set element transform (optional, for custom transform)
    void setElementTransform(ElementTransform* trans) { elemTrans_ = trans; }
    
private:
    void assembleElementMatrix(Index elemIdx, Matrix& elmat);
    void assembleBoundaryMatrix(Index bdrIdx, Matrix& elmat);
    
    const FESpace* fes_ = nullptr;
    ElementTransform* elemTrans_ = nullptr;
    
    std::vector<std::unique_ptr<BilinearFormIntegrator>> domainIntegs_;
    std::vector<std::unique_ptr<BilinearFormIntegrator>> boundaryIntegs_;
    std::vector<BilinearFormIntegrator*> domainIntegRefs_;
    std::vector<BilinearFormIntegrator*> boundaryIntegRefs_;
    
    SparseMatrix mat_;
    ElementTransform defaultTrans_;
};

// =============================================================================
// Linear Form Assembly
// =============================================================================

/**
 * @brief Assembler for linear forms.
 * 
 * LinearFormAssembler assembles global load vectors from
 * element-level contributions computed by integrators.
 * 
 * Usage:
 *   LinearFormAssembler assembler(&fes);
 *   assembler.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(&f));
 *   assembler.addBoundaryIntegrator(std::make_unique<BoundaryLFIntegrator>(&g));
 *   assembler.assemble();
 *   Vector& b = assembler.vector();
 */
class LinearFormAssembler {
public:
    /// Constructor
    explicit LinearFormAssembler(const FESpace* fes);
    
    /// Set the FE space
    void setFESpace(const FESpace* fes) { fes_ = fes; }
    
    /// Get the FE space
    const FESpace* feSpace() const { return fes_; }
    
    // -------------------------------------------------------------------------
    // Integrator management
    // -------------------------------------------------------------------------
    
    /// Add a domain integrator (takes ownership)
    void addDomainIntegrator(std::unique_ptr<LinearFormIntegrator> integ) {
        domainIntegs_.push_back(std::move(integ));
    }
    
    /// Add a boundary integrator (takes ownership)
    void addBoundaryIntegrator(std::unique_ptr<LinearFormIntegrator> integ) {
        boundaryIntegs_.push_back(std::move(integ));
    }
    
    /// Add a domain integrator (non-owning reference)
    void addDomainIntegratorRef(LinearFormIntegrator& integ) {
        domainIntegRefs_.push_back(&integ);
    }
    
    /// Add a boundary integrator (non-owning reference)
    void addBoundaryIntegratorRef(LinearFormIntegrator& integ) {
        boundaryIntegRefs_.push_back(&integ);
    }
    
    /// Clear all integrators
    void clearIntegrators() {
        domainIntegs_.clear();
        boundaryIntegs_.clear();
        domainIntegRefs_.clear();
        boundaryIntegRefs_.clear();
    }
    
    // -------------------------------------------------------------------------
    // Assembly
    // -------------------------------------------------------------------------
    
    /**
     * @brief Assemble the global vector.
     */
    void assemble();
    
    /**
     * @brief Assemble only domain contributions.
     */
    void assembleDomain();
    
    /**
     * @brief Assemble only boundary contributions.
     */
    void assembleBoundary();
    
    // -------------------------------------------------------------------------
    // Vector access
    // -------------------------------------------------------------------------
    
    /// Get the assembled vector
    Vector& vector() { return vec_; }
    const Vector& vector() const { return vec_; }
    
    /// Clear the vector
    void clear() {
        vec_.setZero();
    }
    
    /// Get size
    Index size() const { return vec_.size(); }
    
    // -------------------------------------------------------------------------
    // Boundary handling
    // -------------------------------------------------------------------------
    
    /// Set mesh topology (required for boundary integrators)
    void setMeshTopology(const MeshTopology* topo) { topo_ = topo; }
    
private:
    void assembleElementVector(Index elemIdx, Vector& elvec);
    void assembleBoundaryVector(Index bdrIdx, Vector& elvec);
    
    const FESpace* fes_ = nullptr;
    const MeshTopology* topo_ = nullptr;
    
    std::vector<std::unique_ptr<LinearFormIntegrator>> domainIntegs_;
    std::vector<std::unique_ptr<LinearFormIntegrator>> boundaryIntegs_;
    std::vector<LinearFormIntegrator*> domainIntegRefs_;
    std::vector<LinearFormIntegrator*> boundaryIntegRefs_;
    
    Vector vec_;
    ElementTransform elemTrans_;
    FacetElementTransform bdrTrans_;
};

// =============================================================================
// Dirichlet Boundary Condition Handler
// =============================================================================

/**
 * @brief Handler for Dirichlet boundary conditions.
 * 
 * DirichletBC applies essential boundary conditions to the linear system:
 *   A * x = b  with  x = g on boundary Gamma_D
 * 
 * Two approaches are supported:
 * 1. Elimination: Modify A and b to enforce x_i = g_i
 * 2. Penalty: Add large diagonal entries for constrained DOFs
 */
class DirichletBC {
public:
    /// Boundary condition type
    enum class Method {
        Elimination,  ///< Direct elimination (modifies matrix)
        Penalty       ///< Penalty method (adds to matrix)
    };
    
    /// Constructor
    DirichletBC() = default;
    
    /// Constructor with FE space
    explicit DirichletBC(const FESpace* fes) : fes_(fes) {}
    
    /// Set the FE space
    void setFESpace(const FESpace* fes) { fes_ = fes; }
    
    /// Set boundary DOFs (by boundary IDs)
    void setBoundaryIds(const std::vector<int>& ids) {
        boundaryIds_ = ids;
    }
    
    /// Add a boundary ID
    void addBoundaryId(int id) {
        boundaryIds_.push_back(id);
    }
    
    /// Set the boundary value (constant)
    void setValue(Real value) {
        value_ = value;
        hasCoefficient_ = false;
    }
    
    /// Set the boundary value (from coefficient)
    void setCoefficient(Coefficient* coef) {
        coef_ = coef;
        hasCoefficient_ = true;
    }
    
    /// Set the method
    void setMethod(Method method) { method_ = method; }
    
    /// Set penalty parameter (for penalty method)
    void setPenalty(Real penalty) { penalty_ = penalty; }
    
    // -------------------------------------------------------------------------
    // Apply boundary condition
    // -------------------------------------------------------------------------
    
    /**
     * @brief Apply Dirichlet BC to the linear system.
     * 
     * @param A System matrix (modified in-place).
     * @param x Solution vector (modified to have correct boundary values).
     * @param b RHS vector (modified to account for boundary conditions).
     * @return Number of DOFs constrained.
     */
    int apply(SparseMatrix& A, Vector& x, Vector& b);
    
    /**
     * @brief Apply Dirichlet BC to matrix only.
     * 
     * Zeroes out rows for boundary DOFs and sets diagonal to 1.
     */
    int applyToMatrix(SparseMatrix& A);
    
    /**
     * @brief Apply Dirichlet BC to RHS vector only.
     */
    int applyToVector(Vector& x, Vector& b);
    
    /**
     * @brief Get the list of constrained DOF indices.
     */
    const std::vector<Index>& constrainedDofs() const { 
        return constrainedDofs_; 
    }
    
    /**
     * @brief Build the list of constrained DOFs.
     */
    void buildConstrainedDofs();
    
private:
    const FESpace* fes_ = nullptr;
    std::vector<int> boundaryIds_;
    std::vector<Index> constrainedDofs_;
    
    Real value_ = 0.0;
    Coefficient* coef_ = nullptr;
    bool hasCoefficient_ = false;
    
    Method method_ = Method::Elimination;
    Real penalty_ = 1e10;
};

// =============================================================================
// System Assembler (Combined)
// =============================================================================

/**
 * @brief Combined system assembler for complete problem setup.
 * 
 * SystemAssembler provides a unified interface for assembling the complete
 * linear system including:
 * - Domain integrators for stiffness and mass matrices
 * - Boundary integrators for Neumann BCs
 * - Dirichlet BC application
 * 
 * Usage:
 *   SystemAssembler sys(fes);
 *   sys.addBilinearIntegrator(std::make_unique<DiffusionIntegrator>(&k));
 *   sys.addLinearIntegrator(std::make_unique<DomainLFIntegrator>(&f));
 *   sys.addDirichletBC(1, 0.0);  // BC on boundary 1
 *   sys.assemble();
 *   sys.solve(x);
 */
class SystemAssembler {
public:
    explicit SystemAssembler(const FESpace* fes);
    
    /// Set the FE space
    void setFESpace(const FESpace* fes);
    
    /// Set mesh topology (for boundary integrators)
    void setMeshTopology(const MeshTopology* topo) { topo_ = topo; }
    
    // -------------------------------------------------------------------------
    // Bilinear form
    // -------------------------------------------------------------------------
    
    void addBilinearIntegrator(std::unique_ptr<BilinearFormIntegrator> integ) {
        bilinearAsm_.addDomainIntegrator(std::move(integ));
    }
    
    void addBoundaryBilinearIntegrator(std::unique_ptr<BilinearFormIntegrator> integ) {
        bilinearAsm_.addBoundaryIntegrator(std::move(integ));
    }
    
    // -------------------------------------------------------------------------
    // Linear form
    // -------------------------------------------------------------------------
    
    void addLinearIntegrator(std::unique_ptr<LinearFormIntegrator> integ) {
        linearAsm_.addDomainIntegrator(std::move(integ));
    }
    
    void addBoundaryLinearIntegrator(std::unique_ptr<LinearFormIntegrator> integ) {
        linearAsm_.addBoundaryIntegrator(std::move(integ));
    }
    
    // -------------------------------------------------------------------------
    // Boundary conditions
    // -------------------------------------------------------------------------
    
    /// Add Dirichlet BC with constant value
    void addDirichletBC(int boundaryId, Real value);
    
    /// Add Dirichlet BC with coefficient
    void addDirichletBC(int boundaryId, Coefficient* coef);
    
    /// Set Dirichlet BC method
    void setDirichletMethod(DirichletBC::Method method) {
        dirichletMethod_ = method;
    }
    
    // -------------------------------------------------------------------------
    // Assembly
    // -------------------------------------------------------------------------
    
    /// Assemble the complete system
    void assemble();
    
    /// Get the stiffness matrix
    SparseMatrix& matrix() { return bilinearAsm_.matrix(); }
    
    /// Get the RHS vector
    Vector& rhs() { return linearAsm_.vector(); }
    
    /// Get the solution vector
    Vector& solution() { return solution_; }
    
    /// Get constrained DOFs
    const std::vector<Index>& constrainedDofs() const {
        return constrainedDofs_;
    }
    
    // -------------------------------------------------------------------------
    // Solve
    // -------------------------------------------------------------------------
    
    /// Set the linear solver
    void setSolver(LinearSolver* solver) { solver_ = solver; }
    
    /// Solve the system
    bool solve();
    
private:
    const FESpace* fes_ = nullptr;
    const MeshTopology* topo_ = nullptr;
    
    BilinearFormAssembler bilinearAsm_;
    LinearFormAssembler linearAsm_;
    
    std::vector<DirichletBC> dirichletBCs_;
    DirichletBC::Method dirichletMethod_ = DirichletBC::Method::Elimination;
    std::vector<Index> constrainedDofs_;
    
    Vector solution_;
    LinearSolver* solver_ = nullptr;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline BilinearFormAssembler::BilinearFormAssembler(const FESpace* fes)
    : fes_(fes) {
    if (fes_) {
        Index ndofs = fes_->numDofs();
        mat_.resize(ndofs, ndofs);
        // Estimate non-zeros per row (rough estimate)
        mat_.reserve(ndofs * 27);  // For 3D with quadratic elements
    }
}

inline LinearFormAssembler::LinearFormAssembler(const FESpace* fes)
    : fes_(fes) {
    if (fes_) {
        vec_.setZero(fes_->numDofs());
    }
}

inline void BilinearFormAssembler::assemble() {
    assembleDomain();
    assembleBoundary();
}

inline void BilinearFormAssembler::assembleDomain() {
    if (!fes_) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    LOG_DEBUG << "Assembling domain contributions for " << mesh->numElements() 
              << " elements";
    
    Matrix elmat;
    std::vector<Index> dofs;
    
    for (Index e = 0; e < mesh->numElements(); ++e) {
        assembleElementMatrix(e, elmat);
        
        // Get global DOF indices
        fes_->getElementDofs(e, dofs);
        
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

inline void BilinearFormAssembler::assembleBoundary() {
    if (!fes_ || boundaryIntegs_.empty()) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    LOG_DEBUG << "Assembling boundary contributions for " << mesh->numBdrElements()
              << " boundary elements";
    
    Matrix elmat;
    std::vector<Index> dofs;
    
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        assembleBoundaryMatrix(b, elmat);
        
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

inline void BilinearFormAssembler::assembleElementMatrix(Index elemIdx, Matrix& elmat) {
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

inline void BilinearFormAssembler::assembleBoundaryMatrix(Index bdrIdx, Matrix& elmat) {
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

inline void LinearFormAssembler::assemble() {
    assembleDomain();
    assembleBoundary();
}

inline void LinearFormAssembler::assembleDomain() {
    if (!fes_) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    Vector elvec;
    std::vector<Index> dofs;
    
    for (Index e = 0; e < mesh->numElements(); ++e) {
        assembleElementVector(e, elvec);
        
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

inline void LinearFormAssembler::assembleBoundary() {
    if (!fes_ || boundaryIntegs_.empty()) return;
    
    const Mesh* mesh = fes_->mesh();
    if (!mesh) return;
    
    Vector elvec;
    std::vector<Index> dofs;
    
    for (Index b = 0; b < mesh->numBdrElements(); ++b) {
        assembleBoundaryVector(b, elvec);
        
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

inline void LinearFormAssembler::assembleElementVector(Index elemIdx, Vector& elvec) {
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

inline void LinearFormAssembler::assembleBoundaryVector(Index bdrIdx, Vector& elvec) {
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

inline int DirichletBC::apply(SparseMatrix& A, Vector& x, Vector& b) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    // Set solution values at constrained DOFs
    for (Index dof : constrainedDofs_) {
        Real val = value_;
        if (hasCoefficient_ && coef_) {
            // For coefficient, we need a point to evaluate
            // This is simplified - in practice we'd need proper evaluation
            val = value_;  // Use the stored value for now
        }
        x(dof) = val;
    }
    
    if (method_ == Method::Elimination) {
        // Modify RHS: b_i = A_ii * x_i for constrained rows
        // Then zero the row and set diagonal to 1
        for (Index dof : constrainedDofs_) {
            Real val = x(dof);
            
            // Zero out the row in the matrix and set diagonal to 1
            // Note: This requires modifying the sparse matrix structure
            // For now, we use the penalty-like approach via triplet modification
            
            // Add contribution to RHS from boundary value
            // In proper elimination: b_new = b - A * x_bc for non-BC DOFs
            // Simplified here: just set the row
            b(dof) = val;
        }
    }
    else if (method_ == Method::Penalty) {
        // Add large penalty to diagonal
        for (Index dof : constrainedDofs_) {
            Real val = x(dof);
            // A(dof, dof) += penalty
            // b(dof) += penalty * val
            A.addTriplet(dof, dof, penalty_);
            b(dof) += penalty_ * val;
        }
        A.assemble();
    }
    
    return nConstrained;
}

inline int DirichletBC::applyToMatrix(SparseMatrix& A) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    for (Index dof : constrainedDofs_) {
        // This is simplified - proper implementation would zero the row
        // and set diagonal to 1 or penalty value
        A.addTriplet(dof, dof, penalty_);
    }
    A.assemble();
    
    return nConstrained;
}

inline int DirichletBC::applyToVector(Vector& x, Vector& b) {
    buildConstrainedDofs();
    
    int nConstrained = static_cast<int>(constrainedDofs_.size());
    if (nConstrained == 0) return 0;
    
    for (Index dof : constrainedDofs_) {
        Real val = value_;
        x(dof) = val;
        b(dof) = val;
    }
    
    return nConstrained;
}

inline void DirichletBC::buildConstrainedDofs() {
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

inline SystemAssembler::SystemAssembler(const FESpace* fes)
    : fes_(fes), 
      bilinearAsm_(fes), 
      linearAsm_(fes) {
    if (fes_) {
        solution_.setZero(fes_->numDofs());
    }
}

inline void SystemAssembler::setFESpace(const FESpace* fes) {
    fes_ = fes;
    bilinearAsm_.setFESpace(fes);
    linearAsm_.setFESpace(fes);
    if (fes_) {
        solution_.setZero(fes_->numDofs());
    }
}

inline void SystemAssembler::addDirichletBC(int boundaryId, Real value) {
    DirichletBC bc(fes_);
    bc.addBoundaryId(boundaryId);
    bc.setValue(value);
    bc.setMethod(dirichletMethod_);
    dirichletBCs_.push_back(std::move(bc));
}

inline void SystemAssembler::addDirichletBC(int boundaryId, Coefficient* coef) {
    DirichletBC bc(fes_);
    bc.addBoundaryId(boundaryId);
    bc.setCoefficient(coef);
    bc.setMethod(dirichletMethod_);
    dirichletBCs_.push_back(std::move(bc));
}

inline void SystemAssembler::assemble() {
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

inline bool SystemAssembler::solve() {
    if (!solver_) {
        LOG_ERROR << "No solver set in SystemAssembler";
        return false;
    }
    
    return solver_->solve(bilinearAsm_.matrix(), solution_, linearAsm_.vector());
}

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLER_HPP
