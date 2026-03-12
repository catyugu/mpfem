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
    
    /// Assemble the global matrix
    void assemble();
    
    /// Assemble only domain contributions
    void assembleDomain();
    
    /// Assemble only boundary contributions
    void assembleBoundary();
    
    /// Assemble for a specific element (for matrix-free or testing)
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
    void finalize() { mat_.makeCompressed(); }
    
    /// Clear the matrix
    void clear() { mat_.clear(); }
    
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
    
    /// Assemble the global vector
    void assemble();
    
    /// Assemble only domain contributions
    void assembleDomain();
    
    /// Assemble only boundary contributions
    void assembleBoundary();
    
    // -------------------------------------------------------------------------
    // Vector access
    // -------------------------------------------------------------------------
    
    /// Get the assembled vector
    Vector& vector() { return vec_; }
    const Vector& vector() const { return vec_; }
    
    /// Clear the vector
    void clear() { vec_.setZero(); }
    
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
    void setBoundaryIds(const std::vector<int>& ids) { boundaryIds_ = ids; }
    
    /// Add a boundary ID
    void addBoundaryId(int id) { boundaryIds_.push_back(id); }
    
    /// Set the boundary value (constant)
    void setValue(Real value) { value_ = value; hasCoefficient_ = false; }
    
    /// Set the boundary value (from coefficient)
    void setCoefficient(Coefficient* coef) { coef_ = coef; hasCoefficient_ = true; }
    
    /// Set the method
    void setMethod(Method method) { method_ = method; }
    
    /// Set penalty parameter (for penalty method)
    void setPenalty(Real penalty) { penalty_ = penalty; }
    
    // -------------------------------------------------------------------------
    // Apply boundary condition
    // -------------------------------------------------------------------------
    
    /// Apply Dirichlet BC to the linear system
    int apply(SparseMatrix& A, Vector& x, Vector& b);
    
    /// Apply Dirichlet BC to matrix only
    int applyToMatrix(SparseMatrix& A);
    
    /// Apply Dirichlet BC to RHS vector only
    int applyToVector(Vector& x, Vector& b);
    
    /// Get the list of constrained DOF indices
    const std::vector<Index>& constrainedDofs() const { return constrainedDofs_; }
    
    /// Build the list of constrained DOFs
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
    void setDirichletMethod(DirichletBC::Method method) { dirichletMethod_ = method; }
    
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
    const std::vector<Index>& constrainedDofs() const { return constrainedDofs_; }
    
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

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLER_HPP