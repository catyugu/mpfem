/**
 * @file pardiso_solver.hpp
 * @brief MKL PARDISO direct solver (optional)
 * 
 * This solver is only available when MPFEM_ENABLE_PARDISO is defined
 * and Intel MKL is linked.
 */

#ifndef MPFEM_LINALG_PARDISO_SOLVER_HPP
#define MPFEM_LINALG_PARDISO_SOLVER_HPP

#include "solver_base.hpp"

// Only compile if PARDISO is enabled
#ifdef MPFEM_ENABLE_PARDISO

namespace mpfem {

/**
 * @brief PARDISO solver using Intel MKL
 * 
 * High-performance direct solver for large sparse systems.
 * Requires Intel MKL to be linked.
 */
class PardisoSolver : public SolverBase {
public:
    /**
     * @brief Matrix type for PARDISO
     */
    enum class MatrixType {
        RealStructurallySymmetric = 1,
        RealSymmetricPositiveDefinite = 2,
        RealSymmetricIndefinite = -2,
        RealNonsymmetric = 11
    };
    
    PardisoSolver();
    ~PardisoSolver() override;
    
    /**
     * @brief Set matrix type
     */
    void set_matrix_type(MatrixType type) { matrix_type_ = type; }
    
    /**
     * @brief Set number of threads for PARDISO
     */
    void set_num_threads(int n) { num_threads_ = n; }
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;
    
    /**
     * @brief Clear stored factorization
     */
    void clear();

private:
    void* pt_[64] = {};         ///< PARDISO internal data
    int iparm_[64] = {};        ///< PARDISO parameters
    int matrix_type_ = 11;      ///< Default: real nonsymmetric
    int num_threads_ = 1;
    bool initialized_ = false;
    Index n_dofs_ = 0;
    Index nnz_ = 0;
    
    // Store matrix in CSR format for PARDISO
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<double> values_;
    
    void initialize_pardiso();
    void release_pardiso();
};

} // namespace mpfem

#endif // MPFEM_ENABLE_PARDISO

#endif // MPFEM_LINALG_PARDISO_SOLVER_HPP
