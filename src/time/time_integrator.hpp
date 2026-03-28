#ifndef MPFEM_TIME_INTEGRATOR_HPP
#define MPFEM_TIME_INTEGRATOR_HPP

#include "core/types.hpp"
#include "time/time_scheme.hpp"
#include "solver/sparse_matrix.hpp"

namespace mpfem {

class TransientProblem;

class TimeIntegrator {

public:
    virtual ~TimeIntegrator() = default;
    
    virtual bool step(TransientProblem& problem) = 0;

protected:
    SparseMatrix A_;   // Pre-allocated system matrix (reused every step)
    Vector rhs_;       // Pre-allocated RHS vector (reused every step)
    bool initialized_ = false;  // Track if A_ and rhs_ are sized
    
    // Helper to ensure A_ and rhs_ are sized correctly
    void ensureSize(Index nRows, Index nCols) {
        if (!initialized_ || A_.rows() != nRows || A_.cols() != nCols) {
            A_.resize(nRows, nCols);
            rhs_.resize(nRows);
            initialized_ = true;
        }
    }
};

TimeIntegrator* createTimeIntegrator(TimeScheme scheme);

}  // namespace mpfem

#endif  // MPFEM_TIME_INTEGRATOR_HPP
