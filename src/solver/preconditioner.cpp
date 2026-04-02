#include "solver/preconditioner.hpp"

namespace mpfem {

    // =============================================================================
    // DiagonalPreconditioner Implementation
    // =============================================================================

    void DiagonalPreconditioner::apply(const SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!initialized_) {
            // Manually compute inverse diagonal
            const auto diag = A.eigen().diagonal();
            invDiag_.resize(diag.size());
            for (int i = 0; i < diag.size(); ++i) {
                invDiag_[i] = (std::abs(diag[i]) > 1e-12) ? 1.0 / diag[i] : 1.0;
            }
            initialized_ = true;
        }
        x = invDiag_.array() * b.array();
    }

    void DiagonalPreconditioner::setParameters(const std::map<std::string, Real>& params)
    {
        (void)params; // Diagonal preconditioner has no tunable parameters
    }

    std::string DiagonalPreconditioner::name() const
    {
        return "DiagonalPreconditioner";
    }

    // =============================================================================
    // ICCPreconditioner Implementation
    // =============================================================================

    void ICCPreconditioner::apply(const SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!initialized_ || fingerprint_ != A.fingerprint()) {
            preconditioner_.setInitialShift(shift_);
            preconditioner_.compute(A.eigen());
            fingerprint_ = A.fingerprint();
            initialized_ = true;
        }
        x = preconditioner_.solve(b);
    }

    void ICCPreconditioner::setParameters(const std::map<std::string, Real>& params)
    {
        auto it = params.find("shift");
        if (it != params.end()) {
            shift_ = it->second;
            initialized_ = false; // Force recompute with new shift
        }
    }

    std::string ICCPreconditioner::name() const
    {
        return "ICCPreconditioner";
    }

    // =============================================================================
    // ILUPreconditioner Implementation
    // =============================================================================

    void ILUPreconditioner::apply(const SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!initialized_ || fingerprint_ != A.fingerprint()) {
            preconditioner_.setDroptol(dropTolerance_);
            preconditioner_.setFillfactor(fillLevel_);
            preconditioner_.compute(A.eigen());
            fingerprint_ = A.fingerprint();
            initialized_ = true;
        }
        x = preconditioner_.solve(b);
    }

    void ILUPreconditioner::setParameters(const std::map<std::string, Real>& params)
    {
        bool needRecompute = false;

        auto fillIt = params.find("fillLevel");
        if (fillIt != params.end()) {
            int newLevel = static_cast<int>(fillIt->second);
            if (newLevel != fillLevel_) {
                fillLevel_ = newLevel;
                needRecompute = true;
            }
        }

        auto dropIt = params.find("dropTolerance");
        if (dropIt != params.end()) {
            Real newTol = dropIt->second;
            if (newTol != dropTolerance_) {
                dropTolerance_ = newTol;
                needRecompute = true;
            }
        }

        if (needRecompute) {
            initialized_ = false;
        }
    }

    std::string ILUPreconditioner::name() const
    {
        return "ILUPreconditioner";
    }

} // namespace mpfem
