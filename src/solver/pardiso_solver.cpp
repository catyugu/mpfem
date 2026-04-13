#include "solver/pardiso_solver.hpp"

namespace mpfem {

#ifdef MPFEM_USE_MKL

    PardisoSolver::PardisoSolver()
    {
        // Initialize handle array to null
        std::memset(pt_, 0, sizeof(pt_));

        // Initialize iparm with default values using pardisoinit
        mtype_ = 11; // Real unsymmetric matrix
        pardisoinit(pt_, &mtype_, iparm_);

        // Override some default parameters
        iparm_[0] = 1; // Use user-defined iparm values
        iparm_[1] = 2; // Fill-in reducing: METIS
        // Note: iparm_[34] = 0 (default) means 1-based indexing for ia/ja arrays

        maxfct_ = 1;
        mnum_ = 1;
        phase_ = 0;
        msglvl_ = 0; // No output
        error_ = 0;
    }

    PardisoSolver::~PardisoSolver()
    {
        // Release all memory
        if (initialized_) {
            phase_ = -1;
            MKL_INT nrhs = 1;
            MKL_INT dummy_n = 0;
            pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                &dummy_n, nullptr, nullptr, nullptr,
                nullptr, &nrhs, iparm_, &msglvl_, nullptr, nullptr, &error_);
        }
    }

    std::string_view PardisoSolver::name() const
    {
        return "Pardiso";
    }

    void PardisoSolver::setup(const SparseMatrix* A)
    {
        if (!A) {
            throw std::runtime_error("PardisoSolver: null matrix in setup");
        }

        const auto& mat = A->eigen();
        n_ = static_cast<MKL_INT>(mat.rows());
        convertToCSR(mat);
        phase_ = 12;

        MKL_INT nrhs = 1;
        initialized_ = true;

        // Factorize
        Vector temp(n_);
        temp.setZero();
        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
            &n_, a_.data(), ia_.data(), ja_.data(),
            nullptr, &nrhs, iparm_, &msglvl_,
            temp.data(), temp.data(), &error_);

        if (error_ != 0) {
            LOG_ERROR << "PARDISO factorization failed with error: " << error_;
            throw std::runtime_error("PardisoSolver: factorization failed");
        }

        set_matrix(A);
        mark_setup();
    }

    void PardisoSolver::apply(const Vector& b, Vector& x)
    {
        if (!is_setup()) {
            throw std::runtime_error("PardisoSolver: not setup");
        }

        MKL_INT nrhs = 1;
        phase_ = 33; // Solve

        // PARDISO modifies RHS, so copy to buffer
        Vector b_copy = b;
        x.resize(n_);

        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
            &n_, a_.data(), ia_.data(), ja_.data(),
            nullptr, &nrhs, iparm_, &msglvl_,
            b_copy.data(), x.data(), &error_);

        if (error_ != 0) {
            LOG_ERROR << "PARDISO solve failed with error: " << error_;
            throw std::runtime_error("PardisoSolver: solve failed");
        }

        iterations_ = 1;
        residual_ = 0.0;
    }

    int PardisoSolver::iterations() const
    {
        return iterations_;
    }

    Real PardisoSolver::residual() const
    {
        return residual_;
    }

    void PardisoSolver::convertToCSR(const SparseMatrix::Storage& mat)
    {
        const MKL_INT n = static_cast<MKL_INT>(mat.rows());
        const MKL_INT nnz = static_cast<MKL_INT>(mat.nonZeros());

        // PARDISO requires CSR format with 1-based indexing
        ia_.resize(n + 1);
        ja_.resize(nnz);
        a_.resize(nnz);

        // Convert from column-major to row-major CSR format
        // Count entries per row first
        std::vector<MKL_INT> rowCounts(n, 0);
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
            for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                rowCounts[it.row()]++;
            }
        }

        // Build row pointers (1-based)
        ia_[0] = 1;
        for (MKL_INT i = 0; i < n; ++i) {
            ia_[i + 1] = ia_[i] + rowCounts[i];
        }

        // Fill column indices and values
        std::vector<MKL_INT> rowOffsets(n, 0);
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
            for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                MKL_INT row = static_cast<MKL_INT>(it.row());
                MKL_INT pos = ia_[row] - 1 + rowOffsets[row]; // 0-based position
                ja_[pos] = static_cast<MKL_INT>(j) + 1; // 1-based column index
                a_[pos] = it.value();
                rowOffsets[row]++;
            }
        }
    }

#else

    // Stub implementations (simple throws) - declared inline in header

#endif

} // namespace mpfem
