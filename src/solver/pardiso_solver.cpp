#include "solver/pardiso_solver.hpp"
#include "core/logger.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef MPFEM_USE_MKL
#include <Eigen/Sparse>
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#endif

namespace mpfem {

#ifdef MPFEM_USE_MKL

    struct PardisoSolver::Impl {
        _MKL_DSS_HANDLE_t pt[64];
        MKL_INT iparm[64] = {0};
        std::vector<MKL_INT> ia;
        std::vector<MKL_INT> ja;
        std::vector<Real> a;
        MKL_INT n = 0;
        MKL_INT maxfct = 1;
        MKL_INT mnum = 1;
        MKL_INT mtype = 11;
        MKL_INT phase = 0;
        MKL_INT msglvl = 0;
        MKL_INT error = 0;
        bool initialized = false;
        int iterations = 1;
        Real residual = 0.0;

        void convertToCSR(const SparseMatrix::Storage& mat) {
            const MKL_INT n_size = static_cast<MKL_INT>(mat.rows());
            const MKL_INT nnz = static_cast<MKL_INT>(mat.nonZeros());

            ia.resize(n_size + 1);
            ja.resize(nnz);
            a.resize(nnz);

            std::vector<MKL_INT> rowCounts(n_size, 0);
            for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
                for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                    rowCounts[it.row()]++;
                }
            }

            ia[0] = 1;
            for (MKL_INT i = 0; i < n_size; ++i) {
                ia[i + 1] = ia[i] + rowCounts[i];
            }

            std::vector<MKL_INT> rowOffsets(n_size, 0);
            for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
                for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                    MKL_INT row = static_cast<MKL_INT>(it.row());
                    MKL_INT pos = ia[row] - 1 + rowOffsets[row];
                    ja[pos] = static_cast<MKL_INT>(j) + 1;
                    a[pos] = it.value();
                    rowOffsets[row]++;
                }
            }
        }
    };

    PardisoSolver::PardisoSolver() : impl_(std::make_unique<Impl>())
    {
        std::memset(impl_->pt, 0, sizeof(impl_->pt));
        impl_->mtype = 11;
        pardisoinit(impl_->pt, &impl_->mtype, impl_->iparm);
        impl_->iparm[0] = 1;
        impl_->iparm[1] = 2;
    }

    PardisoSolver::~PardisoSolver()
    {
        if (impl_->initialized) {
            impl_->phase = -1;
            MKL_INT nrhs = 1;
            MKL_INT dummy_n = 0;
            pardiso(impl_->pt, &impl_->maxfct, &impl_->mnum, &impl_->mtype, &impl_->phase,
                &dummy_n, nullptr, nullptr, nullptr,
                nullptr, &nrhs, impl_->iparm, &impl_->msglvl, nullptr, nullptr, &impl_->error);
        }
    }

    void PardisoSolver::setup(const SparseMatrix* A)
    {
        if (!A) throw std::runtime_error("PardisoSolver: null matrix in setup");
        const auto& mat = A->eigen();
        impl_->n = static_cast<MKL_INT>(mat.rows());
        impl_->convertToCSR(mat);
        impl_->phase = 12;
        MKL_INT nrhs = 1;
        impl_->initialized = true;

        Vector temp(impl_->n);
        temp.setZero();
        pardiso(impl_->pt, &impl_->maxfct, &impl_->mnum, &impl_->mtype, &impl_->phase,
            &impl_->n, impl_->a.data(), impl_->ia.data(), impl_->ja.data(),
            nullptr, &nrhs, impl_->iparm, &impl_->msglvl,
            temp.data(), temp.data(), &impl_->error);

        if (impl_->error != 0) throw std::runtime_error("PardisoSolver: factorization failed");
        set_matrix(A);
        mark_setup();
    }

    void PardisoSolver::apply(const Vector& b, Vector& x)
    {
        if (!is_setup()) throw std::runtime_error("PardisoSolver: not setup");
        MKL_INT nrhs = 1;
        impl_->phase = 33;
        Vector b_copy = b;
        x.resize(impl_->n);
        pardiso(impl_->pt, &impl_->maxfct, &impl_->mnum, &impl_->mtype, &impl_->phase,
            &impl_->n, impl_->a.data(), impl_->ia.data(), impl_->ja.data(),
            nullptr, &nrhs, impl_->iparm, &impl_->msglvl,
            b_copy.data(), x.data(), &impl_->error);

        if (impl_->error != 0) throw std::runtime_error("PardisoSolver: solve failed");
        impl_->iterations = 1;
        impl_->residual = 0.0;
    }

    int PardisoSolver::iterations() const { return impl_->iterations; }
    Real PardisoSolver::residual() const { return impl_->residual; }

#else

    struct PardisoSolver::Impl {};
    PardisoSolver::PardisoSolver() { throw std::runtime_error("PardisoSolver: MKL not available"); }
    PardisoSolver::~PardisoSolver() = default;
    void PardisoSolver::setup(const SparseMatrix*) { throw std::runtime_error("PardisoSolver: MKL not available"); }
    void PardisoSolver::apply(const Vector&, Vector&) { throw std::runtime_error("PardisoSolver: MKL not available"); }
    int PardisoSolver::iterations() const { return 0; }
    Real PardisoSolver::residual() const { return 0.0; }

#endif

} // namespace mpfem
