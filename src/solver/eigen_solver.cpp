#include "solver/eigen_solver.hpp"
#include "eigen_preconditioner_adapter.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

    // Helper template to reduce boilerplate in the .cpp file
    template <typename EigenSolverType>
    struct IterativeImpl {
        int maxIterations = 1000;
        Real tolerance = 1e-10;
        int iterations = 0;
        Real residual = 0.0;
        EigenSolverType solver;
    };

    // =============================================================================
    // CgOperator
    // =============================================================================

    using EigenCgSolver = Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>, Eigen::Lower | Eigen::Upper, EigenPreconditionerAdapter>;

    struct CgOperator::Impl : IterativeImpl<EigenCgSolver> { };

    CgOperator::CgOperator() : impl_(std::make_unique<Impl>()) { }
    CgOperator::~CgOperator() = default;

    void CgOperator::setup(const SparseMatrix* A)
    {
        if (!A) throw std::runtime_error("CG: null matrix in setup");
        impl_->solver.setMaxIterations(impl_->maxIterations);
        impl_->solver.setTolerance(impl_->tolerance);
        impl_->solver.compute(A->eigen());

        if (preconditioner()) {
            preconditioner()->setup(A);
            impl_->solver.preconditioner().set_operator(preconditioner());
        } else {
            impl_->solver.preconditioner().set_operator(nullptr);
        }
        set_matrix(A);
        mark_setup();
    }

    void CgOperator::apply(const Vector& b, Vector& x)
    {
        x = impl_->solver.solveWithGuess(b, x);
        impl_->iterations = static_cast<int>(impl_->solver.iterations());
        impl_->residual = impl_->solver.error();
    }

    void CgOperator::configure(const LinearOperatorConfig& config)
    {
        if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
            impl_->maxIterations = static_cast<int>(it->second);
        }
        if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
            impl_->tolerance = it->second;
        }
    }

    int CgOperator::iterations() const { return impl_->iterations; }
    Real CgOperator::residual() const { return impl_->residual; }

    // =============================================================================
    // GmresOperator
    // =============================================================================

    using EigenGmresSolver = Eigen::DGMRES<Eigen::SparseMatrix<Real>, EigenPreconditionerAdapter>;

    struct GmresOperator::Impl : IterativeImpl<EigenGmresSolver> { };

    GmresOperator::GmresOperator() : impl_(std::make_unique<Impl>()) { }
    GmresOperator::~GmresOperator() = default;

    void GmresOperator::setup(const SparseMatrix* A)
    {
        if (!A) throw std::runtime_error("DGMRES: null matrix in setup");
        impl_->solver.setMaxIterations(impl_->maxIterations);
        impl_->solver.setTolerance(impl_->tolerance);
        impl_->solver.compute(A->eigen());

        if (preconditioner()) {
            preconditioner()->setup(A);
            impl_->solver.preconditioner().set_operator(preconditioner());
        } else {
            impl_->solver.preconditioner().set_operator(nullptr);
        }
        set_matrix(A);
        mark_setup();
    }

    void GmresOperator::apply(const Vector& b, Vector& x)
    {
        x = impl_->solver.solveWithGuess(b, x);
        impl_->iterations = static_cast<int>(impl_->solver.iterations());
        impl_->residual = impl_->solver.error();
    }

    void GmresOperator::configure(const LinearOperatorConfig& config)
    {
        if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
            impl_->maxIterations = static_cast<int>(it->second);
        }
        if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
            impl_->tolerance = it->second;
        }
    }

    int GmresOperator::iterations() const { return impl_->iterations; }
    Real GmresOperator::residual() const { return impl_->residual; }

    // =============================================================================
    // EigenSparseLUOperator
    // =============================================================================

    struct EigenSparseLUOperator::Impl {
        Eigen::SparseLU<Eigen::SparseMatrix<Real>> solver;
    };

    EigenSparseLUOperator::EigenSparseLUOperator() : impl_(std::make_unique<Impl>()) { }
    EigenSparseLUOperator::~EigenSparseLUOperator() = default;

    void EigenSparseLUOperator::setup(const SparseMatrix* A)
    {
        if (!A)
            throw std::runtime_error("EigenSparseLUOperator: null matrix in setup");
        impl_->solver.compute(A->eigen());
        if (impl_->solver.info() != Eigen::Success) {
            throw std::runtime_error("EigenSparseLUOperator: factorization failed");
        }
        set_matrix(A);
        mark_setup();
    }

    void EigenSparseLUOperator::apply(const Vector& b, Vector& x)
    {
        x = impl_->solver.solve(b);
        if (impl_->solver.info() != Eigen::Success) {
            throw std::runtime_error("EigenSparseLUOperator: solve failed");
        }
    }

} // namespace mpfem
