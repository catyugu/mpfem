#ifndef MPFEM_ASSEMBLER_HPP
#define MPFEM_ASSEMBLER_HPP

#include "assembly/integrator.hpp"
#include "fe/fe_space.hpp"
#include "operator/sparse_matrix.hpp"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mpfem {

    // =============================================================================
    // 线程本地缓冲区（零分配）
    // =============================================================================

    struct alignas(64) ThreadBuffer {
        Eigen::Matrix<Real, MaxVectorDofsPerElement, MaxVectorDofsPerElement, Eigen::RowMajor> elmatVector;
        Eigen::Matrix<Real, MaxVectorDofsPerElement, 1> elvecVector;
        std::array<Index, MaxVectorDofsPerElement> dofs; // Fixed-size DOF buffer, no heap allocation
        Index numDofs = 0; // Track actual number of DOFs
        // Reusable dynamic buffers for integrator APIs that take Matrix/Vector.
        Matrix dynMatrix;
        Vector dynVector;
        // Pre-sized valid DOF index buffer for branch-free triplet writing
        std::array<int, MaxVectorDofsPerElement> validDofs;
        int numValidDofs = 0;

        ThreadBuffer() = default;

        // Pre-size dynMatrix to maximum possible size once
        void ensureDynMatrixSize(int maxTotalDofs)
        {
            if (dynMatrix.rows() < maxTotalDofs || dynMatrix.cols() < maxTotalDofs) {
                dynMatrix.resize(maxTotalDofs, maxTotalDofs);
            }
        }
    };

    // =============================================================================
    // 双线性型组装器
    // =============================================================================

    class BilinearFormAssembler {
    public:
        explicit BilinearFormAssembler(const FESpace* fes);

        // Empty domain set means all domains.
        void addDomainIntegrator(std::unique_ptr<DomainBilinearIntegratorBase> integ,
            const std::set<int>& domains = {})
        {
            if (integ) {
                maxIvdim_ = std::max(maxIvdim_, integ->vdim());
            }
            domainIntegs_.push_back(std::move(integ));
            domainSets_.emplace_back(domains.begin(), domains.end());
        }
        void addBoundaryIntegrator(std::unique_ptr<FaceBilinearIntegratorBase> integ, int bid = -1)
        {
            bdrIntegs_.push_back(std::move(integ));
            bdrIds_.push_back(bid);
        }

        void clearIntegrators()
        {
            domainIntegs_.clear();
            domainSets_.clear();
            bdrIntegs_.clear();
            bdrIds_.clear();
        }
        void clear() { mat_.setZero(); }

        void assemble();
        void finalize() { mat_.makeCompressed(); }

        SparseMatrix& matrix() { return mat_; }
        Index rows() const { return mat_.rows(); }

    private:
        const FESpace* fes_;
        std::vector<std::unique_ptr<DomainBilinearIntegratorBase>> domainIntegs_;
        std::vector<std::vector<int>> domainSets_;
        std::vector<std::unique_ptr<FaceBilinearIntegratorBase>> bdrIntegs_;
        std::vector<int> bdrIds_;
        SparseMatrix mat_;
        std::vector<ThreadBuffer> buffers_;
        std::vector<SparseMatrix::Triplet> triplets_;
        int maxIvdim_ = 1; // Track maximum ivdim across all integrators for pre-sizing
    };

    // =============================================================================
    // 线性型组装器
    // =============================================================================

    class LinearFormAssembler {
    public:
        explicit LinearFormAssembler(const FESpace* fes);

        // Empty domain set means all domains.
        void addDomainIntegrator(std::unique_ptr<DomainLinearIntegratorBase> integ,
            const std::set<int>& domains = {})
        {
            domainIntegs_.push_back(std::move(integ));
            domainSets_.emplace_back(domains.begin(), domains.end());
        }
        void addBoundaryIntegrator(std::unique_ptr<FaceLinearIntegratorBase> integ, int bid = -1)
        {
            bdrIntegs_.push_back(std::move(integ));
            bdrIds_.push_back(bid);
        }

        void clearIntegrators()
        {
            domainIntegs_.clear();
            domainSets_.clear();
            bdrIntegs_.clear();
            bdrIds_.clear();
        }
        void clear() { vec_.setZero(); }

        void assemble();

        Vector& vector() { return vec_; }

    private:
        const FESpace* fes_;
        std::vector<std::unique_ptr<DomainLinearIntegratorBase>> domainIntegs_;
        std::vector<std::vector<int>> domainSets_;
        std::vector<std::unique_ptr<FaceLinearIntegratorBase>> bdrIntegs_;
        std::vector<int> bdrIds_;
        Vector vec_;
        std::vector<ThreadBuffer> buffers_;
        std::vector<Vector> threadVectors_;
    };

} // namespace mpfem

#endif // MPFEM_ASSEMBLER_HPP
