#ifndef MPFEM_ASSEMBLER_HPP
#define MPFEM_ASSEMBLER_HPP

#include "assembly/integrator.hpp"
#include "core/sparse_matrix.hpp"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>

namespace mpfem {

    // Forward declarations
    class FESpace;

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
        std::vector<SparseMatrix::Triplet> triplets_;
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
        std::vector<Vector> threadVectors_;
    };

} // namespace mpfem

#endif // MPFEM_ASSEMBLER_HPP
