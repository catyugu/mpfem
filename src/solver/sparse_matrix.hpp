#ifndef MPFEM_SPARSE_MATRIX_HPP
#define MPFEM_SPARSE_MATRIX_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mpfem
{

    /**
     * @brief Sparse matrix wrapper using Eigen::SparseMatrix.
     *
     * This is a thin wrapper around Eigen's sparse matrix to provide
     * a consistent interface for the solver module. Uses column-major
     * storage for better compatibility with sparse solvers.
     */
    class SparseMatrix
    {
    public:
        using Storage = Eigen::SparseMatrix<Real, Eigen::ColMajor, Index>;
        using Triplet = Eigen::Triplet<Real, Index>;

        static constexpr std::uint64_t DynamicTag = std::numeric_limits<std::uint64_t>::max();

        SparseMatrix() = default;

        explicit SparseMatrix(Index rows, Index cols)
            : mat_(rows, cols) {}

        explicit SparseMatrix(const Storage& mat)
            : mat_(mat) {}

        explicit SparseMatrix(Storage&& mat)
            : mat_(std::move(mat)) {}

        SparseMatrix(const SparseMatrix&) = default;
        SparseMatrix(SparseMatrix&&) noexcept = default;
        SparseMatrix& operator=(const SparseMatrix&) = default;
        SparseMatrix& operator=(SparseMatrix&&) noexcept = default;

        SparseMatrix& operator=(const Storage& mat)
        {
            mat_ = mat;
            return *this;
        }

        SparseMatrix& operator=(Storage&& mat)
        {
            mat_ = std::move(mat);
            return *this;
        }

        /// Get number of rows
        Index rows() const { return mat_.rows(); }

        /// Get number of columns
        Index cols() const { return mat_.cols(); }

        /// Get number of non-zeros
        Index nonZeros() const { return mat_.nonZeros(); }

        /// Resize matrix
        void resize(Index rows, Index cols)
        {
            mat_.resize(rows, cols);
        }

        /// Reserve space for non-zeros per column
        void reserve(Index nonZerosPerCol)
        {
            if (mat_.rows() == 0 || mat_.cols() == 0) return;
            Eigen::VectorXi reserveSize(mat_.cols());
            reserveSize.setConstant(nonZerosPerCol);
            mat_.reserve(reserveSize);
        }

        /// Set from triplets (efficient batch insertion)
        void setFromTriplets(const std::vector<Triplet> &triplets)
        {
            mat_.setFromTriplets(triplets.begin(), triplets.end());
        }

        /// Set from triplets (move version)
        void setFromTriplets(std::vector<Triplet> &&triplets)
        {
            mat_.setFromTriplets(triplets.begin(), triplets.end());
        }

        /// Add a triplet for later assembly
        void addTriplet(Index row, Index col, Real value)
        {
            triplets_.emplace_back(row, col, value);
        }

        /// Assemble from accumulated triplets
        void assemble()
        {
            mat_.setFromTriplets(triplets_.begin(), triplets_.end());
            triplets_.clear();
            triplets_.shrink_to_fit();
        }
        
        /// Get reference to internal triplets (for efficient merging)
        std::vector<Triplet>& triplets() { return triplets_; }
        const std::vector<Triplet>& triplets() const { return triplets_; }

        /// Clear all data
        void clear()
        {
            mat_.setZero();
            triplets_.clear();
        }

        /// Set all entries to zero (keep structure)
        void setZero()
        {
            mat_.setZero();
        }

        /// Coefficient access (slow, for debugging)
        Real coeff(Index row, Index col) const
        {
            return mat_.coeff(row, col);
        }

        /// Mutable coefficient access (slow, creates entry if not exists)
        Real &coeffRef(Index row, Index col)
        {
            return mat_.coeffRef(row, col);
        }

        /// Get underlying Eigen matrix (const)
        const Storage &eigen() const { return mat_; }

        /// Get underlying Eigen matrix (mutable)
        Storage &eigen() { return mat_; }

        /// Make compressed (required for some solvers)
        void makeCompressed()
        {
            mat_.makeCompressed();
        }

        /// Check if compressed
        bool isCompressed() const
        {
            return mat_.isCompressed();
        }

        /**
         * @brief Eliminate multiple rows for Dirichlet BCs (efficient version).
         * 
         * Optimized batch elimination using sparse column iteration.
         * 
         * @param eliminated Vector of eliminated DOF indices (must be sorted).
         * @param dofValues Vector of values for eliminated DOFs (aligned with eliminated).
         * @param b RHS vector.
         */
        void eliminateRows(const std::vector<Index>& eliminated, 
                          const std::vector<Real>& dofValues, 
                          Vector& b)
        {
            if (eliminated.empty()) return;
            
            const Index n = mat_.rows();
            const size_t numEliminated = eliminated.size();
            
            std::vector<char> isEliminated(n, 0);
            std::vector<Real> eliminatedValues(n, 0.0);
            for (size_t i = 0; i < numEliminated; ++i) {
                isEliminated[eliminated[i]] = 1;
                eliminatedValues[eliminated[i]] = dofValues[eliminated[i]];
            }
            
            for (Index col = 0; col < mat_.outerSize(); ++col) {
                for (Storage::InnerIterator it(mat_, col); it; ++it) {
                    Index row = it.row();
                    
                    if (isEliminated[col] && isEliminated[row]) {
                        if (row != col) {
                            it.valueRef() = 0.0;
                        }
                    } else if (isEliminated[col]) {
                        b(row) -= it.value() * eliminatedValues[col];
                        it.valueRef() = 0.0;
                    } else if (isEliminated[row]) {
                        it.valueRef() = 0.0;
                    }
                }
            }
            
            for (size_t i = 0; i < numEliminated; ++i) {
                Index dof = eliminated[i];
                mat_.coeffRef(dof, dof) = 1.0;
                b(dof) = dofValues[dof];
            }
        }

        /// Write to Matrix Market format
        void writeToMatrixMarket(const std::string &filename) const
        {
            std::ofstream file(filename);
            file << "%%MatrixMarket matrix coordinate real general\n";
            file << rows() << " " << cols() << " " << nonZeros() << "\n";
            for (int k = 0; k < mat_.outerSize(); ++k)
            {
                for (Storage::InnerIterator it(mat_, k); it; ++it)
                {
                    file << it.row() + 1 << " " << it.col() + 1 << " "
                         << it.value() << "\n";
                }
            }
        }

        /// SparseMatrix += SparseMatrix
        SparseMatrix& operator+=(const SparseMatrix& B) {
            MPFEM_ASSERT(rows() == B.rows() && cols() == B.cols(),
                "SparseMatrix size mismatch in +=");
            mat_ += B.mat_;
            return *this;
        }

        /// SparseMatrix += alpha * SparseMatrix
        SparseMatrix& addScaled(const SparseMatrix& B, Real alpha) {
            MPFEM_ASSERT(rows() == B.rows() && cols() == B.cols(),
                "SparseMatrix size mismatch in addScaled");
            if (alpha != 0.0) {
                mat_ += alpha * B.mat_;
            }
            return *this;
        }

        /// SparseMatrix -= SparseMatrix
        SparseMatrix& operator-=(const SparseMatrix& B) {
            MPFEM_ASSERT(rows() == B.rows() && cols() == B.cols(),
                "SparseMatrix size mismatch in -=");
            mat_ -= B.mat_;
            return *this;
        }

        /// SparseMatrix *= Scalar
        SparseMatrix& operator*=(Real alpha) {
            mat_ *= alpha;
            return *this;
        }

        /// SparseMatrix + SparseMatrix
        SparseMatrix operator+(const SparseMatrix& B) const {
            MPFEM_ASSERT(rows() == B.rows() && cols() == B.cols(),
                "SparseMatrix size mismatch in +");
            SparseMatrix C(rows(), cols());
            C.mat_ = mat_ + B.mat_;
            return C;
        }

        /// SparseMatrix - SparseMatrix
        SparseMatrix operator-(const SparseMatrix& B) const {
            MPFEM_ASSERT(rows() == B.rows() && cols() == B.cols(),
                "SparseMatrix size mismatch in -");
            SparseMatrix C(rows(), cols());
            C.mat_ = mat_ - B.mat_;
            return C;
        }

        /// SparseMatrix * Scalar
        SparseMatrix operator*(Real alpha) const {
            SparseMatrix C(rows(), cols());
            C.mat_ = mat_ * alpha;
            return C;
        }

        /// Scalar * SparseMatrix (left scalar multiplication is provided
        /// as a free function below to avoid use of `friend`.)

        /// SparseMatrix * SparseMatrix
        SparseMatrix operator*(const SparseMatrix& B) const {
            MPFEM_ASSERT(cols() == B.rows(),
                "SparseMatrix size mismatch in matrix multiplication");
            SparseMatrix C(rows(), B.cols());
            C.mat_ = mat_ * B.mat_;
            return C;
        }

        /// SparseMatrix * Vector
        Vector operator*(const Vector& v) const {
            MPFEM_ASSERT(cols() == v.size(),
                "SparseMatrix size mismatch in matrix-vector multiplication");
            return mat_ * v;
        }

        /// Fast fingerprint for matrix-identity based caches.
        /// Requires compressed storage for deterministic index arrays.
        std::uint64_t fingerprint() const {
            MPFEM_ASSERT(mat_.isCompressed(), "SparseMatrix::fingerprint requires compressed matrix");

            constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
            constexpr std::uint64_t kFnvPrime = 1099511628211ull;

            auto mixBytes = [](std::uint64_t seed, const void* data, std::size_t len) {
                const auto* bytes = static_cast<const unsigned char*>(data);
                for (std::size_t i = 0; i < len; ++i) {
                    seed ^= static_cast<std::uint64_t>(bytes[i]);
                    seed *= kFnvPrime;
                }
                return seed;
            };

            std::uint64_t h = kFnvOffset;
            const Index r = rows();
            const Index c = cols();
            const Index nz = nonZeros();

            h = mixBytes(h, &r, sizeof(r));
            h = mixBytes(h, &c, sizeof(c));
            h = mixBytes(h, &nz, sizeof(nz));

            const auto* outer = mat_.outerIndexPtr();
            const auto* inner = mat_.innerIndexPtr();
            const auto* vals = mat_.valuePtr();

            h = mixBytes(h, outer, static_cast<std::size_t>(mat_.outerSize() + 1) * sizeof(Index));
            h = mixBytes(h, inner, static_cast<std::size_t>(nz) * sizeof(Index));
            h = mixBytes(h, vals, static_cast<std::size_t>(nz) * sizeof(Real));

            return h;
        }

    private:
        Storage mat_;
        std::vector<Triplet> triplets_;
    };

// Left scalar multiplication for SparseMatrix without using `friend`.
inline SparseMatrix operator*(Real alpha, const SparseMatrix& A) {
    return A * alpha;
}

} // namespace mpfem

#endif // MPFEM_SPARSE_MATRIX_HPP
