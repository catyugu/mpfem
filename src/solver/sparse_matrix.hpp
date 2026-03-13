#ifndef MPFEM_SPARSE_MATRIX_HPP
#define MPFEM_SPARSE_MATRIX_HPP

#include "core/types.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_set>

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

        SparseMatrix() = default;

        explicit SparseMatrix(Index rows, Index cols)
            : mat_(rows, cols) {}

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

        /// Reserve space for non-zeros
        void reserve(Index nonZeros)
        {
            mat_.reserve(nonZeros);
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
         * @param dofValues Map from DOF index to known value.
         * @param b RHS vector.
         */
        void eliminateRows(const std::map<Index, Real>& dofValues, Vector& b)
        {
            if (dofValues.empty()) return;
            
            const Index n = mat_.rows();
            
            // Build vectors for O(1) lookup
            std::vector<bool> isEliminated(n, false);
            std::vector<Real> eliminatedValues(n, 0.0);
            for (const auto& [dof, val] : dofValues) {
                isEliminated[dof] = true;
                eliminatedValues[dof] = val;
            }
            
            // Process all columns
            for (Index col = 0; col < mat_.outerSize(); ++col) {
                // Iterate over all rows in this column
                for (Storage::InnerIterator it(mat_, col); it; ++it) {
                    Index row = it.row();
                    
                    if (isEliminated[col] && isEliminated[row]) {
                        // Both eliminated: zero off-diagonal entries
                        if (row != col) {
                            it.valueRef() = 0.0;
                        }
                    } else if (isEliminated[col]) {
                        // Column eliminated (but row not): subtract from RHS, zero entry
                        b(row) -= it.value() * eliminatedValues[col];
                        it.valueRef() = 0.0;
                    } else if (isEliminated[row]) {
                        // Row eliminated (but column not): just zero the entry
                        it.valueRef() = 0.0;
                    }
                    // else: neither eliminated, keep the entry
                }
            }
            
            // Set diagonal to 1 and RHS to BC value for all eliminated DOFs
            for (const auto& [dof, val] : dofValues) {
                mat_.coeffRef(dof, dof) = 1.0;
                b(dof) = val;
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

    private:
        Storage mat_;
        std::vector<Triplet> triplets_;
    };

} // namespace mpfem

#endif // MPFEM_SPARSE_MATRIX_HPP
