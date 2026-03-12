#ifndef MPFEM_SPARSE_MATRIX_HPP
#define MPFEM_SPARSE_MATRIX_HPP

#include "core/types.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <iostream>

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

        /// Get triplets (for external assembly)
        std::vector<Triplet> &triplets() { return triplets_; }
        const std::vector<Triplet> &triplets() const { return triplets_; }

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
         * @brief Eliminate a row: zero all entries and set diagonal to 1.
         * 
         * Also modifies the RHS vector to account for known values.
         * This is the standard "static condensation" approach for Dirichlet BCs.
         * 
         * @param row Row index to eliminate.
         * @param value Known value for this DOF.
         * @param b RHS vector (modified to subtract eliminated column contributions).
         */
        void eliminateRow(Index row, Real value, Vector& b)
        {
            // For each column j in this row:
            // - If j != row: add A(row,j) * value to b(j), then zero A(row,j)
            // - If j == row: set A(row,j) = 1, set b(row) = value
            
            // Note: Eigen sparse matrix iteration is column-major
            // We need to iterate over all columns to find entries in this row
            
            for (Index j = 0; j < mat_.cols(); ++j) {
                Real a_ij = mat_.coeff(row, j);
                if (std::abs(a_ij) > 0) {
                    if (j != row) {
                        // Add contribution to RHS for column j
                        b(j) -= a_ij * value;
                        // Zero the entry
                        mat_.coeffRef(row, j) = 0.0;
                    }
                }
            }
            
            // Set diagonal to 1 and RHS to value
            mat_.coeffRef(row, row) = 1.0;
            b(row) = value;
        }

        /**
         * @brief Eliminate multiple rows for Dirichlet BCs.
         * 
         * @param rows Row indices to eliminate.
         * @param values Corresponding known values.
         * @param b RHS vector.
         */
        void eliminateRows(const std::vector<Index>& rows, const Vector& values, Vector& b)
        {
            for (size_t i = 0; i < rows.size(); ++i) {
                eliminateRow(rows[i], values(i), b);
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