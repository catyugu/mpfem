#ifndef MPFEM_SPARSE_MATRIX_HPP
#define MPFEM_SPARSE_MATRIX_HPP

#include "core/types.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>

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
            // Proper Dirichlet BC elimination:
            // For row elimination, we need to:
            // 1. For each column j != row in this row, zero the entry
            // 2. For each row k != row with entry in column 'row', 
            //    subtract A(k,row) * value from b(k) and zero the entry
            // 3. Set A(row,row) = 1 and b(row) = value
            
            // Step 1 & 2: Process column 'row' using column iterator (efficient for sparse matrix)
            // The column 'row' contains entries A(k,row) for various k
            for (Storage::InnerIterator it(mat_, row); it; ++it) {
                Index k = it.row();
                if (k == row) continue;  // Skip diagonal
                
                Real a_ki = it.value();
                // Subtract contribution from RHS
                b(k) -= a_ki * value;
                // Zero the column entry (this modifies the matrix)
                it.valueRef() = 0.0;
            }
            
            // Step 3: Zero row entries using row iterator (if row-major) or scan
            // For column-major storage, we need to scan columns
            for (Index col = 0; col < mat_.cols(); ++col) {
                if (col == row) continue;
                // Use coeffRef to access and modify (creates entry if needed, but typically exists)
                mat_.coeffRef(row, col) = 0.0;
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
            std::map<Index, Real> dofValues;
            for (size_t i = 0; i < rows.size(); ++i) {
                dofValues[rows[i]] = values(i);
            }
            eliminateRows(dofValues, b);
        }

        /**
         * @brief Eliminate multiple rows for Dirichlet BCs (efficient version).
         * 
         * Optimized batch elimination using a map for O(1) value lookup.
         * 
         * @param dofValues Map from DOF index to known value.
         * @param b RHS vector.
         */
        void eliminateRows(const std::map<Index, Real>& dofValues, Vector& b)
        {
            if (dofValues.empty()) return;
            
            // Create a set of eliminated DOFs for fast lookup
            std::vector<bool> isEliminated(mat_.rows(), false);
            for (const auto& [row, val] : dofValues) {
                isEliminated[row] = true;
            }
            
            // Process all columns that have eliminated DOFs
            // For each eliminated DOF, process its column (efficient using column iterator)
            for (const auto& [row, val] : dofValues) {
                for (Storage::InnerIterator it(mat_, row); it; ++it) {
                    Index k = it.row();
                    if (isEliminated[k]) continue;  // Skip if this DOF is also eliminated
                    
                    Real a_ki = it.value();
                    // Subtract contribution from RHS
                    b(k) -= a_ki * val;
                    // Zero the column entry
                    it.valueRef() = 0.0;
                }
            }
            
            // Zero all row entries for eliminated DOFs and set diagonal
            for (const auto& [row, val] : dofValues) {
                // Zero row entries (except diagonal)
                for (Index col = 0; col < mat_.cols(); ++col) {
                    if (col != row) {
                        mat_.coeffRef(row, col) = 0.0;
                    }
                }
                // Set diagonal to 1 and RHS to value
                mat_.coeffRef(row, row) = 1.0;
                b(row) = val;
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