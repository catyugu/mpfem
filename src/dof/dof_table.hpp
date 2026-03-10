/**
 * @file dof_table.hpp
 * @brief DoF index table using CSR format
 */

#ifndef MPFEM_DOF_DOF_TABLE_HPP
#define MPFEM_DOF_DOF_TABLE_HPP

#include "core/types.hpp"
#include <vector>

namespace mpfem {

/**
 * @brief CSR-format table for cell-to-DoF mapping
 */
class DoFTable {
public:
    DoFTable() = default;

    DoFTable(SizeType n_cells, int dofs_per_cell)
        : row_ptr_(n_cells + 1, 0)
    {
        for (SizeType i = 0; i <= n_cells; ++i) {
            row_ptr_[i] = static_cast<Index>(i) * dofs_per_cell;
        }
        col_ind_.resize(static_cast<size_t>(n_cells) * dofs_per_cell);
    }

    explicit DoFTable(const std::vector<int>& dofs_per_cell) {
        SizeType n_cells = static_cast<SizeType>(dofs_per_cell.size());
        row_ptr_.resize(n_cells + 1);
        row_ptr_[0] = 0;

        Index total = 0;
        for (SizeType i = 0; i < n_cells; ++i) {
            total += dofs_per_cell[i];
            row_ptr_[i + 1] = total;
        }
        col_ind_.resize(total);
    }

    SizeType n_cells() const { return row_ptr_.size() > 1 ? row_ptr_.size() - 1 : 0; }
    Index total_entries() const { return static_cast<Index>(col_ind_.size()); }

    int dofs_per_cell(SizeType cell_idx) const {
        return cell_idx < n_cells()
               ? static_cast<int>(row_ptr_[cell_idx + 1] - row_ptr_[cell_idx])
               : 0;
    }

    Index& operator()(SizeType cell_idx, int local_dof) {
        return col_ind_[row_ptr_[cell_idx] + local_dof];
    }

    Index operator()(SizeType cell_idx, int local_dof) const {
        return col_ind_[row_ptr_[cell_idx] + local_dof];
    }

    const Index* cell_dofs(SizeType cell_idx) const {
        return col_ind_.data() + row_ptr_[cell_idx];
    }

    Index* cell_dofs(SizeType cell_idx) {
        return col_ind_.data() + row_ptr_[cell_idx];
    }

    void get_cell_dofs(SizeType cell_idx, std::vector<Index>& dofs) const {
        Index start = row_ptr_[cell_idx];
        Index end = row_ptr_[cell_idx + 1];
        dofs.assign(col_ind_.begin() + start, col_ind_.begin() + end);
    }

    void set_cell_dofs(SizeType cell_idx, const std::vector<Index>& dofs) {
        Index start = row_ptr_[cell_idx];
        std::copy(dofs.begin(), dofs.end(), col_ind_.begin() + start);
    }

    void clear() {
        row_ptr_.clear();
        col_ind_.clear();
    }

private:
    std::vector<Index> row_ptr_;
    std::vector<Index> col_ind_;
};

} // namespace mpfem

#endif // MPFEM_DOF_DOF_TABLE_HPP
