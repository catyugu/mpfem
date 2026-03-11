/**
 * @file bilinear_form.hpp
 * @brief Bilinear form assembly using FieldSpace
 */

#ifndef MPFEM_ASSEMBLY_BILINEAR_FORM_HPP
#define MPFEM_ASSEMBLY_BILINEAR_FORM_HPP

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "dof/field_space.hpp"
#include <functional>
#include <unordered_map>

namespace mpfem {

class FEValues;

using LocalMatrixAssembler = std::function<void(
    Index cell_id,
    const FEValues& fe_values,
    DynamicMatrix& local_matrix)>;

/**
 * @brief Bilinear form assembler with workspace optimization
 * 
 * PERFORMANCE:
 * - Pre-allocates local matrix and DoF vectors to reduce memory allocation
 * - Reuses triplet storage across assemble calls
 */
class BilinearForm {
public:
    explicit BilinearForm(const FieldSpace* field);
    
    void assemble(LocalMatrixAssembler local_assembler,
                  SparseMatrix& matrix,
                  bool symmetrize = false);
    
    void assemble_with_coefficients(
        LocalMatrixAssembler local_assembler,
        SparseMatrix& matrix,
        const std::unordered_map<Index, Scalar>& coefficients,
        bool symmetrize = false);
    
    size_t n_entries() const { return n_entries_; }

private:
    const FieldSpace* field_;
    const Mesh* mesh_;
    UpdateFlags update_flags_ = UpdateFlags::UpdateDefault;
    size_t n_entries_;
    
    // PERFORMANCE: Pre-allocated workspace to avoid repeated allocation
    DynamicMatrix local_matrix_;
    std::vector<Index> cell_dofs_;
    std::vector<Eigen::Triplet<Scalar>> triplets_;
};

namespace BilinearForms {
    LocalMatrixAssembler laplacian(Scalar conductivity = 1.0);
    LocalMatrixAssembler laplacian_anisotropic(const Tensor<2, 3>& K_tensor);
    LocalMatrixAssembler mass(Scalar coefficient = 1.0);
    LocalMatrixAssembler elasticity(Scalar E, Scalar nu, int dim);
    LocalMatrixAssembler convection_bc(Scalar h);
}

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_BILINEAR_FORM_HPP
