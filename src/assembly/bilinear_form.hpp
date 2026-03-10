/**
 * @file bilinear_form.hpp
 * @brief BilinearForm - interface for assembling stiffness matrices
 * 
 * Provides a flexible interface for assembling bilinear forms (weak forms)
 * into global sparse matrices. Users provide the local element matrix
 * computation via a callback function.
 */

#ifndef MPFEM_ASSEMBLY_BILINEAR_FORM_HPP
#define MPFEM_ASSEMBLY_BILINEAR_FORM_HPP

#include "core/types.hpp"
#include "fe_values.hpp"
#include "dof/dof_handler.hpp"
#include "mesh/mesh.hpp"
#include <Eigen/Sparse>
#include <functional>
#include <vector>

namespace mpfem {

/**
 * @brief Local matrix assembly callback type
 * 
 * Called for each cell to compute the local element matrix.
 * 
 * @param cell_id Current cell index
 * @param fe_values FEValues object initialized for this cell
 * @param local_matrix Output: local matrix [dofs_per_cell x dofs_per_cell]
 */
using LocalMatrixAssembler = std::function<void(
    Index cell_id,
    const FEValues& fe_values,
    DynamicMatrix& local_matrix)>;

/**
 * @brief Bilinear form assembler for stiffness matrices
 * 
 * Assembles the global stiffness matrix from local element contributions.
 * 
 * Usage:
 * @code
 * BilinearForm bilinear_form(dof_handler);
 * 
 * // Define local matrix computation (e.g., Laplacian: K_ij = int(grad(N_i) . grad(N_j)))
 * auto laplacian_assembler = [](Index cell, const FEValues& fe, DynamicMatrix& K_local) {
 *     int n = fe.dofs_per_cell();
 *     K_local.setZero(n, n);
 *     for (int q = 0; q < fe.n_quadrature_points(); ++q) {
 *         double jxw = fe.JxW(q);
 *         for (int i = 0; i < n; ++i) {
 *             for (int j = 0; j < n; ++j) {
 *                 K_local(i, j) += fe.shape_grad(i, q).dot(fe.shape_grad(j, q)) * jxw;
 *             }
 *         }
 *     }
 * };
 * 
 * SparseMatrix K;
 * bilinear_form.assemble(laplacian_assembler, K);
 * @endcode
 */
class BilinearForm {
public:
    /**
     * @brief Construct with DoFHandler
     * @param dof_handler DoF handler (must be initialized)
     */
    explicit BilinearForm(const DoFHandler* dof_handler);
    
    /**
     * @brief Assemble global matrix
     * @param local_assembler Callback for computing local element matrices
     * @param matrix Output: global sparse matrix
     * @param symmetrize If true, enforce symmetry (K = 0.5 * (K + K^T))
     */
    void assemble(LocalMatrixAssembler local_assembler,
                  SparseMatrix& matrix,
                  bool symmetrize = true);
    
    /**
     * @brief Assemble with coefficient (e.g., conductivity, permittivity)
     * @param local_assembler Callback for computing local element matrices
     * @param matrix Output: global sparse matrix
     * @param coefficients Per-domain coefficients [domain_id -> value]
     * @param symmetrize If true, enforce symmetry
     */
    void assemble_with_coefficients(
        LocalMatrixAssembler local_assembler,
        SparseMatrix& matrix,
        const std::unordered_map<Index, Scalar>& coefficients,
        bool symmetrize = true);
    
    /**
     * @brief Set update flags for FEValues
     */
    void set_update_flags(UpdateFlags flags) { update_flags_ = flags; }
    
    /**
     * @brief Get number of assembled entries
     */
    size_t n_entries() const { return n_entries_; }
    
private:
    const DoFHandler* dof_handler_;
    const Mesh* mesh_;
    const FESpace* fe_space_;
    
    UpdateFlags update_flags_;
    size_t n_entries_;
};

/**
 * @brief Common bilinear form assemblers
 * 
 * Predefined local matrix assemblers for common PDEs.
 */
namespace BilinearForms {

/**
 * @brief Laplacian (diffusion) operator
 * 
 * K_ij = int_Omega (grad(N_i) . grad(N_j)) dx
 * 
 * For anisotropic materials: K_ij = int_Omega (grad(N_i) . K . grad(N_j)) dx
 * where K is the conductivity/diffusivity tensor.
 * 
 * @param conductivity Isotropic conductivity (default 1.0)
 */
LocalMatrixAssembler laplacian(Scalar conductivity = 1.0);

/**
 * @brief Anisotropic Laplacian operator
 * 
 * K_ij = int_Omega (grad(N_i) . K_tensor . grad(N_j)) dx
 * 
 * @param conductivity_tensor 3x3 conductivity tensor
 */
LocalMatrixAssembler laplacian_anisotropic(const Tensor<2, 3>& conductivity_tensor);

/**
 * @brief Mass matrix
 * 
 * M_ij = int_Omega (N_i * N_j) dx
 */
LocalMatrixAssembler mass(Scalar coefficient = 1.0);

/**
 * @brief Linear elasticity (isotropic)
 * 
 * For displacement field u = [u_x, u_y, u_z]:
 * 
 * K_ij = int_Omega (B_i^T : D : B_j) dx
 * 
 * where B_i is the strain-displacement matrix and D is the elasticity tensor.
 * 
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @param dim Spatial dimension (2 or 3)
 */
LocalMatrixAssembler elasticity(Scalar E, Scalar nu, int dim = 3);

/**
 * @brief Convection boundary condition (Robin BC)
 * 
 * For surface integral: int_Gamma (h * N_i * N_j) dS
 * 
 * @param h Convection coefficient
 */
LocalMatrixAssembler convection_bc(Scalar h);

}  // namespace BilinearForms

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_BILINEAR_FORM_HPP
