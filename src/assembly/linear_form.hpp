/**
 * @file linear_form.hpp
 * @brief LinearForm - interface for assembling right-hand side vectors
 * 
 * Provides a flexible interface for assembling linear forms into global vectors.
 * Supports volume source terms and boundary flux terms.
 */

#ifndef MPFEM_ASSEMBLY_LINEAR_FORM_HPP
#define MPFEM_ASSEMBLY_LINEAR_FORM_HPP

#include "core/types.hpp"
#include "fe_values.hpp"
#include "dof/dof_handler.hpp"
#include "mesh/mesh.hpp"
#include <functional>
#include <vector>

namespace mpfem {

/**
 * @brief Local vector assembly callback type
 * 
 * Called for each cell to compute the local load vector.
 * 
 * @param cell_id Current cell index
 * @param fe_values FEValues object initialized for this cell
 * @param local_vector Output: local vector [dofs_per_cell]
 */
using LocalVectorAssembler = std::function<void(
    Index cell_id,
    const FEValues& fe_values,
    DynamicVector& local_vector)>;

/**
 * @brief Linear form assembler for right-hand side vectors
 * 
 * Assembles the global load vector from local element contributions.
 * 
 * Usage:
 * @code
 * LinearForm linear_form(dof_handler);
 * 
 * // Define local vector computation (e.g., source term: f_i = int(f * N_i))
 * auto source_assembler = [](Index, const FEValues& fe, DynamicVector& F_local) {
 *     int n = fe.dofs_per_cell();
 *     F_local.setZero(n);
 *     for (int q = 0; q < fe.n_quadrature_points(); ++q) {
 *         double jxw = fe.JxW(q);
 *         double f = 1.0;  // source term value
 *         for (int i = 0; i < n; ++i) {
 *             F_local[i] += f * fe.shape_value(i, q) * jxw;
 *         }
 *     }
 * };
 * 
 * DynamicVector F;
 * linear_form.assemble(source_assembler, F);
 * @endcode
 */
class LinearForm {
public:
    /**
     * @brief Construct with DoFHandler
     * @param dof_handler DoF handler (must be initialized)
     */
    explicit LinearForm(const DoFHandler* dof_handler);
    
    /**
     * @brief Assemble global vector from volume integrals
     * @param local_assembler Callback for computing local element vectors
     * @param vector Output: global vector
     */
    void assemble(LocalVectorAssembler local_assembler,
                  DynamicVector& vector);
    
    /**
     * @brief Assemble with domain-dependent source
     * @param local_assembler Callback for computing local element vectors
     * @param vector Output: global vector
     * @param sources Per-domain source values [domain_id -> value]
     */
    void assemble_with_source(
        LocalVectorAssembler local_assembler,
        DynamicVector& vector,
        const std::unordered_map<Index, Scalar>& sources);
    
    /**
     * @brief Assemble boundary contribution
     * @param boundary_id Boundary entity ID
     * @param local_assembler Callback for computing local face vectors
     * @param vector Output: global vector (accumulated)
     */
    void assemble_boundary(Index boundary_id,
                          LocalVectorAssembler local_assembler,
                          DynamicVector& vector);
    
    /**
     * @brief Assemble boundary contribution with value
     * @param boundary_id Boundary entity ID
     * @param value Boundary value (e.g., flux, T_inf for convection)
     * @param local_assembler Callback
     * @param vector Output: global vector (accumulated)
     */
    void assemble_boundary_with_value(
        Index boundary_id,
        Scalar value,
        LocalVectorAssembler local_assembler,
        DynamicVector& vector);
    
    /**
     * @brief Set update flags for FEValues
     */
    void set_update_flags(UpdateFlags flags) { update_flags_ = flags; }
    
private:
    const DoFHandler* dof_handler_;
    const Mesh* mesh_;
    const FESpace* fe_space_;
    
    UpdateFlags update_flags_;
};

/**
 * @brief Common linear form assemblers
 */
namespace LinearForms {

/**
 * @brief Volume source term
 * 
 * F_i = int_Omega (f * N_i) dx
 * 
 * @param source Source term value
 */
LocalVectorAssembler source(Scalar source);

/**
 * @brief Function-based source term
 * 
 * F_i = int_Omega (f(x) * N_i) dx
 * 
 * @param source_func Function f(x, y, z) returning source value
 */
using SourceFunction = std::function<Scalar(Scalar x, Scalar y, Scalar z)>;
LocalVectorAssembler source_function(SourceFunction source_func);

/**
 * @brief Neumann boundary flux
 * 
 * F_i = int_Gamma (g * N_i) dS
 * 
 * @param flux Flux value (g)
 */
LocalVectorAssembler neumann_flux(Scalar flux);

/**
 * @brief Convection boundary (Robin BC)
 * 
 * For convection: q = h * (T - T_inf)
 * The linear part: F_i = int_Gamma (h * T_inf * N_i) dS
 * 
 * @param h Convection coefficient
 * @param T_inf Ambient temperature
 */
LocalVectorAssembler convection_rhs(Scalar h, Scalar T_inf);

/**
 * @brief Thermal strain load vector
 * 
 * For thermal expansion: F_i = int_Omega (D * eps_th * B_i) dx
 * where eps_th = alpha * (T - T_ref) * I
 * 
 * @param alpha Thermal expansion coefficient
 * @param delta_T Temperature change (T - T_ref)
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @param dim Spatial dimension
 */
LocalVectorAssembler thermal_strain(Scalar alpha, Scalar delta_T,
                                    Scalar E, Scalar nu, int dim);

}  // namespace LinearForms

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_LINEAR_FORM_HPP
