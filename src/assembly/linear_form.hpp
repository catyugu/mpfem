/**
 * @file linear_form.hpp
 * @brief LinearForm - interface for assembling right-hand side vectors
 */

#ifndef MPFEM_ASSEMBLY_LINEAR_FORM_HPP
#define MPFEM_ASSEMBLY_LINEAR_FORM_HPP

#include "core/types.hpp"
#include "dof/field_space.hpp"
#include "mesh/mesh.hpp"
#include <functional>
#include <vector>

namespace mpfem {

class FEValues;

using LocalVectorAssembler = std::function<void(
    Index cell_id,
    const FEValues& fe_values,
    DynamicVector& local_vector)>;

class LinearForm {
public:
    explicit LinearForm(const FieldSpace* field);
    
    void assemble(LocalVectorAssembler local_assembler, DynamicVector& vector);
    
    void assemble_with_source(
        LocalVectorAssembler local_assembler,
        DynamicVector& vector,
        const std::unordered_map<Index, Scalar>& sources);
    
    void assemble_boundary(Index boundary_id,
                          LocalVectorAssembler local_assembler,
                          DynamicVector& vector);
    
    void set_update_flags(UpdateFlags flags) { update_flags_ = flags; }

private:
    const FieldSpace* field_;
    const Mesh* mesh_;
    UpdateFlags update_flags_ = UpdateFlags::UpdateDefault;
};

namespace LinearForms {
    LocalVectorAssembler source(Scalar source);
    using SourceFunction = std::function<Scalar(Scalar x, Scalar y, Scalar z)>;
    LocalVectorAssembler source_function(SourceFunction source_func);
    LocalVectorAssembler neumann_flux(Scalar flux);
    LocalVectorAssembler convection_rhs(Scalar h, Scalar T_inf);
    LocalVectorAssembler thermal_strain(Scalar alpha, Scalar delta_T,
                                        Scalar E, Scalar nu, int dim);
}

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_LINEAR_FORM_HPP
