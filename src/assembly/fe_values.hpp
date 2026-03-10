/**
 * @file fe_values.hpp
 * @brief FEValues - evaluates finite element quantities at quadrature points
 */

#ifndef MPFEM_ASSEMBLY_FE_VALUES_HPP
#define MPFEM_ASSEMBLY_FE_VALUES_HPP

#include "core/types.hpp"
#include "fem/fe_base.hpp"
#include "fem/element_transformation.hpp"
#include "mesh/mesh.hpp"
#include "dof/field_registry.hpp"
#include "dof/field_space.hpp"
#include <vector>
#include <memory>

namespace mpfem {

struct QuadraturePointData {
    Scalar JxW;
    Point<3> physical_point;
    Tensor<1, 3> normal;
    std::vector<Scalar> shape_values;
    std::vector<Tensor<1, 3>> shape_gradients;
};

/**
 * @brief FEValues - evaluates finite element quantities on physical cells
 * 
 * USAGE PATTERN:
 * 1. reinit(field, cell_id) - set current cell and field
 * 2. Access quadrature point data (JxW, shape_grad, etc.)
 * 3. Query coupled field values (field_value, field_gradient)
 * 4. Assemble to global system (assemble_local_to_global)
 */
class FEValues {
public:
    FEValues(const FiniteElement* fe, UpdateFlags flags = UpdateFlags::UpdateDefault);
    
    // ============================================================
    // Reinitialize for a new cell
    // ============================================================
    
    void reinit(const FieldSpace& field, Index cell_id);
    void reinit_face(const FieldSpace& field, Index face_id, Index cell_id, int local_face_index);
    
    // ============================================================
    // High-Level Assembly (hides DoF indices from users)
    // ============================================================
    
    void assemble_local_to_global(SparseMatrix& K, const DynamicMatrix& local_K) const;
    void assemble_rhs(DynamicVector& f, const DynamicVector& local_f) const;
    
    // ============================================================
    // Basic Accessors
    // ============================================================
    
    int n_quadrature_points() const { return n_qpoints_; }
    int dofs_per_cell() const { return dofs_per_cell_; }
    int n_components() const { return n_components_; }
    const FiniteElement* fe() const { return fe_; }
    
    // ============================================================
    // Quadrature Point Data
    // ============================================================
    
    Scalar JxW(int q) const { return qp_data_[q].JxW; }
    Scalar shape_value(int i, int q) const { return qp_data_[q].shape_values[i]; }
    const Tensor<1, 3>& shape_grad(int i, int q) const { return qp_data_[q].shape_gradients[i]; }
    const Point<3>& quadrature_point(int q) const { return qp_data_[q].physical_point; }
    const Tensor<1, 3>& normal(int q) const { return qp_data_[q].normal; }
    const std::vector<Scalar>& shape_values(int q) const { return qp_data_[q].shape_values; }
    const std::vector<Tensor<1, 3>>& shape_gradients(int q) const { return qp_data_[q].shape_gradients; }
    
    // ============================================================
    // Current Field Value Access
    // ============================================================
    
    Scalar value(int q) const;
    Tensor<1, 3> gradient(int q) const;
    void values(std::vector<Scalar>& vals) const;
    void gradients(std::vector<Tensor<1, 3>>& grads) const;
    Tensor<1, 3> vector_value(int q) const;
    
    // ============================================================
    // Coupled Field Access (by field name)
    // ============================================================
    
    void set_field_registry(const FieldRegistry* registry) { field_registry_ = registry; }
    
    Scalar field_value(const FieldID& field_name, int q) const;
    void field_values(const FieldID& field_name, std::vector<Scalar>& values) const;
    Tensor<1, 3> field_vector(const FieldID& field_name, int q) const;
    void field_vectors(const FieldID& field_name, std::vector<Tensor<1, 3>>& values) const;
    Tensor<1, 3> field_gradient(const FieldID& field_name, int q) const;
    void field_gradients(const FieldID& field_name, std::vector<Tensor<1, 3>>& gradients) const;
    
    // ============================================================
    // Element Transformation
    // ============================================================
    
    Scalar det_jacobian() const { return trans_.det_jacobian(); }
    const Tensor<2, 3>& jacobian() const { return trans_.jacobian(); }
    const Tensor<2, 3>& inverse_jacobian() const { return trans_.inverse_jacobian(); }

private:
    const FiniteElement* fe_;
    UpdateFlags flags_;
    
    int n_qpoints_;
    int dofs_per_cell_;
    int n_components_;
    
    ElementTransformation trans_;
    std::vector<QuadraturePointData> qp_data_;
    
    bool is_face_;
    Index current_face_id_;
    int current_local_face_;
    
    const FieldSpace* current_field_ = nullptr;
    Index current_cell_id_ = InvalidIndex;
    
    std::vector<Index> cell_dofs_;  // Cached DoF indices for current cell
    const FieldRegistry* field_registry_ = nullptr;
    
    void compute_cell_data(const Mesh& mesh, Index cell_id);
    void compute_face_data(const Mesh& mesh, Index face_id, Index cell_id, int local_face);
};

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_FE_VALUES_HPP
