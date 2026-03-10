/**
 * @file fe_values.hpp
 * @brief FEValues - evaluates finite element quantities at quadrature points
 * 
 * This class provides the bridge between reference element computations
 * and physical element assembly. It handles:
 * - Jacobian transformation and integration weights (JxW)
 * - Physical gradients (transformed from reference gradients)
 * - Projection of global solution to local quadrature points
 */

#ifndef MPFEM_ASSEMBLY_FE_VALUES_HPP
#define MPFEM_ASSEMBLY_FE_VALUES_HPP

#include "core/types.hpp"
#include "fem/fe_base.hpp"
#include "fem/element_transformation.hpp"
#include "mesh/mesh.hpp"
#include <vector>
#include <memory>

namespace mpfem {

/**
 * @brief Update flags for FEValues
 * 
 * Controls which quantities are computed during reinit()
 */
enum class UpdateFlags : int {
    None = 0,
    
    /// Compute Jacobian determinant and JxW
    UpdateJxW = 1 << 0,
    
    /// Compute physical gradients (requires Jacobian inverse)
    UpdateGradients = 1 << 1,
    
    /// Compute shape function values (trivial, just copy from FE)
    UpdateValues = 1 << 2,
    
    /// Compute physical quadrature point positions
    UpdateQuadraturePoints = 1 << 3,
    
    /// Normal vectors for face integration
    UpdateNormals = 1 << 4,
    
    /// Default: values, gradients, JxW
    UpdateDefault = UpdateValues | UpdateGradients | UpdateJxW,
    
    /// All quantities
    UpdateAll = UpdateValues | UpdateGradients | UpdateJxW | UpdateQuadraturePoints | UpdateNormals
};

// Bitwise operators for UpdateFlags
inline UpdateFlags operator|(UpdateFlags a, UpdateFlags b) {
    return static_cast<UpdateFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline UpdateFlags operator&(UpdateFlags a, UpdateFlags b) {
    return static_cast<UpdateFlags>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool has_flag(UpdateFlags flags, UpdateFlags flag) {
    return (flags & flag) == flag;
}

/**
 * @brief Stores precomputed values at a single quadrature point
 */
struct QuadraturePointData {
    Scalar JxW;                          ///< det(J) * weight
    Point<3> physical_point;             ///< Physical coordinates
    Tensor<1, 3> normal;                 ///< Normal vector (for faces)
    
    /// Shape function values at this quadrature point [dofs_per_cell]
    std::vector<Scalar> shape_values;
    
    /// Physical gradients at this quadrature point [dofs_per_cell]
    std::vector<Tensor<1, 3>> shape_gradients;
};

/**
 * @brief FEValues - evaluates finite element quantities on physical cells
 * 
 * Usage:
 * @code
 * FEValues fe_values(fe, update_flags);
 * for (cell in mesh) {
 *     fe_values.reinit(mesh, cell_id);
 *     for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
 *         double jxw = fe_values.JxW(q);
 *         for (int i = 0; i < fe_values.dofs_per_cell(); ++i) {
 *             auto grad_i = fe_values.shape_grad(i, q);
 *             // ... assemble
 *         }
 *     }
 * }
 * @endcode
 */
class FEValues {
public:
    /**
     * @brief Construct FEValues for a given finite element
     * @param fe Pointer to finite element (must remain valid)
     * @param flags Update flags controlling which quantities to compute
     */
    FEValues(const FiniteElement* fe, UpdateFlags flags = UpdateFlags::UpdateDefault);
    
    /**
     * @brief Reinitialize for a new cell
     * @param mesh The mesh
     * @param cell_id Global cell index
     */
    void reinit(const Mesh& mesh, Index cell_id);
    
    /**
     * @brief Reinitialize for a face (for boundary integration)
     * @param mesh The mesh
     * @param face_id Global face index
     * @param cell_id The cell adjacent to this face (for Jacobian computation)
     * @param local_face_index Local face index in the cell
     */
    void reinit_face(const Mesh& mesh, Index face_id, Index cell_id, int local_face_index);
    
    // ============================================================
    // Basic Accessors
    // ============================================================
    
    /// Number of quadrature points
    int n_quadrature_points() const { return n_qpoints_; }
    
    /// Number of DoFs per cell
    int dofs_per_cell() const { return dofs_per_cell_; }
    
    /// Number of components
    int n_components() const { return n_components_; }
    
    /// Get the finite element
    const FiniteElement* fe() const { return fe_; }
    
    // ============================================================
    // Quadrature Point Data
    // ============================================================
    
    /// Get JxW at quadrature point q (det(J) * weight)
    Scalar JxW(int q) const {
        return qp_data_[q].JxW;
    }
    
    /// Get shape function value at quadrature point q
    Scalar shape_value(int i, int q) const {
        return qp_data_[q].shape_values[i];
    }
    
    /// Get physical gradient at quadrature point q
    /// grad_phys = inv(J)^T * grad_ref
    const Tensor<1, 3>& shape_grad(int i, int q) const {
        return qp_data_[q].shape_gradients[i];
    }
    
    /// Get physical quadrature point coordinates
    const Point<3>& quadrature_point(int q) const {
        return qp_data_[q].physical_point;
    }
    
    /// Get normal vector at quadrature point (for face integration)
    const Tensor<1, 3>& normal(int q) const {
        return qp_data_[q].normal;
    }
    
    /// Get all shape values at quadrature point q
    const std::vector<Scalar>& shape_values(int q) const {
        return qp_data_[q].shape_values;
    }
    
    /// Get all shape gradients at quadrature point q
    const std::vector<Tensor<1, 3>>& shape_gradients(int q) const {
        return qp_data_[q].shape_gradients;
    }
    
    // ============================================================
    // Solution Projection (for nonlinear/dependent problems)
    // ============================================================
    
    /**
     * @brief Get solution values at quadrature points
     * @param global_solution Global solution vector
     * @param local_dofs Local DoF indices for current cell
     * @param qpoint_values Output: values at each quadrature point [n_qpoints]
     */
    void get_function_values(const DynamicVector& global_solution,
                             const std::vector<Index>& local_dofs,
                             std::vector<Scalar>& qpoint_values) const;
    
    /**
     * @brief Get solution gradients at quadrature points
     * @param global_solution Global solution vector
     * @param local_dofs Local DoF indices for current cell
     * @param qpoint_grads Output: gradients at each quadrature point [n_qpoints]
     */
    void get_function_gradients(const DynamicVector& global_solution,
                                const std::vector<Index>& local_dofs,
                                std::vector<Tensor<1, 3>>& qpoint_grads) const;
    
    /**
     * @brief Get vector field values at quadrature points (for displacement, etc.)
     * @param global_solution Global solution vector
     * @param local_dofs Local DoF indices for current cell
     * @param qpoint_values Output: vector values at each quadrature point [n_qpoints]
     */
    void get_vector_values(const DynamicVector& global_solution,
                           const std::vector<Index>& local_dofs,
                           std::vector<Tensor<1, 3>>& qpoint_values) const;
    
    // ============================================================
    // Element Transformation Access
    // ============================================================
    
    /// Get Jacobian determinant
    Scalar det_jacobian() const { return trans_.det_jacobian(); }
    
    /// Get Jacobian matrix
    const Tensor<2, 3>& jacobian() const { return trans_.jacobian(); }
    
    /// Get inverse Jacobian matrix
    const Tensor<2, 3>& inverse_jacobian() const { return trans_.inverse_jacobian(); }
    
private:
    const FiniteElement* fe_;
    UpdateFlags flags_;
    
    int n_qpoints_;
    int dofs_per_cell_;
    int n_components_;
    
    ElementTransformation trans_;
    std::vector<QuadraturePointData> qp_data_;
    
    // Face-related data
    bool is_face_;
    Index current_face_id_;
    int current_local_face_;
    
    void compute_cell_data(const Mesh& mesh, Index cell_id);
    void compute_face_data(const Mesh& mesh, Index face_id, Index cell_id, int local_face);
};

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_FE_VALUES_HPP
