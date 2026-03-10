/**
 * @file fe_space.hpp
 * @brief Finite element space - manages the collection of finite elements on a mesh
 */

#ifndef MPFEM_DOF_FE_SPACE_HPP
#define MPFEM_DOF_FE_SPACE_HPP

#include "mesh/mesh.hpp"
#include "fem/fe_base.hpp"
#include "fem/fe_collection.hpp"
#include "core/types.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace mpfem {

/**
 * @brief Finite element space on a mesh
 * 
 * Associates finite elements with mesh cells, handling multiple
 * domains and element types.
 */
class FESpace {
public:
    /**
     * @brief Construct FESpace with uniform FE across all domains
     * @param mesh The mesh
     * @param fe_name FE name (e.g., "Lagrange1", "Lagrange2")
     * @param n_components Number of components (1 for scalar, 2/3 for vector)
     */
    FESpace(const Mesh* mesh, const std::string& fe_name, int n_components = 1);

    ~FESpace() = default;

    // Accessors
    const Mesh* mesh() const { return mesh_; }
    int n_components() const { return n_components_; }
    Index n_dofs() const { return n_dofs_; }

    /**
     * @brief Get FE for a cell by global cell index
     * @param global_cell_idx Global cell index
     * @return Pointer to finite element (nullptr if not found)
     */
    const FiniteElement* get_fe(Index global_cell_idx) const;

    /**
     * @brief Get number of dofs per cell for given geometry type
     */
    int dofs_per_cell(GeometryType geom_type) const;

    /**
     * @brief Get global dof indices for a cell
     * @param block_idx Block index
     * @param local_elem_idx Local element index within block
     * @param dofs Output vector of global dof indices
     */
    void get_cell_dofs(SizeType block_idx, SizeType local_elem_idx,
                       std::vector<Index>& dofs) const;

    /**
     * @brief Initialize and compute DoF numbering
     */
    void initialize();

private:
    const Mesh* mesh_;
    int n_components_;
    Index n_dofs_;

    /// FE for each geometry type (all use same degree)
    std::unordered_map<GeometryType, std::unique_ptr<FiniteElement>> fe_by_geom_;

    /// Mapping: global cell index -> (block_idx, local_elem_idx)
    struct CellLocation {
        SizeType block_idx;
        SizeType local_idx;
        GeometryType geom_type;
    };
    std::vector<CellLocation> cell_locations_;

    /// Build cell location map
    void build_cell_map();
};

} // namespace mpfem

#endif // MPFEM_DOF_FE_SPACE_HPP
