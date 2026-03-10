/**
 * @file fe_collection.hpp
 * @brief Finite element collection factory
 */

#ifndef MPFEM_FEM_FE_COLLECTION_HPP
#define MPFEM_FEM_FE_COLLECTION_HPP

#include "fe_base.hpp"
#include "mesh/element.hpp"
#include "core/exception.hpp"
#include <memory>
#include <string>

namespace mpfem {

/**
 * @brief Create finite element by geometry type and degree
 * (declared in fe_base.hpp with default argument)
 */
std::unique_ptr<FiniteElement> create_fe(
    GeometryType geom_type, int degree, int n_components);

/**
 * @brief Factory for creating FEs by name
 */
class FECollection {
public:
    /**
     * @brief Create FE by name
     * @param name FE name (e.g., "Lagrange1", "Lagrange2", "H1")
     * @param geom_type Geometry type
     * @param n_components Number of components
     */
    static std::unique_ptr<FiniteElement> create(
        const std::string& name, GeometryType geom_type, int n_components = 1);
};

}  // namespace mpfem

#endif  // MPFEM_FEM_FE_COLLECTION_HPP