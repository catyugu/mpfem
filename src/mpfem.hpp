#ifndef MPFEM_HPP
#define MPFEM_HPP

/**
 * @file mpfem.hpp
 * @brief Unified header for the mpfem library.
 *
 * Include this header to use all mpfem components.
 */

// Core
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/types.hpp"

// Mesh
#include "io/mphtxt_reader.hpp"
#include "mesh/element.hpp"
#include "mesh/geometry.hpp"
#include "mesh/mesh.hpp"
#include "mesh/vertex.hpp"

// Finite Element
#include "fe/coefficient.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "fe/quadrature.hpp"
#include "fe/reference_element.hpp"
#include "fe/shape_function.hpp"

// Operator (unified LinearOperator interface)
#include "operator/linear_operator.hpp"
#include "operator/operator_config.hpp"
#include "operator/operator_factory.hpp"
#include "operator/parameter_list.hpp"

// Version
#define MPFEM_VERSION_MAJOR 0
#define MPFEM_VERSION_MINOR 1
#define MPFEM_VERSION_PATCH 0

namespace mpfem {

    /// Get version string
    inline const char* version()
    {
        return "0.1.0";
    }

    /// Get version as integer (major * 10000 + minor * 100 + patch)
    inline int versionNumber()
    {
        return MPFEM_VERSION_MAJOR * 10000 + MPFEM_VERSION_MINOR * 100 + MPFEM_VERSION_PATCH;
    }

} // namespace mpfem

#endif // MPFEM_HPP
