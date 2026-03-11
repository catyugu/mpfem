/**
 * @file fe_collection.cpp
 * @brief Implementation of finite element collection
 */

#include "fe_collection.hpp"
#include "fe_h1.hpp"
#include "core/exception.hpp"
#include <string>

namespace mpfem {

// create_fe is implemented in fe_h1.cpp

std::unique_ptr<FiniteElement> FECollection::create(
    const std::string& name, GeometryType geom_type, int n_components) {

    int degree = 1;

    if (name.find("Lagrange") == 0 && name.size() > 8) {
        degree = std::stoi(name.substr(8));
    } else if (name == "H1_2") {
        degree = 2;
    }

    return create_fe(geom_type, degree, n_components);
}

std::shared_ptr<const FiniteElement> FECollection::create_cached(
    const std::string& name, GeometryType geom_type, int n_components) {
    
    int degree = 1;

    if (name.find("Lagrange") == 0 && name.size() > 8) {
        degree = std::stoi(name.substr(8));
    } else if (name == "H1_2") {
        degree = 2;
    }

    return get_cached_fe(geom_type, degree, n_components);
}

}  // namespace mpfem
