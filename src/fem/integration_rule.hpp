/**
 * @file integration_rule.hpp
 * @brief Integration rules for different element types
 */

#ifndef MPFEM_FEM_INTEGRATION_RULE_HPP
#define MPFEM_FEM_INTEGRATION_RULE_HPP

#include "quadrature.hpp"
#include "mesh/element.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Integration rule for a specific element geometry
 */
class IntegrationRule {
public:
    IntegrationRule() = default;
    
    /// Create integration rule for given geometry and order
    static std::unique_ptr<IntegrationRule> create(
        GeometryType geom_type, int order);
    
    /// Number of quadrature points
    int n_points() const { return static_cast<int>(points_.size()); }
    
    /// Get quadrature point
    const IntegrationPoint& point(int q) const { return points_[q]; }
    
    /// Get all points
    const std::vector<IntegrationPoint>& points() const { return points_; }
    
    /// Get geometry type
    GeometryType geometry_type() const { return geom_type_; }
    
    /// Get integration order
    int order() const { return order_; }
    
protected:
    GeometryType geom_type_ = GeometryType::Invalid;
    int order_ = 0;
    std::vector<IntegrationPoint> points_;
};

}  // namespace mpfem

#endif  // MPFEM_FEM_INTEGRATION_RULE_HPP