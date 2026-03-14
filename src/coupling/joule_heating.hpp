#ifndef MPFEM_COUPLING_JOULE_HEATING_HPP
#define MPFEM_COUPLING_JOULE_HEATING_HPP

#include "fe/coefficient.hpp"
#include "fe/grid_function.hpp"
#include "core/types.hpp"
#include <memory>
#include <set>

namespace mpfem {

/**
 * @file joule_heating.hpp
 * @brief Joule heating coupling coefficient - independent from single-field solvers.
 * 
 * Design principle: Coupling should be separate from individual field solvers.
 * This module provides the coupling between electrostatics and heat transfer.
 */

/**
 * @brief Joule heat coefficient: Q = sigma * |grad V|^2
 * 
 * Computes the volumetric heat source from electrical current:
 * Q = sigma * (dV/dx)^2 + sigma * (dV/dy)^2 + sigma * (dV/dz)^2
 */
class JouleHeatCoefficient : public Coefficient {
public:
    /// Set electric potential field (non-owning pointer)
    void setPotential(const GridFunction* V) { V_ = V; }
    
    /// Set conductivity coefficient (non-owning pointer)
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// Restrict to specific domains (empty = all domains)
    void setDomains(const std::set<int>& domains) { domains_ = domains; }
    
    Real eval(ElementTransform& trans) const override {
        if (!V_ || !sigma_) return 0.0;
        
        // Check domain restriction
        if (!domains_.empty()) {
            int attr = static_cast<int>(trans.attribute());
            if (domains_.find(attr) == domains_.end()) return 0.0;
        }
        
        // Key: evaluate sigma first, then gradient
        // Because gradient() calls setIntegrationPoint which changes trans state
        Real sigma_val = sigma_->eval(trans);
        Vector3 g = V_->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
        return sigma_val * g.squaredNorm();
    }
    
private:
    const GridFunction* V_ = nullptr;
    const Coefficient* sigma_ = nullptr;
    std::set<int> domains_;
};

/**
 * @brief Manager for Joule heating coupling
 */
class JouleHeatingCoupling {
public:
    /// Set the electrostatics potential field
    void setPotentialField(const GridFunction* V) { potential_ = V; }
    
    /// Set the conductivity coefficient
    void setConductivity(const Coefficient* sigma) { conductivity_ = sigma; }
    
    /// Restrict coupling to specific domains
    void setDomains(const std::set<int>& domains) { domains_ = domains; }
    
    /// Get the Joule heat coefficient (lazy creation)
    const Coefficient* getHeatSource() {
        if (!jouleHeat_) {
            jouleHeat_ = std::make_unique<JouleHeatCoefficient>();
        }
        jouleHeat_->setPotential(potential_);
        jouleHeat_->setConductivity(conductivity_);
        jouleHeat_->setDomains(domains_);
        return jouleHeat_.get();
    }
    
    /// Update coupling (call after potential field changes)
    void update() {
        // The coefficient evaluates lazily, so no explicit update needed
    }
    
private:
    const GridFunction* potential_ = nullptr;
    const Coefficient* conductivity_ = nullptr;
    std::set<int> domains_;
    std::unique_ptr<JouleHeatCoefficient> jouleHeat_;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_JOULE_HEATING_HPP
