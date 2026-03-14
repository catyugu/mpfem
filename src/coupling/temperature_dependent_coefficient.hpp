#ifndef MPFEM_COUPLING_TEMPERATURE_DEPENDENT_CONDUCTIVITY_HPP
#define MPFEM_COUPLING_TEMPERATURE_DEPENDENT_CONDUCTIVITY_HPP

#include "fe/coefficient.hpp"
#include "fe/grid_function.hpp"
#include "core/types.hpp"
#include <vector>
#include <set>

namespace mpfem {

/**
 * @file temperature_dependent_coefficient.hpp
 * @brief Temperature-dependent conductivity for coupled electro-thermal analysis.
 * 
 * Design principle: Field-dependent material parameters should be centralized
 * and independent from single-field solvers.
 */

/**
 * @brief Temperature-dependent electrical conductivity
 * 
 * Model: sigma = 1 / (rho0 * (1 + alpha * (T - Tref)))
 * where:
 *   rho0  = resistivity at reference temperature
 *   alpha = temperature coefficient of resistivity
 *   Tref  = reference temperature
 *   T     = temperature field
 */
class TemperatureDependentSigma : public Coefficient {
public:
    /// Set material parameters for a domain
    void setMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        ensureSize(domainId);
        rho0_[domainId - 1] = rho0;
        alpha_[domainId - 1] = alpha;
        tref_[domainId - 1] = tref;
    }
    
    /// Set constant conductivity (non-temperature-dependent) for a domain
    void setConstant(int domainId, Real sigma) {
        ensureSize(domainId);
        sigma0_[domainId - 1] = sigma;
        rho0_[domainId - 1] = 0.0;  // rho0 = 0 means constant conductivity
    }
    
    /// Set temperature field (non-owning pointer)
    void setTemperatureField(const GridFunction* T) { T_ = T; }
    
    /// Restrict to specific domains (empty = all domains)
    void setDomains(const std::set<int>& domains) { domains_ = domains; }
    
    Real eval(ElementTransform& trans) const override {
        int attr = static_cast<int>(trans.attribute());
        if (attr < 1 || attr > static_cast<int>(rho0_.size())) return 1.0;
        
        // Check domain restriction
        if (!domains_.empty() && domains_.find(attr) == domains_.end()) {
            return 1.0;
        }
        
        Real rho0 = rho0_[attr - 1];
        
        // rho0 <= 0 means use constant conductivity
        if (rho0 <= 0.0) {
            return sigma0_[attr - 1];
        }
        
        Real alpha = alpha_[attr - 1];
        Real tref = tref_[attr - 1];
        
        // Get temperature
        Real temp = tref;
        if (T_) {
            const auto& ip = trans.integrationPoint();
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            temp = T_->eval(trans.elementIndex(), xi);
        }
        
        // Linear resistivity model: rho = rho0 * (1 + alpha * (T - Tref))
        Real factor = 1.0 + alpha * (temp - tref);
        
        // Numerical protection: prevent negative or zero resistivity
        if (factor <= 0.0) {
            factor = 1e-10;
        }
        
        Real rho = rho0 * factor;
        return 1.0 / rho;
    }
    
private:
    void ensureSize(int domainId) {
        if (static_cast<int>(rho0_.size()) < domainId) {
            rho0_.resize(domainId, 0.0);
            alpha_.resize(domainId, 0.0);
            tref_.resize(domainId, 298.0);
            sigma0_.resize(domainId, 0.0);
        }
    }
    
    std::vector<Real> rho0_;   ///< Resistivity at reference temperature
    std::vector<Real> alpha_;  ///< Temperature coefficient
    std::vector<Real> tref_;   ///< Reference temperature
    std::vector<Real> sigma0_; ///< Constant conductivity (used when rho0 <= 0)
    const GridFunction* T_ = nullptr;  ///< Temperature field (non-owning)
    std::set<int> domains_;    ///< Domain restriction
};

/**
 * @brief Manager for temperature-dependent conductivity coupling
 */
class TemperatureDependentConductivityCoupling {
public:
    /// Set temperature field
    void setTemperatureField(const GridFunction* T) { 
        tempField_ = T;
        if (sigma_) {
            sigma_->setTemperatureField(T);
        }
    }
    
    /// Set material for a domain (temperature-dependent)
    void setMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        ensureSigma();
        sigma_->setMaterial(domainId, rho0, alpha, tref);
    }
    
    /// Set constant conductivity for a domain
    void setConstant(int domainId, Real sigma) {
        ensureSigma();
        sigma_->setConstant(domainId, sigma);
    }
    
    /// Get the conductivity coefficient
    const Coefficient* getConductivity() {
        return sigma_.get();
    }
    
    /// Update after temperature changes
    void update() {
        // Coefficient evaluates lazily
    }
    
private:
    void ensureSigma() {
        if (!sigma_) {
            sigma_ = std::make_unique<TemperatureDependentSigma>();
        }
    }
    
    const GridFunction* tempField_ = nullptr;
    std::unique_ptr<TemperatureDependentSigma> sigma_;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_TEMPERATURE_DEPENDENT_CONDUCTIVITY_HPP
