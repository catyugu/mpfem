#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "coupling/joule_heating.hpp"
#include "coupling/temperature_dependent_coefficient.hpp"
#include "core/logger.hpp"
#include <deque>

namespace mpfem {

enum class IterationMethod { Picard, Anderson };

struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

/**
 * @brief Coupling manager for electro-thermal analysis.
 * 
 * Design principle: Coupling logic is centralized here, NOT in single-field solvers.
 * This manager handles:
 * - Temperature-dependent conductivity coupling
 * - Joule heating coupling
 * - Picard iteration for coupled solve
 */
class CouplingManager {
public:
    CouplingManager() = default;
    
    void setElectrostaticsSolver(ElectrostaticsSolver* s) { esSolver_ = s; }
    void setHeatTransferSolver(HeatTransferSolver* s) { htSolver_ = s; }
    void setTolerance(Real tol) { tol_ = tol; }
    void setMaxIterations(int n) { maxIter_ = n; }
    
    /// Enable temperature-dependent conductivity for specific domains
    void enableTempDependentConductivity(const std::set<int>& domains = {}) {
        tempDepDomains_ = domains;
        hasTempDepSigma_ = true;
    }
    
    /// Set material parameters for temperature-dependent conductivity
    void setTempDepMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        ensureTempDepCoupling();
        tempDepCoupling_->setMaterial(domainId, rho0, alpha, tref);
    }
    
    /// Set constant conductivity for a domain
    void setConstantConductivity(int domainId, Real sigma) {
        ensureTempDepCoupling();
        tempDepCoupling_->setConstant(domainId, sigma);
    }
    
    CouplingResult solve() {
        CouplingResult result;
        if (!esSolver_ || !htSolver_) return result;
        
        for (int i = 0; i < maxIter_; ++i) {
            // Update temperature-dependent conductivity
            if (hasTempDepSigma_ && tempDepCoupling_) {
                tempDepCoupling_->setTemperatureField(&htSolver_->field());
                esSolver_->setConductivity(tempDepCoupling_->getConductivity());
            }
            
            // Solve electrostatics
            esSolver_->assemble();
            esSolver_->solve();
            
            // Update Joule heat and solve heat transfer
            updateJouleHeat();
            htSolver_->assemble();
            htSolver_->solve();
            
            // Compute error
            Real err = computeError();
            result.iterations = i + 1;
            result.residual = err;
            
            LOG_INFO << "Coupling iteration " << (i+1) << ", residual = " << err;
            
            if (err < tol_) {
                result.converged = true;
                break;
            }
        }
        return result;
    }
    
private:
    void ensureTempDepCoupling() {
        if (!tempDepCoupling_) {
            tempDepCoupling_ = std::make_unique<TemperatureDependentConductivityCoupling>();
        }
    }
    
    void updateJouleHeat() {
        if (!jouleHeating_) {
            jouleHeating_ = std::make_unique<JouleHeatingCoupling>();
            jouleHeating_->setDomains(jouleHeatDomains_);
        }
        
        jouleHeating_->setPotentialField(&esSolver_->field());
        jouleHeating_->setConductivity(esSolver_->conductivity());
        htSolver_->setHeatSource(jouleHeating_->getHeatSource());
    }
    
    Real computeError() {
        if (prevT_.size() == 0) {
            prevT_ = htSolver_->field().values();
            return 1.0;
        }
        Real diff = (htSolver_->field().values() - prevT_).norm();
        prevT_ = htSolver_->field().values();
        return diff / (htSolver_->field().values().norm() + 1e-15);
    }
    
    ElectrostaticsSolver* esSolver_ = nullptr;
    HeatTransferSolver* htSolver_ = nullptr;
    
    // Coupling modules
    std::unique_ptr<JouleHeatingCoupling> jouleHeating_;
    std::unique_ptr<TemperatureDependentConductivityCoupling> tempDepCoupling_;
    
    // Configuration
    std::set<int> tempDepDomains_;
    std::set<int> jouleHeatDomains_;
    bool hasTempDepSigma_ = false;
    
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP
