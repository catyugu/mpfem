#ifndef MPFEM_STEADY_PROBLEM_HPP
#define MPFEM_STEADY_PROBLEM_HPP

#include "problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

struct SteadyResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

class SteadyProblem : public Problem {
public:

    int couplingMaxIter = 15;
    Real couplingTol = 1e-6;
    
    bool hasElectrostatics() const { return electrostatics != nullptr; }
    bool hasHeatTransfer() const { return heatTransfer != nullptr; }
    bool hasStructural() const { return structural != nullptr; }
    bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
    bool hasThermalExpansion() const { return hasHeatTransfer() && hasStructural(); }
    bool isCoupled() const { return hasJouleHeating() || hasThermalExpansion(); }
    
    SteadyResult solve() {
        ScopedTimer timer("Coupling solve");
        SteadyResult result;

        if (!isCoupled()) {
            if (hasElectrostatics()) {
                electrostatics->assemble();
                electrostatics->solve();
            }
            return result;
        }

        for (int i = 0; i < couplingMaxIter; ++i) {
            electrostatics->assemble();
            electrostatics->solve();
            heatTransfer->assemble();
            heatTransfer->solve();
            
            Real err = computeCouplingError();
            result.iterations = i + 1;
            result.residual = err;
            LOG_INFO << "Coupling iteration " << (i + 1) << ", residual = " << err;
            if (err < couplingTol) { result.converged = true; break; }
        }

        if (hasStructural()) {
            structural->assemble();
            structural->solve();
        }
        return result;
    }

private:
    Vector prevT_;
    
    Real computeCouplingError() {
        if (!heatTransfer) return 0.0;
        const auto& T = heatTransfer->field().values();
        if (prevT_.size() == 0) { prevT_ = T; return 1.0; }
        Real diff = (T - prevT_).norm();
        prevT_ = T;
        return diff / (T.norm() + 1e-15);
    }
};

}  // namespace mpfem

#endif  // MPFEM_STEADY_PROBLEM_HPP
