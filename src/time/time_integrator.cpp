#include "time_integrator.hpp"
#include "time/bdf1_integrator.hpp"
#include "time/bdf2_integrator.hpp"
#include "core/logger.hpp"

namespace mpfem {

TimeIntegrator* createTimeIntegrator(TimeScheme scheme) {
    switch (scheme) {
        case TimeScheme::BackwardEuler:
            return new BDF1Integrator();
        case TimeScheme::BDF2:
            return new BDF2Integrator();
        case TimeScheme::CrankNicolson:
            LOG_ERROR << "CrankNicolson not implemented yet";
            return nullptr;
        default:
            LOG_ERROR << "Unknown time scheme";
            return nullptr;
    }
}

}  // namespace mpfem
