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
    }
    LOG_ERROR << "Unknown time scheme";
    return nullptr;
}

}  // namespace mpfem
