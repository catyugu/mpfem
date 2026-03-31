#include "time_integrator.hpp"
#include "time/bdf1_integrator.hpp"
#include "time/bdf2_integrator.hpp"
#include "core/logger.hpp"

#include <memory>

namespace mpfem {

std::unique_ptr<TimeIntegrator> createTimeIntegrator(TimeScheme scheme) {
    switch (scheme) {
        case TimeScheme::BDF1:
            return std::make_unique<BDF1Integrator>();
        case TimeScheme::BDF2:
            return std::make_unique<BDF2Integrator>();
    }
    LOG_ERROR << "Unknown time scheme";
    return {};
}

}  // namespace mpfem
