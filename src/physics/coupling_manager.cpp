#include "physics/coupling_manager.hpp"
#include "core/exception.hpp"
#include <Eigen/Dense>

namespace mpfem {

// =============================================================================
// CouplingManager Implementation
// =============================================================================

CouplingResult CouplingManager::solve() {
    CouplingResult result;
    
    if (!electrostaticsSolver_ || !heatTransferSolver_) {
        LOG_ERROR << "CouplingManager: solvers not configured";
        return result;
    }
    
    converged_ = false;
    currentIteration_ = 0;
    temperatureHistory_.clear();
    voltageHistory_.clear();
    
    // Store initial solutions
    previousTemperature_ = heatTransferSolver_->field().values();
    previousVoltage_ = electrostaticsSolver_->field().values();
    
    LOG_INFO << "Starting coupled electro-thermal solve (method: " 
             << (method_ == IterationMethod::Picard ? "Picard" : "Anderson") << ")";
    
    for (int iter = 0; iter < maxIterations_; ++iter) {
        currentIteration_ = iter + 1;
        
        // Perform one coupling iteration
        Real error = iterate(iter);
        
        result.iterations = currentIteration_;
        result.residual = error;
        currentResidual_ = error;
        
        LOG_INFO << "Coupling iteration " << currentIteration_ 
                 << ", error = " << std::scientific << error;
        
        // Check convergence
        if (error < tolerance_) {
            converged_ = true;
            result.converged = true;
            LOG_INFO << "Coupling converged after " << currentIteration_ << " iterations";
            break;
        }
    }
    
    if (!converged_) {
        LOG_WARN << "Coupling did not converge after " << maxIterations_ << " iterations";
    }
    
    return result;
}

Real CouplingManager::iterate(int iter) {
    // 1. Solve electrostatics
    LOG_DEBUG << "  Solving electrostatics...";
    
    // Temperature-dependent conductivity is handled by the coefficient itself
    // No need to manually update - the coefficient reads temperature at each integration point
    
    electrostaticsSolver_->assemble();
    electrostaticsSolver_->solve();
    
    // 2. Update Joule heating source
    if (jouleHeatingEnabled_) {
        updateJouleHeating();
    }
    
    // 3. Solve heat transfer
    LOG_DEBUG << "  Solving heat transfer...";
    heatTransferSolver_->assemble();
    heatTransferSolver_->solve();
    
    // 4. Compute errors
    Real voltageError = computeRelativeError(
        electrostaticsSolver_->field().values(), previousVoltage_);
    Real temperatureError = computeRelativeError(
        heatTransferSolver_->field().values(), previousTemperature_);
    
    LOG_DEBUG << "  Voltage error: " << std::scientific << voltageError
              << ", Temperature error: " << temperatureError;
    
    // 5. Apply acceleration/relaxation
    if (method_ == IterationMethod::Anderson && iter > 0) {
        // Anderson acceleration for temperature
        Vector acceleratedT = applyAndersonAcceleration(
            heatTransferSolver_->field().values(), temperatureHistory_);
        
        // Update solution with accelerated value
        heatTransferSolver_->field().values() = acceleratedT;
        
        // Store history
        temperatureHistory_.push_back(heatTransferSolver_->field().values());
        if (static_cast<int>(temperatureHistory_.size()) > andersonDepth_) {
            temperatureHistory_.pop_front();
        }
    } else {
        // Picard with optional under-relaxation
        if (relaxation_ < 1.0) {
            heatTransferSolver_->field().values() = 
                relaxation_ * heatTransferSolver_->field().values() + 
                (1.0 - relaxation_) * previousTemperature_;
        }
    }
    
    // Store previous solutions
    previousTemperature_ = heatTransferSolver_->field().values();
    previousVoltage_ = electrostaticsSolver_->field().values();
    
    // Return the maximum relative error
    return std::max(voltageError, temperatureError);
}

void CouplingManager::updateJouleHeating() {
    // Set Joule heating source: Q = σ|∇V|²
    heatTransferSolver_->setJouleHeating(
        &electrostaticsSolver_->field(),
        electrostaticsSolver_->conductivity()
    );
}

Real CouplingManager::computeRelativeError(const Vector& current, 
                                            const Vector& previous) const {
    Real diff = (current - previous).norm();
    Real norm = current.norm();
    
    if (norm < 1e-15) {
        return diff;
    }
    
    return diff / norm;
}

// =============================================================================
// Anderson Acceleration Implementation
// =============================================================================

Vector CouplingManager::applyAndersonAcceleration(const Vector& current,
                                                   std::deque<Vector>& history) {
    // Store current iteration
    history.push_back(current);
    
    // Limit history size
    while (static_cast<int>(history.size()) > andersonDepth_ + 1) {
        history.pop_front();
    }
    
    int m = static_cast<int>(history.size()) - 1;
    
    if (m <= 0) {
        return current;
    }
    
    // Compute residuals: r_k = G(x_k) - x_k = x_{k+1} - x_k
    std::deque<Vector> residuals;
    for (int k = 0; k < m; ++k) {
        residuals.push_back(history[k + 1] - history[k]);
    }
    
    // Solve least-squares problem for optimal weights
    Vector weights = solveAndersonLS(residuals);
    
    // Compute accelerated solution: x* = x_0 + sum_k w_k * (x_{k+1} - x_k)
    Vector result = history[0];
    for (int k = 0; k < m; ++k) {
        result += weights[k] * (history[k + 1] - history[k]);
    }
    
    return result;
}

Vector CouplingManager::solveAndersonLS(const std::deque<Vector>& residuals) {
    int m = static_cast<int>(residuals.size());
    int n = residuals[0].size();
    
    // Build matrix of residuals: R = [r_0, r_1, ..., r_{m-1}]
    Eigen::MatrixXd R(n, m);
    for (int k = 0; k < m; ++k) {
        R.col(k) = residuals[k];
    }
    
    // Build RHS: -r_{m-1}
    Eigen::VectorXd rhs = -residuals[m - 1];
    
    // Solve least-squares using Householder QR (more stable link)
    Eigen::VectorXd weights = R.householderQr().solve(rhs);
    
    // Normalize weights to sum to 1 (for stability)
    Real sum = weights.sum();
    if (std::abs(sum) > 1e-10) {
        // weights /= sum;  // Optional: comment out for standard Anderson
    }
    
    // Add weight for last residual (w_m = 1 - sum(w_1..w_{m-1}))
    Vector result(m);
    for (int k = 0; k < m - 1; ++k) {
        result[k] = weights[k];
    }
    result[m - 1] = 1.0 - weights.head(m - 1).sum();
    
    return result;
}

}  // namespace mpfem
