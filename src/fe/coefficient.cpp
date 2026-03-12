#include "coefficient.hpp"
#include "element_transform.hpp"
#include "grid_function.hpp"
#include "quadrature.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <algorithm>

namespace mpfem {

// =============================================================================
// Coefficient (base class)
// =============================================================================

Real Coefficient::eval(ElementTransform& trans, const IntegrationPoint& ip) const {
    trans.setIntegrationPoint(ip);
    return eval(trans);
}

// =============================================================================
// VectorCoefficient (base class)
// =============================================================================

void VectorCoefficient::eval(ElementTransform& trans, const IntegrationPoint& ip, 
                              Real* result) const {
    trans.setIntegrationPoint(ip);
    eval(trans, result);
}

// =============================================================================
// PWConstCoefficient
// =============================================================================

Real PWConstCoefficient::eval(ElementTransform& trans) const {
    Index attr = trans.attribute();
    if (attr < 1 || static_cast<size_t>(attr) > constants_.size()) {
        MPFEM_THROW(RangeException, 
            "PWConstCoefficient: invalid attribute " + std::to_string(attr) +
            ", valid range is [1, " + std::to_string(constants_.size()) + "]");
    }
    return constants_[attr - 1];
}

// =============================================================================
// PWCoefficient
// =============================================================================

Real PWCoefficient::eval(ElementTransform& trans) const {
    Index attr = trans.attribute();
    
    // Check owned coefficients
    auto it = pieces_.find(attr);
    if (it != pieces_.end()) {
        return it->second->eval(trans);
    }
    
    // Check non-owning references
    auto refIt = pieceRefs_.find(attr);
    if (refIt != pieceRefs_.end()) {
        return refIt->second->eval(trans);
    }
    
    return 0.0;
}

// =============================================================================
// FunctionCoefficient
// =============================================================================

Real FunctionCoefficient::eval(ElementTransform& trans) const {
    Vector3 x;
    trans.transform(trans.integrationPoint(), x);
    
    if (tdFunc_) {
        return tdFunc_(x.x(), x.y(), x.z(), time_);
    }
    
    if (func_) {
        return func_(x.x(), x.y(), x.z());
    }
    
    MPFEM_THROW(Exception, "FunctionCoefficient: no function set");
}

// =============================================================================
// GridFunctionCoefficient
// =============================================================================

Real GridFunctionCoefficient::eval(ElementTransform& trans) const {
    if (!gf_) {
        MPFEM_THROW(Exception, "GridFunctionCoefficient: null GridFunction");
    }
    
    Index elemIdx = trans.elementIndex();
    const IntegrationPoint& ip = trans.integrationPoint();
    
    // For scalar fields
    if (gf_->vdim() == 1) {
        return gf_->eval(elemIdx, ip);
    }
    
    // For vector fields, return the specified component
    // Note: This is a simplified implementation
    // A full implementation would extract just the component
    return gf_->eval(elemIdx, ip);
}

// =============================================================================
// RestrictedCoefficient
// =============================================================================

Real RestrictedCoefficient::eval(ElementTransform& trans) const {
    if (!coef_) {
        MPFEM_THROW(Exception, "RestrictedCoefficient: null coefficient");
    }
    
    Index attr = trans.attribute();
    if (activeAttr_.count(attr) > 0) {
        return coef_->eval(trans);
    }
    
    return 0.0;
}

// =============================================================================
// RatioCoefficient
// =============================================================================

Real RatioCoefficient::eval(ElementTransform& trans) const {
    Real denom = denom_->eval(trans);
    if (std::abs(denom) < 1e-30) {
        MPFEM_THROW(Exception, "RatioCoefficient: division by zero");
    }
    return num_->eval(trans) / denom;
}

// =============================================================================
// TransformedCoefficient
// =============================================================================

Real TransformedCoefficient::eval(ElementTransform& trans) const {
    if (q2_ && transform2_) {
        return transform2_(q1_->eval(trans), q2_->eval(trans));
    }
    if (q1_ && transform1_) {
        return transform1_(q1_->eval(trans));
    }
    MPFEM_THROW(Exception, "TransformedCoefficient: no transform function set");
}

// =============================================================================
// TemperatureDependentConductivityCoefficient
// =============================================================================

void TemperatureDependentConductivityCoefficient::setMaterialFields(
    const std::vector<Real>& rho0,
    const std::vector<Real>& alpha,
    const std::vector<Real>& tref,
    const std::vector<Real>& sigma0)
{
    rho0_ = rho0;
    alpha_ = alpha;
    tref_ = tref;
    sigma0_ = sigma0;
}

Real TemperatureDependentConductivityCoefficient::eval(ElementTransform& trans) const {
    Index attr = trans.attribute();
    
    if (attr < 1 || static_cast<size_t>(attr) > sigma0_.size()) {
        MPFEM_THROW(RangeException, 
            "TemperatureDependentConductivity: invalid attribute " 
            + std::to_string(attr) + ", valid range is [1, " 
            + std::to_string(sigma0_.size()) + "]");
    }
    
    const Real rho0 = rho0_[attr - 1];
    
    // Use resistivity model if rho0 is set
    if (rho0 > 0.0) {
        const Real alpha = alpha_[attr - 1];
        const Real tref = tref_[attr - 1];
        
        Real temp = tref;
        if (temperature_) {
            temp = temperature_->eval(trans.elementIndex(), trans.integrationPoint());
        }
        
        // Linear resistivity model: rho = rho0 * (1 + alpha * (T - Tref))
        Real rho = rho0 * (1.0 + alpha * (temp - tref));
        
        if (!std::isfinite(rho) || rho <= 0.0) {
            MPFEM_THROW(Exception, 
                "TemperatureDependentConductivity: invalid resistivity " 
                + std::to_string(rho) + " at temperature " + std::to_string(temp));
        }
        
        return 1.0 / rho;
    }
    
    // Use constant conductivity
    Real sigma = sigma0_[attr - 1];
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        MPFEM_THROW(Exception, 
            "TemperatureDependentConductivity: invalid conductivity " 
            + std::to_string(sigma) + " for attribute " + std::to_string(attr));
    }
    
    return sigma;
}

// =============================================================================
// TemperatureDependentThermalConductivityCoefficient
// =============================================================================

void TemperatureDependentThermalConductivityCoefficient::setMaterialFields(
    const std::vector<Real>& k0,
    const std::vector<Real>& alpha,
    const std::vector<Real>& tref)
{
    k0_ = k0;
    alpha_ = alpha;
    tref_ = tref;
}

Real TemperatureDependentThermalConductivityCoefficient::eval(ElementTransform& trans) const {
    Index attr = trans.attribute();
    
    if (attr < 1 || static_cast<size_t>(attr) > k0_.size()) {
        MPFEM_THROW(RangeException, 
            "TemperatureDependentThermalConductivity: invalid attribute " 
            + std::to_string(attr) + ", valid range is [1, " 
            + std::to_string(k0_.size()) + "]");
    }
    
    Real temp = tref_[attr - 1];
    if (temperature_) {
        temp = temperature_->eval(trans.elementIndex(), trans.integrationPoint());
    }
    
    // Linear thermal conductivity model: k(T) = k0 * (1 + alpha * (T - Tref))
    Real k0 = k0_[attr - 1];
    Real alpha = alpha_[attr - 1];
    Real tref = tref_[attr - 1];
    
    Real k = k0 * (1.0 + alpha * (temp - tref));
    
    if (!std::isfinite(k) || k <= 0.0) {
        MPFEM_THROW(Exception, 
            "TemperatureDependentThermalConductivity: invalid conductivity " 
            + std::to_string(k) + " at temperature " + std::to_string(temp));
    }
    
    return k;
}

// =============================================================================
// VectorGridFunctionCoefficient
// =============================================================================

VectorGridFunctionCoefficient::VectorGridFunctionCoefficient(const GridFunction* gf)
    : VectorCoefficient(gf ? gf->vdim() : 3), gf_(gf) {
}

void VectorGridFunctionCoefficient::setGridFunction(const GridFunction* gf) {
    gf_ = gf;
    if (gf) {
        vdim_ = gf->vdim();
    }
}

void VectorGridFunctionCoefficient::eval(ElementTransform& trans, Real* result) const {
    if (!gf_) {
        for (int i = 0; i < vdim_; ++i) {
            result[i] = 0.0;
        }
        return;
    }
    
    Index elemIdx = trans.elementIndex();
    const IntegrationPoint& ip = trans.integrationPoint();
    
    Vector3 v = gf_->evalVector(elemIdx, &ip.xi);
    result[0] = v.x();
    result[1] = v.y();
    result[2] = v.z();
}

// =============================================================================
// PWMatrixCoefficient
// =============================================================================

void PWMatrixCoefficient::eval(ElementTransform& trans, Real* result) const {
    Index attr = trans.attribute();
    
    auto it = matrices_.find(attr);
    if (it != matrices_.end()) {
        const auto& data = it->second;
        std::copy(data.begin(), data.end(), result);
    } else {
        // Return zero matrix
        for (int i = 0; i < rows_ * cols_; ++i) {
            result[i] = 0.0;
        }
    }
}

}  // namespace mpfem
