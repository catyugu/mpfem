#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"
#include <cmath>

namespace mpfem {

/**
 * @file integrators.hpp
 * @brief Concrete implementations of common integrators.
 */

// =============================================================================
// Diffusion Integrator
// =============================================================================

/**
 * @brief Diffusion (stiffness) integrator.
 * 
 * Computes the element matrix for the diffusion term:
 *   A_ij = integral( k * grad(phi_i) . grad(phi_j) ) dV
 * 
 * Where k is the diffusion coefficient (e.g., thermal conductivity,
 * electrical conductivity).
 * 
 * This is the standard Laplacian matrix for Poisson-type equations:
 *   -div(k * grad(u)) = f
 */
class DiffusionIntegrator : public BilinearFormIntegrator {
public:
    DiffusionIntegrator() = default;
    explicit DiffusionIntegrator(Coefficient* q) : BilinearFormIntegrator(q) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "DiffusionIntegrator"; }
};

// =============================================================================
// Mass Integrator
// =============================================================================

/**
 * @brief Mass matrix integrator.
 * 
 * Computes the element matrix for the mass term:
 *   M_ij = integral( rho * phi_i * phi_j ) dV
 * 
 * Where rho is the density or mass coefficient.
 * 
 * This appears in:
 * - Time-dependent problems: rho * du/dt
 * - Eigenvalue problems: A*x = lambda*M*x
 */
class MassIntegrator : public BilinearFormIntegrator {
public:
    MassIntegrator() = default;
    explicit MassIntegrator(Coefficient* q) : BilinearFormIntegrator(q) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "MassIntegrator"; }
};

// =============================================================================
// Convection Integrator
// =============================================================================

/**
 * @brief Convection integrator for advection problems.
 * 
 * Computes the element matrix for the convection term:
 *   C_ij = integral( (b . grad(phi_j)) * phi_i ) dV
 * 
 * Where b is the velocity field (convection coefficient).
 * 
 * This appears in advection-diffusion equations:
 *   b . grad(u) - div(k * grad(u)) = f
 */
class ConvectionIntegrator : public BilinearFormIntegrator {
public:
    ConvectionIntegrator() = default;
    explicit ConvectionIntegrator(VectorCoefficient* b) : velocity_(b) {}
    
    void setVelocity(VectorCoefficient* b) { velocity_ = b; }
    VectorCoefficient* velocity() const { return velocity_; }
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "ConvectionIntegrator"; }
    
private:
    VectorCoefficient* velocity_ = nullptr;
};

// =============================================================================
// Domain Load Integrator
// =============================================================================

/**
 * @brief Domain load (source) integrator.
 * 
 * Computes the element vector for source terms:
 *   b_i = integral( f * phi_i ) dV
 * 
 * Where f is the source term (e.g., heat source, current density).
 */
class DomainLFIntegrator : public LinearFormIntegrator {
public:
    DomainLFIntegrator() = default;
    explicit DomainLFIntegrator(Coefficient* f) : LinearFormIntegrator(f) {}
    
    void assembleElementVector(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Vector& elvec) const override;
    
    const char* name() const override { return "DomainLFIntegrator"; }
};

// =============================================================================
// Boundary Load Integrator
// =============================================================================

/**
 * @brief Boundary load (Neumann BC) integrator.
 * 
 * Computes the element vector for Neumann boundary conditions:
 *   b_i = integral( g * phi_i ) dS
 * 
 * Where g is the boundary flux.
 */
class BoundaryLFIntegrator : public LinearFormIntegrator {
public:
    BoundaryLFIntegrator() = default;
    explicit BoundaryLFIntegrator(Coefficient* g) : LinearFormIntegrator(g) {}
    
    void assembleFaceVector(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
    
    const char* name() const override { return "BoundaryLFIntegrator"; }
};

// =============================================================================
// Convection Boundary Integrator
// =============================================================================

/**
 * @brief Robin/Convection boundary condition integrator.
 * 
 * For heat transfer, this implements:
 *   -k * dT/dn = h * (T - T_inf)
 * 
 * The weak form adds two terms:
 *   Boundary mass: M_bnd_ij = integral( h * phi_i * phi_j ) dS
 *   Boundary load: b_i = integral( h * T_inf * phi_i ) dS
 */
class ConvectionBoundaryIntegrator : public BilinearFormIntegrator {
public:
    ConvectionBoundaryIntegrator() = default;
    
    /// Construct with heat transfer coefficient
    ConvectionBoundaryIntegrator(Coefficient* h, Coefficient* Tinf = nullptr)
        : BilinearFormIntegrator(h), Tinf_(Tinf) {}
    
    void setAmbientTemperature(Coefficient* Tinf) { Tinf_ = Tinf; }
    Coefficient* ambientTemperature() const { return Tinf_; }
    
    void assembleFaceMatrix(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
    
    /**
     * @brief Compute the boundary load vector for convection BC.
     */
    void assembleFaceVector(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Vector& elvec) const;
    
    const char* name() const override { return "ConvectionBoundaryIntegrator"; }
    
private:
    Coefficient* Tinf_ = nullptr;  ///< Ambient temperature
};

// =============================================================================
// Vector Diffusion Integrator (for displacement field)
// =============================================================================

/**
 * @brief Vector diffusion integrator for elasticity.
 * 
 * Computes the stiffness matrix for linear elasticity:
 *   K_ij = integral( sigma(phi_i) : epsilon(phi_j) ) dV
 * 
 * where sigma is the stress tensor and epsilon is the strain tensor.
 * For isotropic materials:
 *   sigma = 2*mu*epsilon + lambda*tr(epsilon)*I
 */
class VectorDiffusionIntegrator : public VectorBilinearFormIntegrator {
public:
    VectorDiffusionIntegrator() = default;
    
    /// Construct with Lame parameters
    VectorDiffusionIntegrator(Coefficient* lambda, Coefficient* mu)
        : lambda_(lambda), mu_(mu) {}
    
    void setLameParameters(Coefficient* lambda, Coefficient* mu) {
        lambda_ = lambda;
        mu_ = mu;
    }
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               int vdim,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "VectorDiffusionIntegrator"; }
    
private:
    Coefficient* lambda_ = nullptr;  ///< First Lame parameter
    Coefficient* mu_ = nullptr;      ///< Second Lame parameter (shear modulus)
};

// =============================================================================
// Vector Mass Integrator
// =============================================================================

/**
 * @brief Vector mass matrix integrator.
 * 
 * Computes the mass matrix for vector fields:
 *   M_ij = integral( rho * phi_i * phi_j ) dV * I_{3x3}
 * 
 * For displacement dynamics:
 *   rho * d^2u/dt^2 = div(sigma) + f
 */
class VectorMassIntegrator : public VectorBilinearFormIntegrator {
public:
    VectorMassIntegrator() = default;
    explicit VectorMassIntegrator(Coefficient* rho) 
        : VectorBilinearFormIntegrator(rho) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               int vdim,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "VectorMassIntegrator"; }
};

// =============================================================================
// Vector Load Integrator
// =============================================================================

/**
 * @brief Vector load integrator for body forces.
 * 
 * Computes the element vector for body forces:
 *   b_i = integral( f * phi_i ) dV
 * 
 * Where f is the body force vector (e.g., gravity).
 */
class VectorDomainLFIntegrator : public VectorLinearFormIntegrator {
public:
    VectorDomainLFIntegrator() = default;
    explicit VectorDomainLFIntegrator(VectorCoefficient* f) : force_(f) {}
    
    void setForce(VectorCoefficient* f) { force_ = f; }
    
    void assembleElementVector(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               int vdim,
                               Vector& elvec) const override;
    
    const char* name() const override { return "VectorDomainLFIntegrator"; }
    
private:
    VectorCoefficient* force_ = nullptr;
};

// =============================================================================
// Thermal Stress Load Integrator
// =============================================================================

/**
 * @brief Thermal stress load integrator.
 * 
 * Computes the load vector due to thermal expansion:
 *   b_i = integral( alpha * E * (T - T_ref) * div(phi_i) ) dV
 * 
 * This is for computing thermal stresses in coupled thermal-structural analysis.
 */
class ThermalStressLoadIntegrator : public VectorLinearFormIntegrator {
public:
    ThermalStressLoadIntegrator() = default;
    
    void setParameters(Coefficient* alpha, Coefficient* E, Coefficient* nu) {
        alpha_ = alpha;  // Thermal expansion coefficient
        E_ = E;          // Young's modulus
        nu_ = nu;        // Poisson's ratio
    }
    
    void setTemperatureField(const GridFunction* T, Real Tref = 293.15) {
        temperature_ = T;
        Tref_ = Tref;
    }
    
    void assembleElementVector(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               int vdim,
                               Vector& elvec) const override;
    
    const char* name() const override { return "ThermalStressLoadIntegrator"; }
    
private:
    Coefficient* alpha_ = nullptr;
    Coefficient* E_ = nullptr;
    Coefficient* nu_ = nullptr;
    const GridFunction* temperature_ = nullptr;
    Real Tref_ = 293.15;
};

// =============================================================================
// Joule Heat Source Integrator
// =============================================================================

/**
 * @brief Joule heat source integrator.
 * 
 * Computes the volumetric heat source from electrical current:
 *   Q = sigma * |E|^2 = sigma * |grad(V)|^2
 * 
 * This is used for electro-thermal coupling in Joule heating problems.
 */
class JouleHeatIntegrator : public LinearFormIntegrator {
public:
    JouleHeatIntegrator() = default;
    
    void setConductivity(Coefficient* sigma) { sigma_ = sigma; }
    void setVoltageField(const GridFunction* V) { voltage_ = V; }
    
    void assembleElementVector(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Vector& elvec) const override;
    
    const char* name() const override { return "JouleHeatIntegrator"; }
    
private:
    Coefficient* sigma_ = nullptr;
    const GridFunction* voltage_ = nullptr;
};

// =============================================================================
// Inline Implementations
// =============================================================================

inline void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                        ElementTransform& trans,
                                                        Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int dim = refElem.dim();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function gradients in reference coordinates
        ShapeValues sv = refElem.evalShape(ip);
        
        // Transform gradients to physical coordinates
        std::vector<Vector3> physGrad(nd);
        for (int i = 0; i < nd; ++i) {
            trans.transformGradient(sv.gradients[i], physGrad[i]);
        }
        
        // Get coefficient value at this point
        Real coeff = q_ ? q_->eval(trans) : 1.0;
        
        // Add to element matrix
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                Real dot = 0.0;
                for (int d = 0; d < dim; ++d) {
                    dot += physGrad[i][d] * physGrad[j][d];
                }
                elmat(i, j) += w * coeff * dot;
            }
        }
    }
}

inline void MassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                   ElementTransform& trans,
                                                   Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get coefficient value
        Real coeff = q_ ? q_->eval(trans) : 1.0;
        
        // Add to element matrix
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                elmat(i, j) += w * coeff * phi[i] * phi[j];
            }
        }
    }
}

inline void DomainLFIntegrator::assembleElementVector(const ReferenceElement& refElem,
                                                       ElementTransform& trans,
                                                       Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get source value
        Real f = q_ ? q_->eval(trans) : 1.0;
        
        // Add to element vector
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * f * phi[i];
        }
    }
}

inline void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                      FacetElementTransform& trans,
                                                      Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get boundary flux value
        Real g = q_ ? q_->eval(trans) : 1.0;
        
        // Add to element vector
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * g * phi[i];
        }
    }
}

inline void ConvectionBoundaryIntegrator::assembleFaceMatrix(const ReferenceElement& refElem,
                                                              FacetElementTransform& trans,
                                                              Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get heat transfer coefficient
        Real h = q_ ? q_->eval(trans) : 1.0;
        
        // Add to boundary mass matrix
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                elmat(i, j) += w * h * phi[i] * phi[j];
            }
        }
    }
}

inline void ConvectionBoundaryIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                              FacetElementTransform& trans,
                                                              Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    if (!Tinf_) return;
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get h * T_inf
        Real h = q_ ? q_->eval(trans) : 1.0;
        Real Tinf = Tinf_->eval(trans);
        
        // Add to element vector
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * h * Tinf * phi[i];
        }
    }
}

inline void VectorMassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                         ElementTransform& trans,
                                                         int vdim,
                                                         Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nvd = nd * vdim;
    
    elmat.setZero(nvd, nvd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        // Get shape function values
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        // Get density
        Real rho = q_ ? q_->eval(trans) : 1.0;
        
        // Add to element matrix (block diagonal structure)
        for (int c = 0; c < vdim; ++c) {
            for (int i = 0; i < nd; ++i) {
                for (int j = 0; j < nd; ++j) {
                    elmat(c * nd + i, c * nd + j) += w * rho * phi[i] * phi[j];
                }
            }
        }
    }
}

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP
