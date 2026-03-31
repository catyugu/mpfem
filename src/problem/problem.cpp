#include "problem/problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"

namespace mpfem {

Problem::~Problem() = default;


    // Scalar coefficients
    const Coefficient* Problem::getScalarCoef(const std::string& name) const {
        auto it = scalarCoefficients.find(name);
        return it != scalarCoefficients.end() ? it->second.get() : nullptr;
    }
    void Problem::setScalarCoef(const std::string& name, std::unique_ptr<Coefficient> coef) {
        scalarCoefficients[name] = std::move(coef);
    }
    
    // Vector coefficients
    const VectorCoefficient* Problem::getVectorCoef(const std::string& name) const {
        auto it = vectorCoefficients.find(name);
        return it != vectorCoefficients.end() ? it->second.get() : nullptr;
    }
    void Problem::setVectorCoef(const std::string& name, std::unique_ptr<VectorCoefficient> coef) {
        vectorCoefficients[name] = std::move(coef);
    }
    
    // Matrix coefficients
    const MatrixCoefficient* Problem::getMatrixCoef(const std::string& name) const {
        auto it = matrixCoefficients.find(name);
        return it != matrixCoefficients.end() ? it->second.get() : nullptr;
    }
    void Problem::setMatrixCoef(const std::string& name, std::unique_ptr<MatrixCoefficient> coef) {
        matrixCoefficients[name] = std::move(coef);
    }

} // namespace mpfem
