#include "problem/problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include <utility>

namespace mpfem {

    Problem::~Problem() = default;

    const Coefficient* Problem::findDomainScalarCoef(std::string_view property, int domainId) const
    {
        DomainPropertyKey key {std::string(property), domainId};
        auto it = domainScalarCoefficients.find(key);
        return it != domainScalarCoefficients.end() ? it->second.get() : nullptr;
    }

    const MatrixCoefficient* Problem::findDomainMatrixCoef(std::string_view property, int domainId) const
    {
        DomainPropertyKey key {std::string(property), domainId};
        auto it = domainMatrixCoefficients.find(key);
        return it != domainMatrixCoefficients.end() ? it->second.get() : nullptr;
    }

    const Coefficient* Problem::setDomainScalarCoef(std::string property,
        int domainId,
        std::unique_ptr<Coefficient> coef)
    {
        DomainPropertyKey key {std::move(property), domainId};
        auto [it, _] = domainScalarCoefficients.insert_or_assign(std::move(key), std::move(coef));
        return it->second.get();
    }

    const MatrixCoefficient* Problem::setDomainMatrixCoef(std::string property,
        int domainId,
        std::unique_ptr<MatrixCoefficient> coef)
    {
        DomainPropertyKey key {std::move(property), domainId};
        auto [it, _] = domainMatrixCoefficients.insert_or_assign(std::move(key), std::move(coef));
        return it->second.get();
    }

    const Coefficient* Problem::ownScalarCoef(std::unique_ptr<Coefficient> coef)
    {
        ownedScalarCoefficients.push_back(std::move(coef));
        return ownedScalarCoefficients.back().get();
    }

    const VectorCoefficient* Problem::ownVectorCoef(std::unique_ptr<VectorCoefficient> coef)
    {
        ownedVectorCoefficients.push_back(std::move(coef));
        return ownedVectorCoefficients.back().get();
    }

    const MatrixCoefficient* Problem::ownMatrixCoef(std::unique_ptr<MatrixCoefficient> coef)
    {
        ownedMatrixCoefficients.push_back(std::move(coef));
        return ownedMatrixCoefficients.back().get();
    }

} // namespace mpfem
