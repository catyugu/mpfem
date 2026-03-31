#ifndef MPFEM_PROBLEM_HPP
#define MPFEM_PROBLEM_HPP

#include "fe/coefficient.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "physics/field_values.hpp"
#include "core/types.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpfem
{

    class ElectrostaticsSolver;
    class HeatTransferSolver;
    class StructuralSolver;

    class Problem
    {
    protected:
        struct DomainPropertyKey
        {
            std::string property;
            int domainId = -1;

            bool operator==(const DomainPropertyKey &other) const
            {
                return domainId == other.domainId && property == other.property;
            }
        };

        struct DomainPropertyKeyHash
        {
            std::size_t operator()(const DomainPropertyKey &key) const
            {
                const std::size_t h1 = std::hash<std::string>{}(key.property);
                const std::size_t h2 = std::hash<int>{}(key.domainId);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
            }
        };

        std::unordered_map<DomainPropertyKey, std::unique_ptr<Coefficient>, DomainPropertyKeyHash> domainScalarCoefficients;
        std::unordered_map<DomainPropertyKey, std::unique_ptr<MatrixCoefficient>, DomainPropertyKeyHash> domainMatrixCoefficients;
        std::vector<std::unique_ptr<Coefficient>> ownedScalarCoefficients;
        std::vector<std::unique_ptr<VectorCoefficient>> ownedVectorCoefficients;
        std::vector<std::unique_ptr<MatrixCoefficient>> ownedMatrixCoefficients;

    public:
        virtual ~Problem();
        virtual bool isTransient() const { return false; }

        // Physics presence queries
        bool hasElectrostatics() const { return electrostatics != nullptr; }
        bool hasHeatTransfer() const { return heatTransfer != nullptr; }
        bool hasStructural() const { return structural != nullptr; }
        bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
        bool hasThermalExpansion() const { return hasHeatTransfer() && hasStructural(); }
        bool isCoupled() const { return hasJouleHeating() || hasThermalExpansion(); }

        // Coupling parameters for coupled problems
        int couplingMaxIter = 15;
        Real couplingTol = 1e-4;

        std::unique_ptr<ElectrostaticsSolver> electrostatics;
        std::unique_ptr<HeatTransferSolver> heatTransfer;
        std::unique_ptr<StructuralSolver> structural;

        std::string caseName;
        std::unique_ptr<Mesh> mesh;
        MaterialDatabase materials;
        CaseDefinition caseDef;
        FieldValues fieldValues;

        const Coefficient *findDomainScalarCoef(std::string_view property, int domainId) const;
        const MatrixCoefficient *findDomainMatrixCoef(std::string_view property, int domainId) const;
        const Coefficient *setDomainScalarCoef(std::string property,
                                               int domainId,
                                               std::unique_ptr<Coefficient> coef);
        const MatrixCoefficient *setDomainMatrixCoef(std::string property,
                                                     int domainId,
                                                     std::unique_ptr<MatrixCoefficient> coef);

        const Coefficient *ownScalarCoef(std::unique_ptr<Coefficient> coef);
        const VectorCoefficient *ownVectorCoef(std::unique_ptr<VectorCoefficient> coef);
        const MatrixCoefficient *ownMatrixCoef(std::unique_ptr<MatrixCoefficient> coef);
    };

} // namespace mpfem

#endif // MPFEM_PROBLEM_HPP
