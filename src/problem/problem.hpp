#ifndef MPFEM_PROBLEM_HPP
#define MPFEM_PROBLEM_HPP

#include "fe/coefficient.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "physics/field_values.hpp"
#include "core/types.hpp"
#include <map>
#include <string>

namespace mpfem
{

    class ElectrostaticsSolver;
    class HeatTransferSolver;
    class StructuralSolver;

    class Problem
    {
    protected:
        std::map<std::string, std::unique_ptr<Coefficient>> scalarCoefficients;
        std::map<std::string, std::unique_ptr<VectorCoefficient>> vectorCoefficients;
        std::map<std::string, std::unique_ptr<MatrixCoefficient>> matrixCoefficients;

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
        std::map<int, std::string> domainMaterial;
        std::map<int, const MatrixCoefficient*> conductivityByDomain;
        std::map<int, const Coefficient*> youngModulusByDomain;
        std::map<int, const Coefficient*> poissonRatioByDomain;
        FieldValues fieldValues;

        // Scalar coefficients
        const Coefficient *getScalarCoef(const std::string &name) const;
        void setScalarCoef(const std::string &name, std::unique_ptr<Coefficient> coef);

        // Vector coefficients
        const VectorCoefficient *getVectorCoef(const std::string &name) const;
        void setVectorCoef(const std::string &name, std::unique_ptr<VectorCoefficient> coef);

        // Matrix coefficients
        const MatrixCoefficient *getMatrixCoef(const std::string &name) const;
        void setMatrixCoef(const std::string &name, std::unique_ptr<MatrixCoefficient> coef);
    };

} // namespace mpfem

#endif // MPFEM_PROBLEM_HPP
