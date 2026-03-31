#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly_change_tracker.hpp"
#include "fe/coefficient.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace mpfem {

/**
 * @brief Heat transfer solver
 * 
 * Solves: -div(k * grad T) = Q
 * k is a 3x3 thermal conductivity tensor (matrix coefficient)
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    std::string fieldName() const override { return "HeatTransfer"; }
    FieldId fieldId() const override { return FieldId::Temperature; }
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialTemperature = 293.15);
    
    // Material bindings
    void setThermalConductivity(const std::set<int>& domains, const MatrixCoefficient* k);
    
    void setHeatSource(const std::set<int>& domains, const Coefficient* Q);
    void setMassProperties(const std::set<int>& domains, const Coefficient* rho, const Coefficient* Cp);
    
    // Mass matrix for transient terms: M = ∫ ρCp φᵢ φⱼ dΩ
    void assembleMassMatrix();
    const SparseMatrix& massMatrix() const { return massMatrix_; }
    bool massMatrixAssembled() const { return massMatrixAssembled_; }
    
    // Boundary conditions
    void addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature);
    void addConvectionBC(const std::set<int>& boundaryIds, const Coefficient* h, const Coefficient* Tinf);
    void clearBoundaryConditions() { temperatureBCs_.clear(); convBCs_.clear(); }
    
    void assemble() override;
    
    /**
     * @brief Solve a custom linear system Ax = b with applied boundary conditions
     * 
     * This is useful for time integrators like BDF1 that need to solve
     * a modified system (e.g., M + dt*K) rather than the standard K.
     * 
     * @param A System matrix
     * @param x Solution vector (output)
     * @param b Right-hand side vector
     * @return true if solved successfully
     */
    bool solveLinearSystem(const SparseMatrix& A, Vector& x, const Vector& b);
    
    /**
     * @brief Get stiffness matrix before BC application
     * 
     * This is needed for transient time integrators like BDF1 that need
     * to form the combined system (M + dt*K) before BCs are applied.
     * 
     * @return Stiffness matrix K (without BC modifications)
     */
    const SparseMatrix& stiffnessMatrixBeforeBC() const { return stiffnessMatrixBeforeBC_; }
    
    /**
     * @brief Get RHS vector before BC application
     * 
     * This is needed for transient time integrators like BDF1 that need
     * to form the combined RHS (M*T_prev + dt*Q) before BCs are applied.
     * 
     * @return RHS vector Q (without BC modifications)
     */
    const Vector& rhsBeforeBC() const { return rhsBeforeBC_; }

private:
    struct ConvBC {
        const Coefficient* h = nullptr;
        const Coefficient* Tinf = nullptr;

        std::uint64_t stiffnessTag() const {
            return stateTagOf(h);
        }

        std::uint64_t loadTag() const {
            return combineTag(stateTagOf(h), stateTagOf(Tinf));
        }

        std::uint64_t stateTag() const {
            return loadTag();
        }
    };

    struct ConductivityBinding {
        std::set<int> domains;
        const MatrixCoefficient* conductivity = nullptr;

        std::uint64_t stateTag() const {
            return combineTag(stateTagOf(domains), stateTagOf(conductivity));
        }
    };
    struct HeatSourceBinding {
        std::set<int> domains;
        const Coefficient* source = nullptr;

        std::uint64_t stateTag() const {
            return combineTag(stateTagOf(domains), stateTagOf(source));
        }
    };
    struct MassBinding {
        std::set<int> domains;
        const Coefficient* density = nullptr;
        const Coefficient* specificHeat = nullptr;
        std::unique_ptr<ProductCoefficient> rhoCp;

        std::uint64_t stateTag() const {
            return combineTag(stateTagOf(domains), stateTagOf(rhoCp.get()));
        }
    };

    std::vector<ConductivityBinding> conductivityBindings_;
    std::vector<HeatSourceBinding> heatSourceBindings_;
    std::vector<MassBinding> massBindings_;
    SparseMatrix massMatrix_;
    bool massMatrixAssembled_ = false;
    AssemblyTagCache massAssemblyState_;
    std::map<int, const Coefficient*> temperatureBCs_;
    std::map<int, ConvBC> convBCs_;
    
    /// @brief Stiffness matrix before BC application (for transient time integrators)
    SparseMatrix stiffnessMatrixBeforeBC_;
    
    /// @brief RHS vector before BC application (for transient time integrators)
    Vector rhsBeforeBC_;

    // Reusable buffers for transient linear systems after BC application.
    SparseMatrix systemMatrix_;
    Vector systemRhs_;

    AssemblyTagCache stiffnessAssemblyState_;
    AssemblyTagCache loadAssemblyState_;
    AssemblyTagCache bcAssemblyState_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP