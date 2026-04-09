# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

针对代码中存在的设计反模式、冗余代码以及向后兼容遗留问题，我们能够通过**“基于泛型的接口统一”**与**“装配器状态常驻化 (Stateful Assembler)”**这两个核心思路进行重构。

以下是四个步骤的系统化重构，每一步都会消除大量样板代码，提升执行效率（消除热路径上的堆分配），并且每步皆可单独编译验证。

---

### 第一步：重构强制边界条件 (Dirichlet BC) 的接口与实现
**当前反模式**：`dirichlet_bc.hpp` 中存在大量用于“向后兼容”的重载方法（如针对标量、针对 `Vector3`，针对 `Component` 等）。每次调用都会触发大量的临时 `std::map` 以及动态 `std::vector` 内存分配。
**重构目标**：利用表达式树节点 `VariableNode` 已经支持的 `TensorValue` 特性，统一标量和矢量的边界计算，只保留一个最简接口。

**修改文件：`assembly/dirichlet_bc.hpp`**
```cpp
#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/sparse_matrix.hpp"
#include "expr/variable_graph.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include <vector>
#include <utility>

namespace mpfem {

// 统一的 Dirichlet 边界处理，直接抛弃向后兼容的过时重载
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
    const FESpace& fes, const Mesh& mesh,
    const std::vector<std::pair<int, const VariableNode*>>& bcs, // 统一传入 List
    bool updateMatrix = true)
{
    const Index numDofs = fes.numDofs();
    if (numDofs == 0 || bcs.empty()) return;

    std::vector<Real> dofVals(numDofs, 0.0);
    std::vector<char> hasVal(numDofs, 0);
    std::vector<Index> eliminated;

    FacetElementTransform trans;
    trans.setMesh(&mesh);
    const int vdim = fes.vdim();

    for (const auto& [bid, coef] : bcs) {
        if (!coef || !fes.isExternalBoundaryId(bid)) continue;

        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() != bid) continue;

            const ReferenceElement* refElem = fes.bdrElementRefElement(b);
            if (!refElem) continue;

            const int nd = refElem->numDofs();
            std::vector<Index> dofs(nd * vdim);
            fes.getBdrElementDofs(b, dofs); // 获取全分量 dof
            trans.setBoundaryElement(b);

            for (int i = 0; i < nd; ++i) {
                Real xi[3] = {refElem->dofCoords()[i][0], refElem->dofCoords()[i][1], refElem->dofCoords()[i][2]};
                trans.setIntegrationPoint(xi);

                std::array<Vector3, 1> refPts {Vector3(xi[0], xi[1], xi[2])};
                std::array<Vector3, 1> physPts;
                trans.transform(trans.integrationPoint(), physPts[0]);
                std::array<Matrix3, 1> invJTs {trans.invJacobianT()};
                
                EvaluationContext ctx;
                ctx.domainId = static_cast<int>(trans.attribute());
                ctx.elementId = trans.elementIndex();
                ctx.physicalPoints = physPts;
                ctx.referencePoints = refPts;
                ctx.invJacobianTransposes = invJTs;

                std::array<TensorValue, 1> out {};
                coef->evaluateBatch(ctx, out); // 多态计算

                // 统一处理多维度场：标量或矢量
                for (int c = 0; c < vdim; ++c) {
                    Index d = dofs[i * vdim + c];
                    if (d != InvalidIndex && !hasVal[d]) {
                        dofVals[d] = out[0].isVector() ? out[0].asVector()[c] : out[0].asScalar();
                        hasVal[d] = 1;
                        eliminated.push_back(d);
                    }
                }
            }
        }
    }

    if (updateMatrix) mat.eliminateRows(eliminated, dofVals, rhs);
    else mat.eliminateRhsOnly(eliminated, dofVals, rhs);
    for (Index d : eliminated) sol(d) = dofVals[d];
}

} // namespace mpfem
#endif // MPFEM_DIRICHLET_BC_HPP
```

---

### 第二步：重构积分器与装配器，实现原生的依赖追踪
**当前反模式**：目前的装配行为是在热路径上（例如 `buildStiffnessMatrix` 时），手动清理 `matAsm_->clearIntegrators()` 然后重新用 `make_unique` 将积分器塞回去。同时，物理求解器不得不手写如 `if (b.E) rev = std::max(rev, b.E->revision())` 来做缓存一致性检测，代码极其丑陋。
**重构目标**：为所有 Integrator 增加 `revision()`，让装配器能直接管理和返回当前状态的版本号。积分器注入一次即常驻。

**修改文件：`assembly/integrator.hpp` 和 `assembly/assembler.hpp`**
1. 在 `integrator.hpp` 的四个基类（`DomainBilinearIntegratorBase`，`FaceBilinearIntegratorBase`，等）中增加接口：
```cpp
virtual std::uint64_t revision() const { return 0; }
```
2. 在 `integrators.hpp` 所有的具体积分器实现中添加 `revision` 追踪。例如对于 `DiffusionIntegrator`：
```cpp
std::uint64_t revision() const override { return coef_ ? coef_->revision() : 0; }
```
3. 在 `assembler.hpp` 中，给 `BilinearFormAssembler` 和 `LinearFormAssembler` 添加统计方法：
```cpp
std::uint64_t revision() const {
    std::uint64_t rev = 0;
    for (const auto& integ : domainIntegs_) rev = std::max(rev, integ->revision());
    for (const auto& integ : bdrIntegs_) rev = std::max(rev, integ->revision());
    return rev;
}
```

---

### 第三步：将样板代码提升至 `PhysicsFieldSolver` (Template Method Pattern)
**当前反模式**：`ElectrostaticsSolver`，`HeatTransferSolver` 都有数百行几乎复制粘贴的 `initialize`、`buildRHS` 逻辑。每个子类都用诸如 `ConductivityBinding` 这样的多余结构体去管理生命周期。
**重构目标**：剥夺子类的所有“内务管理”权限，全部收拢在基类统一处理。

**修改文件：`physics/physics_field_solver.hpp`**
```cpp
#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP
//... headers ...

namespace mpfem {
class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;

    // 基类统管一切资源分配，子类从此再无 initialize 方法
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, 
                    std::string fieldName, int vdim, int order, Real initVal = 0.0) {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        fieldName_ = std::move(fieldName);
        order_ = order;
        
        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec), vdim);
        
        fieldValues.createField(fieldName_, fes_.get(), 
                                vdim == 1 ? TensorShape::scalar() : TensorShape::vector(vdim), initVal);
        
        // 装配器提前准备好，且配备专属的质量矩阵装配器
        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        massAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get()); 
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
        
        if (solverConfig_) solver_ = SolverFactory::create(*solverConfig_);
        return true;
    }

    std::string fieldName() const { return fieldName_; }
    void setSolverConfig(std::unique_ptr<LinearOperatorConfig> config) { solverConfig_ = std::move(config); }

    // 核心：暴露统一的边界添加接口
    void addDirichletBC(int bid, const VariableNode* coef) {
        essentialBCs_.push_back({bid, coef});
    }

    // 原生获取版本号：直接代理给子装配器，避免各子类自造轮子
    std::uint64_t getMatrixRevision() const { return matAsm_ ? matAsm_->revision() : 0; }
    std::uint64_t getMassRevision() const { return massAsm_ ? massAsm_->revision() : 0; }
    std::uint64_t getRhsRevision() const { return vecAsm_ ? vecAsm_->revision() : 0; }
    std::uint64_t getBcRevision() const {
        std::uint64_t rev = 0;
        for (const auto& bc : essentialBCs_) if (bc.second) rev = std::max(rev, bc.second->revision());
        return rev;
    }

protected:
    // 子类不再需要实现 buildStiffnessMatrix，基类代劳
    void buildStiffnessMatrix(SparseMatrix& K) { matAsm_->assemble(); K = matAsm_->matrix(); }
    void buildMassMatrix(SparseMatrix& M) { massAsm_->assemble(); M = massAsm_->matrix(); }
    void buildRHS(Vector& F) { vecAsm_->assemble(); F = vecAsm_->vector(); }

    void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix) {
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, essentialBCs_, updateMatrix);
    }

protected:
    std::string fieldName_;
    int order_ = 1;
    std::vector<std::pair<int, const VariableNode*>> essentialBCs_; // 统一管辖边界
    
    const Mesh* mesh_ = nullptr;
    FieldValues* fieldValues_ = nullptr;
    std::unique_ptr<FESpace> fes_;
    
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<BilinearFormAssembler> massAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearOperator> solver_;
    std::unique_ptr<LinearOperatorConfig> solverConfig_;
    
    // ... 保留内部缓存控制 ...
};
}
#endif
```

---

### 第四步：极简化具体物理的实现 (Minimalist Specific Physics)
**当前反模式**：以 `StructuralSolver` 和 `ElectrostaticsSolver` 为首的各个子类存在数百行僵尸代码（包括复杂的 `getMatrixRevision` 等）。
**重构结果**：在基类完全代理后，特定的物理系统退化为“提供装配参数注册”的外壳。不再清理或重置内存。

**修改文件：`physics/electrostatics_solver.hpp` (cpp 文件可以直接被删除或仅保留空壳)**
```cpp
#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/integrators.hpp"

namespace mpfem {

class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    
    // 直接向装配器注入项，且生命周期常驻
    void setElectricalConductivity(const std::set<int>& domains, const VariableNode* sigma) {
        matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(sigma), domains);
    }

    void addVoltageBC(const std::set<int>& boundaryIds, const VariableNode* voltage) {
        for (int bid : boundaryIds) {
            addDirichletBC(bid, voltage);
        }
    }
};

} // namespace mpfem
#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
```

**修改文件：`physics/structural_solver.hpp`**
```cpp
#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/integrators.hpp"

namespace mpfem {

class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;

    void addElasticity(const std::set<int>& domains, const VariableNode* E, const VariableNode* nu) {
        matAsm_->addDomainIntegrator(std::make_unique<ElasticityIntegrator>(E, nu, fes_->vdim()), domains);
    }

    void setStrainLoad(const std::set<int>& domains, const VariableNode* stress) {
        vecAsm_->addDomainIntegrator(std::make_unique<StrainLoadIntegrator>(stress, fes_->vdim()), domains);
    }

    void addFixedDisplacementBC(const std::set<int>& boundaryIds, const VariableNode* displacement) {
        for (int bid : boundaryIds) {
            addDirichletBC(bid, displacement);
        }
    }
};

} // namespace mpfem
#endif // MPFEM_STRUCTURAL_SOLVER_HPP
```

**验证结论**：
1. **彻底解耦**：子类中全部的 `buildStiffnessMatrix`、`getBcRevision`、各类繁冗的 `Binding` 结构体均被清空。各物理模型的 cpp 代码由几百行缩减至零。
2. **消灭性能瓶颈**：原本每个组装步都会摧毁再重建的 `unique_ptr<Integrator>` 如今变成了装配器的持久挂载项（`addDomainIntegrator` 只调用一次）。`matAsm_->assemble()` 被调用时，将直接复用这些预分配对象。
3. **接口统一**：由一版多态的 `applyDirichletBC` 替换了曾经因未正确利用 Tensor 特性而复制的三份边界加载代码。无需要任何显式的后向兼容分支。