# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 删除冗余的成员变量、接口等。
* 尽量使用pimpl模式最小化编译依赖与交叉耦合。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

通过对代码的分析，我发现该有限元库（MPFEM）在架构上存在几个较为明显的**设计反模式（Anti-patterns）**。这些问题主要集中在**违反开闭原则（OCP）**、**类型嗅探（RTTI 滥用）**、**时间耦合（Temporal Coupling）**以及**数据结构内聚性差**等方面。

以下是破坏式重构（Breaking Changes）的建议，旨在用最少的不必要复杂度提升代码的健壮性和可扩展性。

### 1. 违反开闭原则的“上帝类” (God Object)
**文件**: `src/problem/problem.hpp`
**现象**: `Problem` 类硬编码了所有具体的物理场求解器（`ElectrostaticsSolver`, `HeatTransferSolver`, `StructuralSolver`），并包含大量的 `hasElectrostatics()`、`hasJouleHeating()` 等硬编码判断。
**反模式**: 典型的 **God Object** 和 **OCP 违背**。每当想要引入一种新的物理场（例如流体动力学、电磁波），都必须修改 `Problem` 类的核心头文件。

**破坏式重构建议**:
将具体的物理场指针替换为基于接口的通用容器。依赖 `PhysicsFieldSolver` 基类，并通过 `FieldId` 或字符串名称进行管理。

```cpp
// 重构后的 Problem 类
class Problem {
public:
    // ...
    // 统一管理所有的物理场求解器
    void addSolver(std::unique_ptr<PhysicsFieldSolver> solver);
    PhysicsFieldSolver* getSolver(const std::string& fieldName);
    const PhysicsFieldSolver* getSolver(const std::string& fieldName) const;

    // 耦合关系不再硬编码，而是通过检查 CaseDefinition 中的 CoupledPhysicsDefinition 动态决定
    bool isCoupled() const { return !caseDef.coupledPhysicsDefinitions.empty(); }

private:
    // 取代硬编码的 unique_ptr<ElectrostaticsSolver> 等
    std::unordered_map<std::string, std::unique_ptr<PhysicsFieldSolver>> solvers_;
};
```

---

### 2. 工厂模式中的向下转型与 RTTI 滥用
**文件**: `src/solver/solver_factory.hpp`
**现象**: `OperatorFactory::applyParameters` 函数中使用了大量的 `dynamic_cast<CgOperator*>(op)` 来判断对象类型，从而调用 `set_max_iterations()` 等专属方法。
**反模式**: **Switch 语句嗅探 / 类型查询 (Type Querying)**。工厂类不仅要知道如何创建对象，还要清楚每个子类的实现细节和参数接口。一旦添加新的求解器（如 BiCGSTAB），必须回头修改这个工厂方法。

**破坏式重构建议**:
将参数配置的职责下放给具体的 `LinearOperator` 子类（或其 Config 结构），在基类中引入通用的参数配置接口。

```cpp
// 1. 在 LinearOperator (基类) 中增加虚函数
class LinearOperator {
public:
    // ...
    virtual void configure(const std::map<std::string, double>& params) {} 
    // 或者使用 std::any 以支持非 double 类型的参数
};

// 2. 在具体的子类中实现 (例如 CgOperator)
void CgOperator::configure(const std::map<std::string, double>& params) override {
    if (auto it = params.find("MaxIterations"); it != params.end()) {
        this->set_max_iterations(static_cast<int>(it->second));
    }
    // ...
}

// 3. 工厂代码变得极其纯粹（甚至不再需要单独的 applyParameters 函数）
static std::unique_ptr<LinearOperator> create(const LinearOperatorConfig& config) {
    std::unique_ptr<LinearOperator> op = createByType(config.type);
    
    // 多态调用，工厂无需知道具体类型
    op->configure(config.parameters);
    
    if (config.preconditioner) {
        op->set_preconditioner(create(*config.preconditioner));
    }
    return op;
}
```

---

### 3. 危险的时间耦合 (Temporal Coupling)
**文件**: `src/model/case_definition.hpp`
**现象**: `CaseDefinition` 同时维护了 `std::vector<VariableEntry> variables` 和 `std::map<std::string, double> variableMap_`。注释中明确要求：“*Call this (buildVariableMap) after variables are populated or modified.*”
**反模式**: **双重数据源验证缺失 / 时间耦合**。调用方在修改 `variables` 后，如果忘记调用 `buildVariableMap()`，系统将读取到过期或错误的数据（且不会有任何编译期报错）。

**破坏式重构建议**:
将变量管理的职责完全封装，对外不暴露裸露的 `vector` 或 `map`，或者废弃冗余结构。如果不需要保持 XML 解析的原始顺序，直接废弃 `vector`。如果需要顺序，就用专门的类封装它们。

```cpp
// 重构：直接隐藏数据成员，强制通过接口操作
class CaseDefinition {
public:
    // 添加变量时同步更新 map 和 vector
    void addVariable(const std::string& name, const std::string& text, double val) {
        variables_.push_back({name, text, val});
        variableMap_[name] = val;
    }

    double getVariable(const std::string& name) const { /* ... */ }

private:
    std::vector<VariableEntry> variables_;
    std::unordered_map<std::string, double> variableMap_;
};
```

---

### 4. 平行数组导致的高内聚缺失
**文件**: `src/assembly/assembler.hpp` (在 `BilinearFormAssembler` 和 `LinearFormAssembler` 中)
**现象**: 
```cpp
std::vector<std::unique_ptr<DomainBilinearIntegratorBase>> domainIntegs_;
std::vector<std::vector<int>> domainSets_;
```
**反模式**: **平行数组 (Parallel Arrays)**。积分器（Integrator）和它适用的域（Domain sets）在逻辑上是强绑定的，但却被分拆存储在两个并行的 `vector` 中。这降低了缓存局部性，增加了错位（Misalignment）的风险。

**破坏式重构建议**:
创建一个轻量级的结构体将它们绑定在一起。

```cpp
// 在 BilinearFormAssembler 内部：
struct DomainIntegratorEntry {
    std::unique_ptr<DomainBilinearIntegratorBase> integrator;
    std::vector<int> domains;
};

// 取代平行数组
std::vector<DomainIntegratorEntry> domainIntegrators_;
```

---

### 5. 掩盖错误的“静默失败” (Silent Failures)
**文件**: `src/physics/physics_field_solver.hpp`
**现象**: 
```cpp
bool solve() {
    if (!solver_ || !matAsm_ || !vecAsm_ || !fieldValues_)
        return false;
    // ...
}
```
**反模式**: 核心数学/物理求解管线不应该**静默失败**。如果这些核心组件为空，说明在 Setup 阶段出现了严重的生命周期错误或逻辑断裂。返回 `false` 并不能让上层排查到“究竟是哪个组件为 Null”。

**破坏式重构建议**:
遵循“快速失败（Fail Fast）”原则。既然依赖缺失无法求解，应该直接抛出异常或触发断言。

```cpp
void solve() { // 甚至不需要返回 bool，除非求解不收敛需要返回状态
    MPFEM_ASSERT(solver_ != nullptr, "Solver not initialized prior to solve()");
    MPFEM_ASSERT(matAsm_ != nullptr, "Matrix Assembler missing");
    // ... 
    
    // 如果求解器发散（迭代不收敛），这才是应该被返回/抛出的错误
    solver_->apply(vecAsm_->vector(), field().values());
}
```