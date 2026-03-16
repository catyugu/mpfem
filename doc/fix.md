# 大纲

依据@prompt.md ，适当参考学习external/mfem和external/hpc-fem-playground中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast, friend, mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 完成一块工作任务后：
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。
  * 提交一次代码，然后继续下一块工作。

## 工作任务1

* 焦耳热加了单独的系数src/coupling/joule_heating.hpp，而热膨胀则没有，缺乏一致性。请删除它。
* CouplingManager里应该是持有一系列Coefficient，以字符串为索引，并全权负责它们的生命周期。重构后所有所有z的引用就应该基于字符串。
* CouplingManager应该管理所有场的解，耦合系数类中应该持有非拥有的GridFunction引用，而GridFunction在求解单场完成后自动更新，应该不需要手动更新才对。从而存在耦合的系数也不再需要单独特殊处理，而是在创建的时候就天然持有引用并且自动根据当前场值计算。这样就可以移除CouplingManager中大部分接口。
* 这些东西影响代码通用性，因此都应该消失：
```cpp
    // 耦合模块（拥有所有权）
    std::unique_ptr<JouleHeatingCoupling> jouleHeating_;
    std::unique_ptr<TemperatureDependentConductivity> tempDepSigma_;
    std::unique_ptr<PWConstCoefficient> thermalAlphaCoef_;
    
    // 外部材料参数（非拥有）
    const Coefficient* structE_ = nullptr;
    const Coefficient* structNu_ = nullptr;
    
    // 配置参数
    std::set<int> tempDepDomains_;
    std::set<int> jouleHeatDomains_;
    std::map<int, Real> thermalAlpha_;
    Real thermalTref_ = 293.15;
    bool hasTempDepSigma_ = false;
```

* 很多场景下，如我们案例中的材料、耦合源的Coefficient值、边界值等是和域/边界编号有关的，并不是一个物理场只有一个，我认为每一个Coefficient应该对应一组域/边界选择，而不是一个求解器只持有一个或者两个特定的Coefficient。
* 移除通用组合系数 ProductCoefficient、DomainRestrictedCoefficient 等。
* 例如，setConductivity, conductivity等接口应该要指定域选择。如果不指定则默认为全部施加，如果同一个域多次被施加，则新的覆盖旧的。
* 边界条件也应该指定边界选择。并且边界条件的数值也应该是Coefficient而不是直接的数字。换言之，求解器及其接口的任何内容总是不应该有直接的数字。