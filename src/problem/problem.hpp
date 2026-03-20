#ifndef MPFEM_PROBLEM_HPP
#define MPFEM_PROBLEM_HPP

#include "core/types.hpp"
#include "fe/coefficient.hpp"
#include "mesh/mesh.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "physics/field_values.hpp"
#include <memory>
#include <map>
#include <string>

namespace mpfem {

/**
 * @brief 问题基类 - 数据持有者
 * 
 * 所有问题类型（稳态/瞬态）的公共数据基类。
 * FieldValues 管理 GridFunction 的生命周期。
 */
class Problem {
public:
    virtual ~Problem() = default;
    
    /// 是否为瞬态问题
    virtual bool isTransient() const { return false; }
    
    // =========================================================================
    // 核心数据
    // =========================================================================
    
    std::string caseName;
    std::unique_ptr<Mesh> mesh;
    MaterialDatabase materials;
    CaseDefinition caseDef;
    std::map<int, std::string> domainMaterial;
    
    /// 场值管理器（管理所有 GridFunction 的生命周期）
    FieldValues fieldValues;
    
    /// 所有系数统一存储
    std::map<std::string, AnyCoefficient> coefficients;
    
    // =========================================================================
    // 系数访问方法
    // =========================================================================
    
    template<typename T>
    const T* getCoef(const std::string& name) const {
        auto it = coefficients.find(name);
        return (it != coefficients.end()) ? it->second.get<T>() : nullptr;
    }
    
    template<typename T>
    void setCoef(const std::string& name, std::unique_ptr<T> coef) {
        coefficients[name] = AnyCoefficient(std::move(coef));
    }
};

}  // namespace mpfem

#endif  // MPFEM_PROBLEM_HPP
