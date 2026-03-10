/**
 * @file material_property.hpp
 * @brief Material property definitions supporting field-dependent properties
 */

#ifndef MPFEM_MATERIAL_MATERIAL_PROPERTY_HPP
#define MPFEM_MATERIAL_MATERIAL_PROPERTY_HPP

#include "core/types.hpp"
#include <string>
#include <functional>
#include <memory>

namespace mpfem {

// Forward declaration
class MaterialEvaluator;

/**
 * @brief Base class for material properties
 * 
 * Supports both constant and field-dependent properties.
 * Field-dependent properties can depend on temperature, displacement, etc.
 */
class MaterialPropertyBase {
public:
    virtual ~MaterialPropertyBase() = default;
    
    /**
     * @brief Get property value at a point
     * @param evaluator Material evaluator providing field values
     * @return Property value
     */
    virtual Scalar value(const MaterialEvaluator& evaluator) const = 0;
    
    /**
     * @brief Check if this property depends on any fields
     */
    virtual bool is_field_dependent() const { return false; }
    
    /**
     * @brief Get the name of the field this property depends on
     * @return Field name or empty string if constant
     */
    virtual std::string field_dependency() const { return ""; }
    
    /**
     * @brief Clone this property
     */
    virtual std::unique_ptr<MaterialPropertyBase> clone() const = 0;
};

/**
 * @brief Constant material property
 */
class ConstantProperty : public MaterialPropertyBase {
public:
    explicit ConstantProperty(Scalar value) : value_(value) {}
    
    Scalar value(const MaterialEvaluator& /*evaluator*/) const override {
        return value_;
    }
    
    std::unique_ptr<MaterialPropertyBase> clone() const override {
        return std::make_unique<ConstantProperty>(value_);
    }
    
    Scalar value() const { return value_; }
    
private:
    Scalar value_;
};

/**
 * @brief Field-dependent material property
 * 
 * Property value depends on a field (e.g., temperature).
 * Uses a callback function to compute the value.
 */
class FieldDependentProperty : public MaterialPropertyBase {
public:
    /// Function type for computing property value from field value
    using ValueFunction = std::function<Scalar(Scalar)>;
    
    /**
     * @brief Construct a field-dependent property
     * @param field_name Name of the field this property depends on
     * @param func Function to compute property value from field value
     */
    FieldDependentProperty(const std::string& field_name, ValueFunction func)
        : field_name_(field_name), func_(std::move(func)) {}
    
    Scalar value(const MaterialEvaluator& evaluator) const override;
    
    bool is_field_dependent() const override { return true; }
    
    std::string field_dependency() const override { return field_name_; }
    
    std::unique_ptr<MaterialPropertyBase> clone() const override {
        return std::make_unique<FieldDependentProperty>(field_name_, func_);
    }
    
private:
    std::string field_name_;
    ValueFunction func_;
};

/**
 * @brief Linearized resistivity property
 * 
 * rho(T) = rho0 * (1 + alpha * (T - Tref))
 */
class LinearizedResistivity : public MaterialPropertyBase {
public:
    LinearizedResistivity(Scalar rho0, Scalar alpha, Scalar Tref)
        : rho0_(rho0), alpha_(alpha), Tref_(Tref) {}
    
    Scalar value(const MaterialEvaluator& evaluator) const override;
    
    bool is_field_dependent() const override { return true; }
    
    std::string field_dependency() const override { return "temperature"; }
    
    std::unique_ptr<MaterialPropertyBase> clone() const override {
        return std::make_unique<LinearizedResistivity>(rho0_, alpha_, Tref_);
    }
    
    /// Get resistivity at a given temperature
    Scalar at_temperature(Scalar T) const {
        return rho0_ * (1.0 + alpha_ * (T - Tref_));
    }
    
    /// Get conductivity at a given temperature
    Scalar conductivity_at_temperature(Scalar T) const {
        Scalar rho = at_temperature(T);
        return (rho > 0) ? (1.0 / rho) : 0.0;
    }
    
private:
    Scalar rho0_;   ///< Reference resistivity
    Scalar alpha_;  ///< Temperature coefficient
    Scalar Tref_;   ///< Reference temperature
};

/**
 * @brief Tensor material property (e.g., anisotropic conductivity)
 */
class TensorProperty {
public:
    /// Default constructor - creates zero tensor
    TensorProperty() {
        tensor_.setZero();
    }
    
    /**
     * @brief Construct an isotropic tensor
     */
    explicit TensorProperty(Scalar value) {
        tensor_.setZero();
        for (int i = 0; i < 3; ++i) {
            tensor_(i, i) = value;
        }
    }
    
    /**
     * @brief Construct from 9 values (row-major)
     */
    explicit TensorProperty(const std::vector<Scalar>& values) {
        tensor_.setZero();
        for (size_t i = 0; i < values.size() && i < 9; ++i) {
            tensor_(i / 3, i % 3) = values[i];
        }
    }
    
    /**
     * @brief Construct from Tensor<2,3>
     */
    explicit TensorProperty(const Tensor<2, 3>& t) : tensor_(t) {}
    
    /// Get the tensor
    const Tensor<2, 3>& tensor() const { return tensor_; }
    
    /// Get a component
    Scalar operator()(int i, int j) const { return tensor_(i, j); }
    
    /// Check if diagonal (isotropic or orthotropic)
    bool is_diagonal() const {
        return tensor_(0, 1) == 0 && tensor_(0, 2) == 0 &&
               tensor_(1, 0) == 0 && tensor_(1, 2) == 0 &&
               tensor_(2, 0) == 0 && tensor_(2, 1) == 0;
    }
    
    /// Check if isotropic (diagonal with equal values)
    bool is_isotropic() const {
        return is_diagonal() &&
               tensor_(0, 0) == tensor_(1, 1) &&
               tensor_(1, 1) == tensor_(2, 2);
    }
    
    /// Get isotropic value (if isotropic)
    Scalar isotropic_value() const {
        if (is_isotropic()) return tensor_(0, 0);
        return 0.0;
    }
    
private:
    Tensor<2, 3> tensor_;
};

/**
 * @brief Material property that can be scalar or tensor
 */
class Property {
public:
    enum class Type {
        Scalar,
        Tensor
    };
    
    Property() : type_(Type::Scalar), scalar_prop_(std::make_unique<ConstantProperty>(0.0)) {}
    
    /// Construct from scalar property
    explicit Property(std::unique_ptr<MaterialPropertyBase> prop)
        : type_(Type::Scalar), scalar_prop_(std::move(prop)) {}
    
    /// Construct from constant scalar
    explicit Property(Scalar value)
        : type_(Type::Scalar), scalar_prop_(std::make_unique<ConstantProperty>(value)) {}
    
    /// Construct from tensor
    explicit Property(const TensorProperty& tensor)
        : type_(Type::Tensor), tensor_prop_(tensor) {}
    
    /// Construct from tensor values
    explicit Property(const std::vector<Scalar>& values)
        : type_(Type::Tensor), tensor_prop_(values) {}
    
    // Move constructor
    Property(Property&& other) noexcept
        : type_(other.type_), scalar_prop_(std::move(other.scalar_prop_)), 
          tensor_prop_(std::move(other.tensor_prop_)) {}
    
    // Move assignment
    Property& operator=(Property&& other) noexcept {
        if (this != &other) {
            type_ = other.type_;
            scalar_prop_ = std::move(other.scalar_prop_);
            tensor_prop_ = std::move(other.tensor_prop_);
        }
        return *this;
    }
    
    // Delete copy operations (unique_ptr is not copyable)
    Property(const Property&) = delete;
    Property& operator=(const Property&) = delete;
    
    Type type() const { return type_; }
    
    bool is_scalar() const { return type_ == Type::Scalar; }
    bool is_tensor() const { return type_ == Type::Tensor; }
    
    /// Get scalar value
    Scalar scalar_value(const MaterialEvaluator& evaluator) const {
        if (type_ == Type::Scalar && scalar_prop_) {
            return scalar_prop_->value(evaluator);
        }
        return 0.0;
    }
    
    /// Get tensor value
    const Tensor<2, 3>& tensor_value() const {
        return tensor_prop_.tensor();
    }
    
    /// Get tensor property
    const TensorProperty& tensor_property() const {
        return tensor_prop_;
    }
    
    /// Check if field-dependent
    bool is_field_dependent() const {
        return type_ == Type::Scalar && scalar_prop_ && scalar_prop_->is_field_dependent();
    }
    
    /// Get field dependency name
    std::string field_dependency() const {
        if (type_ == Type::Scalar && scalar_prop_) {
            return scalar_prop_->field_dependency();
        }
        return "";
    }
    
private:
    Type type_;
    std::unique_ptr<MaterialPropertyBase> scalar_prop_;
    TensorProperty tensor_prop_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_MATERIAL_PROPERTY_HPP
