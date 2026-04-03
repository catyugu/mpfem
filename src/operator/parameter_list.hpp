#ifndef MPFEM_PARAMETER_LIST_HPP
#define MPFEM_PARAMETER_LIST_HPP

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <variant>

namespace mpfem {

    /**
     * @brief Parameter list for solver and operator configuration.
     *
     * Supports typed parameter access with automatic conversion.
     * Designed to work with XML-parsed or programmatically constructed parameter lists.
     */
    class ParameterList {
    public:
        using Value = std::variant<int, double, std::string>;

        ParameterList() = default;

        /// Construct from a map (for backward compatibility)
        explicit ParameterList(const std::map<std::string, double>& params)
        {
            for (const auto& [key, val] : params) {
                params_[key] = val;
            }
        }

        /// Check if a parameter exists
        bool has(const std::string& key) const
        {
            return params_.find(key) != params_.end();
        }

        /// Get a double parameter
        double get_double(const std::string& key) const
        {
            auto it = params_.find(key);
            if (it == params_.end()) {
                throw std::runtime_error("Parameter '" + key + "' not found");
            }
            if (std::holds_alternative<int>(it->second)) {
                return static_cast<double>(std::get<int>(it->second));
            }
            if (std::holds_alternative<double>(it->second)) {
                return std::get<double>(it->second);
            }
            throw std::runtime_error("Parameter '" + key + "' is not a double");
        }

        /// Get an int parameter
        int get_int(const std::string& key) const
        {
            auto it = params_.find(key);
            if (it == params_.end()) {
                throw std::runtime_error("Parameter '" + key + "' not found");
            }
            if (std::holds_alternative<int>(it->second)) {
                return std::get<int>(it->second);
            }
            if (std::holds_alternative<double>(it->second)) {
                return static_cast<int>(std::get<double>(it->second));
            }
            throw std::runtime_error("Parameter '" + key + "' is not an int");
        }

        /// Get a string parameter
        std::string get_string(const std::string& key) const
        {
            auto it = params_.find(key);
            if (it == params_.end()) {
                throw std::runtime_error("Parameter '" + key + "' not found");
            }
            if (std::holds_alternative<std::string>(it->second)) {
                return std::get<std::string>(it->second);
            }
            throw std::runtime_error("Parameter '" + key + "' is not a string");
        }

        /// Set a double parameter
        void set(const std::string& key, double val)
        {
            params_[key] = val;
        }

        /// Set an int parameter
        void set(const std::string& key, int val)
        {
            params_[key] = val;
        }

        /// Set a string parameter
        void set(const std::string& key, const std::string& val)
        {
            params_[key] = val;
        }

        /// Merge another ParameterList into this one (overwrites existing)
        void merge(const ParameterList& other)
        {
            for (const auto& [key, val] : other.params_) {
                params_[key] = val;
            }
        }

        /// Clear all parameters
        void clear()
        {
            params_.clear();
        }

        // --- Sublist support for nested configurations ---

        void setSubList(const std::string& key, const ParameterList& sub)
        {
            subLists_[key] = sub;
        }

        const ParameterList* tryGetSubList(const std::string& key) const
        {
            auto it = subLists_.find(key);
            return (it != subLists_.end()) ? &it->second : nullptr;
        }

    private:
        std::map<std::string, Value> params_;
        std::map<std::string, ParameterList> subLists_;
    };

} // namespace mpfem

#endif // MPFEM_PARAMETER_LIST_HPP