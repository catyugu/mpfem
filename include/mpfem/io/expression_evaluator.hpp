#pragma once

#include <string>
#include <map>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <functional>

namespace mpfem {

// 简单表达式求值器
// 支持变量替换和基本算术运算
class ExpressionEvaluator {
 public:
  ExpressionEvaluator() = default;

  // 设置变量
  void SetVariable(const std::string& name, double value) {
    variables_[name] = value;
  }

  // 批量设置变量
  void SetVariables(const std::map<std::string, double>& vars) {
    for (const auto& [name, value] : vars) {
      variables_[name] = value;
    }
  }

  // 获取变量
  double GetVariable(const std::string& name) const {
    auto it = variables_.find(name);
    if (it != variables_.end()) {
      return it->second;
    }
    throw std::runtime_error("Variable not found: " + name);
  }

  // 检查变量是否存在
  bool HasVariable(const std::string& name) const {
    return variables_.find(name) != variables_.end();
  }

  // 求值表达式
  double Evaluate(const std::string& expr) const {
    // 简单实现：先尝试直接解析为数值
    // 然后尝试变量替换
    std::string processed = expr;

    // 移除空白
    processed.erase(std::remove_if(processed.begin(), processed.end(),
                                    [](char c) { return std::isspace(c); }),
                     processed.end());

    // 尝试直接解析
    try {
      return std::stod(processed);
    } catch (...) {
      // 不是纯数字，继续处理
    }

    // 变量替换
    for (const auto& [name, value] : variables_) {
      size_t pos = 0;
      while ((pos = processed.find(name, pos)) != std::string::npos) {
        // 检查是否是完整的变量名（不是其他变量名的子串）
        bool is_whole = true;
        if (pos > 0 && (std::isalnum(processed[pos - 1]) || processed[pos - 1] == '_')) {
          is_whole = false;
        }
        if (pos + name.length() < processed.length() &&
            (std::isalnum(processed[pos + name.length()]) ||
             processed[pos + name.length()] == '_')) {
          is_whole = false;
        }

        if (is_whole) {
          processed.replace(pos, name.length(), std::to_string(value));
          pos += std::to_string(value).length();
        } else {
          pos += name.length();
        }
      }
    }

    // 简单表达式解析（支持基本运算）
    return ParseSimpleExpression(processed);
  }

  // 静态求值函数
  static double Eval(const std::string& expr,
                     const std::map<std::string, double>& vars = {}) {
    ExpressionEvaluator eval;
    eval.SetVariables(vars);
    return eval.Evaluate(expr);
  }

 private:
  // 解析简单表达式（支持 +, -, *, /, ()）
  double ParseSimpleExpression(const std::string& expr) const {
    std::istringstream iss(expr);
    return ParseAddSub(iss);
  }

  double ParseAddSub(std::istringstream& iss) const {
    double result = ParseMulDiv(iss);
    char op;
    while (iss >> op) {
      if (op == '+') {
        result += ParseMulDiv(iss);
      } else if (op == '-') {
        result -= ParseMulDiv(iss);
      } else {
        iss.putback(op);
        break;
      }
    }
    return result;
  }

  double ParseMulDiv(std::istringstream& iss) const {
    double result = ParsePower(iss);
    char op;
    while (iss >> op) {
      if (op == '*') {
        result *= ParsePower(iss);
      } else if (op == '/') {
        double divisor = ParsePower(iss);
        if (std::abs(divisor) < 1e-15) {
          throw std::runtime_error("Division by zero in expression");
        }
        result /= divisor;
      } else {
        iss.putback(op);
        break;
      }
    }
    return result;
  }

  double ParsePower(std::istringstream& iss) const {
    double result = ParseAtom(iss);
    char op;
    if (iss >> op && op == '^') {
      result = std::pow(result, ParsePower(iss));
    } else {
      iss.putback(op);
    }
    return result;
  }

  double ParseAtom(std::istringstream& iss) const {
    char c;
    iss >> c;

    if (c == '(') {
      double result = ParseAddSub(iss);
      iss >> c;  // 读取 ')'
      return result;
    } else if (c == '-') {
      return -ParseAtom(iss);
    } else if (c == '+') {
      return ParseAtom(iss);
    } else {
      iss.putback(c);
      double result;
      iss >> result;
      return result;
    }
  }

  std::map<std::string, double> variables_;
};

// 边界条件值解析器
class BCValueParser {
 public:
  BCValueParser() = default;

  void SetVariables(const std::map<std::string, double>& vars) {
    eval_.SetVariables(vars);
  }

  // 解析边界条件值
  double Parse(const std::string& value_expr) const {
    return eval_.Evaluate(value_expr);
  }

  // 检查是否是常数
  static bool IsConstant(const std::string& expr) {
    try {
      std::stod(expr);
      return true;
    } catch (...) {
      return false;
    }
  }

 private:
  ExpressionEvaluator eval_;
};

}  // namespace mpfem
