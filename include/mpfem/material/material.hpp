#pragma once

#include <map>
#include <string>

namespace mpfem {

/// 材料属性结构体
struct MaterialProperty {
  std::string raw_value;   ///< 原始值字符串
  double si_value = 0.0;   ///< SI 单位值
};

/// 材料结构体
struct Material {
  std::string tag;                                      ///< 材料标签
  std::string label;                                    ///< 材料名称
  std::map<std::string, MaterialProperty> properties;   ///< 属性映射
};

}  // namespace mpfem
