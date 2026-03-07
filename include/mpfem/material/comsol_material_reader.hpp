#pragma once

#include <map>
#include <string>
#include <vector>

namespace mpfem {

struct MaterialProperty {
  std::string raw_value;
  double si_value = 0.0;
};

struct Material {
  std::string tag;
  std::string label;
  std::map<std::string, MaterialProperty> properties;
};

class ComsolMaterialReader {
 public:
  std::vector<Material> Read(const std::string& xml_path) const;

 private:
  static std::string Trim(const std::string& value);
  static double ConvertToSI(const std::string& value_with_unit);
};

}  // namespace mpfem
