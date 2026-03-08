#pragma once

#include "mpfem/material/material.hpp"

#include <vector>

namespace mpfem {

class ComsolMaterialReader {
 public:
  std::vector<Material> Read(const std::string& xml_path) const;

 private:
  static std::string Trim(const std::string& value);
  static double ConvertToSI(const std::string& value_with_unit);
};

}  // namespace mpfem
