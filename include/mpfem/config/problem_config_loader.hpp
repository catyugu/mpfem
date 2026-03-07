#pragma once

#include <string>
#include <vector>

#include "mpfem/core/problem_definition.hpp"

namespace mpfem {

class ProblemConfigLoader {
 public:
  ProblemDefinition LoadFromXml(const std::string& case_file_path) const;

 private:
  static std::vector<int> ParseIdList(const std::string& text);
  static std::string ResolveRelativePath(const std::string& case_file_path,
                                         const std::string& candidate_path);
};

}  // namespace mpfem
