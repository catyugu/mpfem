#ifndef MPFEM_IO_PROBLEM_INPUT_LOADER_HPP
#define MPFEM_IO_PROBLEM_INPUT_LOADER_HPP

#include "mesh/mesh.hpp"
#include "io/case_definition.hpp"
#include "io/material_database.hpp"

#include <memory>
#include <string>

namespace mpfem {

struct ProblemInputData {
    CaseDefinition caseDefinition;
    std::unique_ptr<Mesh> mesh;
    MaterialDatabase materials;
};

class ProblemInputLoader {
public:
    virtual ~ProblemInputLoader() = default;
    virtual ProblemInputData load(const std::string& caseDir) const = 0;
};

std::unique_ptr<ProblemInputLoader> createXmlProblemInputLoader();

}  // namespace mpfem

#endif  // MPFEM_IO_PROBLEM_INPUT_LOADER_HPP