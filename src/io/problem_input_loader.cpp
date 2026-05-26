#include "io/problem_input_loader.hpp"

#include "core/logger.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "io/mphtxt_reader.hpp"
#include "mesh/mesh.hpp"

namespace mpfem {
    namespace {

        class XmlProblemInputLoader final : public ProblemInputLoader {
        public:
            ProblemInputData load(const std::string& caseDir) const override
            {
                ProblemInputData input;

                const std::string casePath = caseDir + "/case.xml";
                LOG_INFO << "Reading case from " << casePath;
                CaseXmlReader::readFromFile(casePath, input.caseDefinition);

                const std::string meshPath = caseDir + "/" + input.caseDefinition.meshPath;
                Real scaleFactor = 1.0;
                if (input.caseDefinition.meshUnit == "mm") {
                    scaleFactor = 0.001;
                    LOG_INFO << "Mesh unit: mm, scaling coordinates by " << scaleFactor;
                }
                input.mesh = std::make_unique<Mesh>(MphtxtReader::read(meshPath, scaleFactor));
                const std::string materialPath = caseDir + "/" + input.caseDefinition.materialsPath;
                LOG_INFO << "Reading materials from " << materialPath;
                MaterialXmlReader::readFromFile(materialPath, input.materials);

                return input;
            }
        };

    } // namespace

    std::unique_ptr<ProblemInputLoader> createXmlProblemInputLoader()
    {
        return std::make_unique<XmlProblemInputLoader>();
    }

} // namespace mpfem