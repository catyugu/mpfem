#include "mpfem/material/comsol_material_reader.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>

#include "tinyxml2.h"

namespace mpfem {

namespace {

std::string TrimLocal(const std::string& value) {
  auto begin = value.begin();
  while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }

  auto end = value.end();
  while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
    --end;
  }

  return std::string(begin, end);
}

std::string ExtractFirstScalarToken(const std::string& raw_value) {
  if (raw_value.empty()) {
    return {};
  }

  std::string cleaned = raw_value;
  if (cleaned.front() == '{' && cleaned.back() == '}') {
    cleaned = cleaned.substr(1, cleaned.size() - 2);
  }

  cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\''), cleaned.end());

  const std::size_t comma = cleaned.find(',');
  if (comma != std::string::npos) {
    cleaned = cleaned.substr(0, comma);
  }

  return TrimLocal(cleaned);
}

double UnitScale(const std::string& unit) {
  if (unit.empty() || unit == "1" || unit == "1/K") {
    return 1.0;
  }
  if (unit == "GPa") {
    return 1.0e9;
  }
  if (unit == "MPa") {
    return 1.0e6;
  }
  if (unit == "kPa") {
    return 1.0e3;
  }
  if (unit == "S/m" || unit == "W/(m*K)" || unit == "kg/m^3" ||
      unit == "J/(kg*K)" || unit == "K" || unit == "ohm*m") {
    return 1.0;
  }

  throw std::runtime_error("Unsupported unit in material parser: " + unit);
}

}  // namespace

std::vector<ComsolMaterial> ComsolMaterialReader::Read(const std::string& xml_path) const {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to open material XML: " + xml_path);
  }

  const tinyxml2::XMLElement* root = doc.FirstChildElement("archive");
  if (root == nullptr) {
    throw std::runtime_error("Invalid material XML: missing <archive>");
  }
  const tinyxml2::XMLElement* model = root->FirstChildElement("model");
  if (model == nullptr) {
    throw std::runtime_error("Invalid material XML: missing <model>");
  }

  std::vector<ComsolMaterial> materials;
  for (const tinyxml2::XMLElement* material_elem = model->FirstChildElement("material");
       material_elem != nullptr;
       material_elem = material_elem->NextSiblingElement("material")) {
    ComsolMaterial material;

    if (const char* tag = material_elem->Attribute("tag")) {
      material.tag = tag;
    }

    if (const tinyxml2::XMLElement* label_elem = material_elem->FirstChildElement("label")) {
      if (const char* label = label_elem->Attribute("label")) {
        material.label = label;
      }
    }

    for (const tinyxml2::XMLElement* group = material_elem->FirstChildElement("propertyGroup");
         group != nullptr;
         group = group->NextSiblingElement("propertyGroup")) {
      for (const tinyxml2::XMLElement* set_elem = group->FirstChildElement("set");
           set_elem != nullptr;
           set_elem = set_elem->NextSiblingElement("set")) {
        const char* name_attr = set_elem->Attribute("name");
        const char* value_attr = set_elem->Attribute("value");
        if (name_attr == nullptr || value_attr == nullptr) {
          continue;
        }

        MaterialProperty property;
        property.raw_value = value_attr;
        try {
          property.si_value = ConvertToSI(property.raw_value);
          material.properties[name_attr] = property;
        } catch (const std::exception&) {
          // Skip non-numeric values in the first implementation stage.
        }
      }
    }

    if (!material.tag.empty()) {
      materials.push_back(std::move(material));
    }
  }

  return materials;
}

std::string ComsolMaterialReader::Trim(const std::string& value) {
  return TrimLocal(value);
}

double ComsolMaterialReader::ConvertToSI(const std::string& value_with_unit) {
  const std::string token = ExtractFirstScalarToken(value_with_unit);
  if (token.empty()) {
    return 0.0;
  }

  static const std::regex pattern(R"(^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\[([^\]]+)\])?\s*$)");
  std::smatch match;
  if (!std::regex_match(token, match, pattern)) {
    throw std::runtime_error("Failed to parse value token: " + token);
  }

  const double value = std::stod(match[1].str());
  const std::string unit = match[2].matched ? match[2].str() : "";
  return value * UnitScale(unit);
}

}  // namespace mpfem
