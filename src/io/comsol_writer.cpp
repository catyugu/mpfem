/**
 * @file comsol_writer.cpp
 * @brief COMSOL-compatible result file writer implementation
 */

#include "comsol_writer.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <ctime>

namespace mpfem {

ComsolWriter::ComsolWriter() = default;

ComsolWriter::~ComsolWriter() {
    close();
}

bool ComsolWriter::open(const std::string& filename) {
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        MPFEM_ERROR("Failed to open COMSOL result file: " << filename);
        return false;
    }
    
    // Set high precision for output
    file_ << std::setprecision(15);
    
    return true;
}

void ComsolWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

std::string ComsolWriter::get_current_datetime() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%b %e %Y, %H:%M");
    return oss.str();
}

void ComsolWriter::write_header(const Mesh& mesh, 
                                 const std::vector<ComsolField>& fields) {
    // Get current date/time
    std::string datetime = get_current_datetime();
    
    // Build description string
    std::string description;
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) description += ", ";
        description += fields[i].description.empty() 
            ? fields[i].name : fields[i].description;
    }
    
    // Write header comments
    file_ << "% Model:              " << model_name_ << "\n";
    file_ << "% Version:            " << version_ << "\n";
    file_ << "% Date:               " << datetime << "\n";
    file_ << "% Dimension:          " << mesh.dimension() << "\n";
    file_ << "% Nodes:              " << mesh.num_vertices() << "\n";
    file_ << "% Expressions:        " << fields.size() << "\n";
    file_ << "% Description:        " << description << "\n";
    file_ << "% Length unit:        " << length_unit_ << "\n";
    
    // Write column headers
    file_ << "% x                       y                        z";
    for (const auto& field : fields) {
        file_ << "                        " << field.name;
    }
    file_ << "\n";
}

void ComsolWriter::write_data(const Mesh& mesh,
                               const std::vector<ComsolField>& fields) {
    SizeType num_nodes = mesh.num_vertices();
    
    for (SizeType i = 0; i < num_nodes; ++i) {
        // Write coordinates
        Point<3> p = mesh.vertex(static_cast<Index>(i));
        file_ << std::scientific << std::setprecision(15)
              << std::setw(24) << p.x()
              << std::setw(25) << p.y()
              << std::setw(25) << p.z();
        
        // Write field values
        for (const auto& field : fields) {
            if (i < field.values.size()) {
                file_ << std::setw(25) << field.values[i];
            } else {
                file_ << std::setw(25) << 0.0;
            }
        }
        file_ << "\n";
    }
}

void ComsolWriter::write(const std::string& filename,
                         const Mesh& mesh,
                         const std::vector<ComsolField>& fields,
                         const std::string& model_name) {
    ComsolWriter writer;
    writer.set_model_name(model_name);
    
    if (!writer.open(filename)) {
        return;
    }
    
    writer.write_header(mesh, fields);
    writer.write_data(mesh, fields);
    writer.close();
}

ComsolField ComsolWriter::create_field(const std::string& name,
                                        const DynamicVector& solution,
                                        const DoFHandler& dof_handler,
                                        bool is_vector) {
    ComsolField field(name);
    
    const auto* fe_space = dof_handler.fe_space();
    const Mesh& mesh = *fe_space->mesh();
    int n_components = fe_space->n_components();
    
    SizeType num_nodes = mesh.num_vertices();
    field.values.resize(num_nodes, 0.0);
    
    if (!is_vector) {
        // Scalar field
        for (SizeType node = 0; node < num_nodes; ++node) {
            Index dof = static_cast<Index>(node);
            if (dof < solution.size()) {
                field.values[node] = solution[dof];
            }
        }
    } else {
        // Vector field - compute magnitude
        for (SizeType node = 0; node < num_nodes; ++node) {
            Scalar mag = 0.0;
            for (int comp = 0; comp < n_components; ++comp) {
                Index dof = static_cast<Index>(node) * n_components + comp;
                if (dof < solution.size()) {
                    Scalar val = solution[dof];
                    mag += val * val;
                }
            }
            field.values[node] = std::sqrt(mag);
        }
    }
    
    return field;
}

ComsolField ComsolWriter::create_displacement_magnitude(
    const std::string& name,
    const DynamicVector& solution,
    const DoFHandler& dof_handler,
    int dim) {
    
    ComsolField field(name);
    
    const auto* fe_space = dof_handler.fe_space();
    const Mesh& mesh = *fe_space->mesh();
    int n_components = fe_space->n_components();
    
    SizeType num_nodes = mesh.num_vertices();
    field.values.resize(num_nodes, 0.0);
    
    for (SizeType node = 0; node < num_nodes; ++node) {
        Scalar mag = 0.0;
        for (int comp = 0; comp < dim && comp < n_components; ++comp) {
            Index dof = static_cast<Index>(node) * n_components + comp;
            if (dof < solution.size()) {
                Scalar val = solution[dof];
                mag += val * val;
            }
        }
        field.values[node] = std::sqrt(mag);
    }
    
    return field;
}

}  // namespace mpfem
