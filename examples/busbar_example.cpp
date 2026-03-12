/**
 * @file busbar_example.cpp
 * @brief Example demonstrating mesh reading for the busbar test case.
 * 
 * This example shows how to:
 * 1. Read a mesh from mphtxt format
 * 2. Verify mesh statistics (domains, boundaries)
 * 3. Create a finite element space
 */

#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "mesh/mesh_topology.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "core/logger.hpp"

using namespace mpfem;

int main(int argc, char* argv[]) {
    Logger::setLevel(LogLevel::Info);
    
    // Default mesh path
    std::string meshPath = "cases/busbar/mesh.mphtxt";
    
    // Override from command line if provided
    if (argc > 1) {
        meshPath = argv[1];
    }
    
    LOG_INFO << "=== Busbar Example ===";
    LOG_INFO << "Reading mesh from: " << meshPath;
    
    try {
        // Read mesh
        Mesh mesh = MphtxtReader::read(meshPath);
        
        // Print mesh statistics
        LOG_INFO << "Mesh dimension: " << mesh.dim();
        LOG_INFO << "Number of vertices: " << mesh.numVertices();
        LOG_INFO << "Number of volume elements: " << mesh.numElements();
        LOG_INFO << "Number of boundary elements: " << mesh.numBdrElements();
        
        // Get domain and boundary IDs
        auto domains = mesh.domainIds();
        auto boundaries = mesh.boundaryIds();
        
        LOG_INFO << "Number of domains: " << domains.size();
        LOG_INFO << "Domain IDs: " << [&]() {
            std::string s;
            for (auto d : domains) {
                s += std::to_string(d) + " ";
            }
            return s;
        };
        
        LOG_INFO << "Number of boundaries: " << boundaries.size();
        LOG_INFO << "Boundary IDs: " << [&](){
            std::string s;
            for (auto b : boundaries) {
                s += std::to_string(b) + " ";
            }
            return s;
        };
        
        // Build topology
        MeshTopology topology(&mesh);
        LOG_INFO << "Number of faces: " << topology.numFaces();
        LOG_INFO << "Number of boundary faces: " << topology.numBoundaryFaces();
        LOG_INFO << "Number of interior faces: " << topology.numInteriorFaces();
        
        // Create finite element collection (linear H1)
        auto fec = FECollection::createH1(2);
        LOG_INFO << "Created H1 FE collection with order " << fec->order();
        
        // Create finite element space
        FESpace fes(&mesh, std::move(fec));
        LOG_INFO << "Created FE space with " << fes.numDofs() << " DOFs";
        
        // Verify expected values for busbar case
        if (domains.size() != 7) {
            LOG_WARN << "Expected 7 domains, got " << domains.size();
        }
        if (boundaries.size() != 43) {
            LOG_WARN << "Expected 43 boundaries, got " << boundaries.size();
        }
        
        LOG_INFO << "=== Mesh reading successful! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}
