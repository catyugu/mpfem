#include <gtest/gtest.h>
#include "mesh/mesh.hpp"
#include "io/mphtxt_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MeshReadTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Info);
    }
    
    // Helper to get test data path
    static std::string dataPath(const std::string& relativePath) {
#ifdef MPFEM_PROJECT_ROOT
        return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
        return relativePath;
#endif
    }
};

TEST_F(MeshReadTest, ReadBusbarMesh) {
    // Test mesh file path
    std::string meshPath = dataPath("cases/busbar_steady/mesh.mphtxt");
    
    // Read mesh
    Mesh mesh = MphtxtReader::read(meshPath);
    
    // Verify basic stats
    EXPECT_EQ(mesh.dim(), 3);
    EXPECT_EQ(mesh.numVertices(), 7340);  // Actual vertex count in the file
    
    // Verify we have volume elements (tetrahedra)
    EXPECT_GT(mesh.numElements(), 0);
    
    // Verify we have boundary elements (triangles)
    EXPECT_GT(mesh.numBdrElements(), 0);
    
    // Count unique domain IDs (only from tetrahedral elements)
    std::set<Index> domains;
    for (Index i = 0; i < mesh.numElements(); ++i) {
        const auto& elem = mesh.element(i);
        // Only count tetrahedra for domain IDs
        if (elem.geometry() == Geometry::Tetrahedron) {
            domains.insert(elem.attribute());
        }
    }
    EXPECT_EQ(domains.size(), 7) << "Expected 7 domains";
    
    // Count unique boundary IDs
    std::set<Index> boundaries;
    for (Index i = 0; i < mesh.numBdrElements(); ++i) {
        boundaries.insert(mesh.bdrElement(i).attribute());
    }
    EXPECT_EQ(boundaries.size(), 43) << "Expected 43 boundaries";
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
