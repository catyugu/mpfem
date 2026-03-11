#include <gtest/gtest.h>
#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MeshReadTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Debug);
    }
};

TEST_F(MeshReadTest, ReadBusbarMesh) {
    // Test mesh file path
    std::string meshPath = "cases/busbar/mesh.mphtxt";
    
    // Read mesh
    Mesh mesh = MphtxtReader::read(meshPath);
    
    // Verify basic stats
    EXPECT_EQ(mesh.dim(), 3);
    EXPECT_EQ(mesh.numVertices(), 7360);
    
    // Verify we have volume elements (tetrahedra)
    EXPECT_GT(mesh.numElements(), 0);
    
    // Verify we have boundary elements (triangles)
    EXPECT_GT(mesh.numBdrElements(), 0);
    
    // Count unique domain IDs
    std::set<Index> domains;
    for (Index i = 0; i < mesh.numElements(); ++i) {
        domains.insert(mesh.element(i).attribute());
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
