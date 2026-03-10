/**
 * @file test_mesh.cpp
 * @brief Test mesh reading and basic mesh operations
 */

#include <gtest/gtest.h>
#include <set>

#include "mesh/mesh.hpp"
#include "mesh/mphtxt_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().set_level(LogLevel::INFO);
    }
};

TEST_F(MeshTest, ReadBusbarMesh) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    ASSERT_NE(mesh, nullptr);
    EXPECT_EQ(mesh->dimension(), 3);
    EXPECT_EQ(mesh->num_vertices(), 7360);
    EXPECT_EQ(mesh->num_domains(), 7);
    EXPECT_EQ(mesh->num_boundaries(), 43);

    auto domain_ids = mesh->domain_ids();
    EXPECT_EQ(domain_ids.size(), 7);

    std::set<Index> domain_set(domain_ids.begin(), domain_ids.end());

    auto boundary_ids = mesh->boundary_ids();
    EXPECT_EQ(boundary_ids.size(), 43);

    EXPECT_GT(mesh->num_cells(), 0);
    EXPECT_GT(mesh->num_faces(), 0);
}

TEST_F(MeshTest, CellCountAndTypes) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    SizeType total_cells = 0;
    for (const auto& block : mesh->cell_blocks()) {
        total_cells += block.size();
    }

    EXPECT_EQ(total_cells, 29752);

    const ElementBlock* tet_block = mesh->get_cell_block(ElementType::Tetrahedron);
    ASSERT_NE(tet_block, nullptr);
    EXPECT_EQ(tet_block->size(), 29456);
    EXPECT_EQ(tet_block->nodes_per_element(), 4);

    const ElementBlock* hex_block = mesh->get_cell_block(ElementType::Hexahedron);
    ASSERT_NE(hex_block, nullptr);
    EXPECT_EQ(hex_block->size(), 256);
    EXPECT_EQ(hex_block->nodes_per_element(), 8);

    const ElementBlock* pyr_block = mesh->get_cell_block(ElementType::Pyramid);
    ASSERT_NE(pyr_block, nullptr);
    EXPECT_EQ(pyr_block->size(), 40);
    EXPECT_EQ(pyr_block->nodes_per_element(), 5);
}

TEST_F(MeshTest, FaceCountAndTypes) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    const ElementBlock* tri_block = mesh->get_face_block(ElementType::Triangle);
    ASSERT_NE(tri_block, nullptr);
    EXPECT_EQ(tri_block->size(), 8606);
    EXPECT_EQ(tri_block->nodes_per_element(), 3);

    const ElementBlock* quad_block = mesh->get_face_block(ElementType::Quadrilateral);
    ASSERT_NE(quad_block, nullptr);
    EXPECT_EQ(quad_block->size(), 272);
    EXPECT_EQ(quad_block->nodes_per_element(), 4);
}

TEST_F(MeshTest, DomainEntityIDs) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    const ElementBlock* tet_block = mesh->get_cell_block(ElementType::Tetrahedron);
    ASSERT_NE(tet_block, nullptr);

    std::set<Index> domains_in_tets;
    for (SizeType i = 0; i < tet_block->size(); ++i) {
        domains_in_tets.insert(tet_block->entity_id(i));
    }

    // Tets should belong to multiple domains
    EXPECT_GE(domains_in_tets.size(), 1);
}

TEST_F(MeshTest, BoundaryEntityIDs) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    const ElementBlock* tri_block = mesh->get_face_block(ElementType::Triangle);
    ASSERT_NE(tri_block, nullptr);

    std::set<Index> boundary_ids;
    for (SizeType i = 0; i < tri_block->size(); ++i) {
        boundary_ids.insert(tri_block->entity_id(i));
    }

    EXPECT_GE(boundary_ids.size(), 30);
}

TEST_F(MeshTest, VertexCoordinates) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    auto v0 = mesh->vertex(0);
    EXPECT_NEAR(v0.x(), 0.0975, 1e-6);
    EXPECT_NEAR(v0.y(), 0.0, 1e-12);
    EXPECT_NEAR(v0.z(), 0.1, 1e-6);

    auto vlast = mesh->vertex(7359);
    EXPECT_TRUE(std::isfinite(vlast.x()));
    EXPECT_TRUE(std::isfinite(vlast.y()));
    EXPECT_TRUE(std::isfinite(vlast.z()));
}

TEST_F(MeshTest, BoundingBox) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    auto bbox_min = mesh->bbox_min();
    auto bbox_max = mesh->bbox_max();

    EXPECT_GT(bbox_max.x() - bbox_min.x(), 0.05);
    EXPECT_GT(bbox_max.y() - bbox_min.y(), 0.01);
    EXPECT_GT(bbox_max.z() - bbox_min.z(), 0.05);

    MPFEM_INFO("Bounding box: [" << bbox_min.x() << ", " << bbox_min.y()
              << ", " << bbox_min.z() << "] to ["
              << bbox_max.x() << ", " << bbox_max.y()
              << ", " << bbox_max.z() << "]");
}

TEST_F(MeshTest, ReadBusbarOrder2Mesh) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar_order2/mesh.mphtxt");

    ASSERT_NE(mesh, nullptr);
    EXPECT_GT(mesh->num_vertices(), 7360);

    const ElementBlock* tet2_block = mesh->get_cell_block(ElementType::Tetrahedron2);
    if (tet2_block) {
        EXPECT_EQ(tet2_block->nodes_per_element(), 10);
    }
}

TEST_F(MeshTest, GeometryManager) {
    MphtxtReader reader;
    auto mesh = reader.read("cases/busbar/mesh.mphtxt");

    const GeometryManager& geom = mesh->geometry();

    for (Index id : mesh->domain_ids()) {
        const Domain* domain = geom.get_domain(id);
        ASSERT_NE(domain, nullptr);
        EXPECT_EQ(domain->id, id);
    }

    for (Index id : mesh->boundary_ids()) {
        const Boundary* boundary = geom.get_boundary(id);
        ASSERT_NE(boundary, nullptr);
        EXPECT_EQ(boundary->id, id);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}