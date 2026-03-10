/**
 * @file test_dof.cpp
 * @brief Tests for DoF management module
 */

#include <gtest/gtest.h>
#include "dof/fe_space.hpp"
#include "dof/dof_handler.hpp"
#include "dof/dof_table.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mphtxt_reader.hpp"
#include "mesh/element.hpp"
#include "fem/fe_collection.hpp"
#include "core/logger.hpp"

using namespace mpfem;

// ============================================================
// DoFTable Tests
// ============================================================

TEST(DoFTableTest, BasicConstruction) {
    DoFTable table(3, 4);

    EXPECT_EQ(table.n_cells(), 3);
    EXPECT_EQ(table.total_entries(), 12);
    EXPECT_EQ(table.dofs_per_cell(0), 4);
    EXPECT_EQ(table.dofs_per_cell(1), 4);

    table(0, 0) = 0;
    table(0, 1) = 1;
    table(0, 2) = 2;
    table(0, 3) = 3;

    EXPECT_EQ(table(0, 0), 0);
    EXPECT_EQ(table(0, 3), 3);

    std::vector<Index> dofs;
    table.get_cell_dofs(0, dofs);
    ASSERT_EQ(dofs.size(), 4);
    EXPECT_EQ(dofs[0], 0);
    EXPECT_EQ(dofs[3], 3);
}

TEST(DoFTableTest, VariableDofsPerCell) {
    std::vector<int> dp_cell = {3, 4, 2};
    DoFTable table(dp_cell);

    EXPECT_EQ(table.n_cells(), 3);
    EXPECT_EQ(table.total_entries(), 9);
    EXPECT_EQ(table.dofs_per_cell(0), 3);
    EXPECT_EQ(table.dofs_per_cell(1), 4);
    EXPECT_EQ(table.dofs_per_cell(2), 2);
}

// ============================================================
// FESpace Tests
// ============================================================

class FESpaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<Mesh>();
        mesh_->set_dimension(3);
        mesh_->initialize_vertices(4);
        mesh_->set_vertex(0, Point<3>(0, 0, 0));
        mesh_->set_vertex(1, Point<3>(1, 0, 0));
        mesh_->set_vertex(2, Point<3>(0, 1, 0));
        mesh_->set_vertex(3, Point<3>(0, 0, 1));

        auto* block = mesh_->add_cell_block(ElementType::Tetrahedron);
        Index tet[] = {0, 1, 2, 3};
        block->add_element(tet, 1);  // domain_id = 1

        auto* faces = mesh_->add_face_block(ElementType::Triangle);
        Index tri[] = {0, 1, 2};
        faces->add_element(tri, 1);  // boundary_id = 1
    }

    std::unique_ptr<Mesh> mesh_;
};

TEST_F(FESpaceTest, CreateScalarSpace) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 1);
    EXPECT_EQ(fe_space.n_components(), 1);
    EXPECT_EQ(fe_space.mesh(), mesh_.get());
    EXPECT_EQ(fe_space.dofs_per_cell(GeometryType::Tetrahedron), 4);
}

TEST_F(FESpaceTest, CreateVectorSpace) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 3);
    EXPECT_EQ(fe_space.n_components(), 3);
    EXPECT_EQ(fe_space.dofs_per_cell(GeometryType::Tetrahedron), 12);
}

TEST_F(FESpaceTest, Initialize) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 1);
    fe_space.initialize();
    EXPECT_EQ(fe_space.n_dofs(), 4);
}

TEST_F(FESpaceTest, GetCellDofs) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 1);
    fe_space.initialize();

    std::vector<Index> dofs;
    fe_space.get_cell_dofs(0, 0, dofs);

    ASSERT_EQ(dofs.size(), 4);
    EXPECT_EQ(dofs[0], 0);
    EXPECT_EQ(dofs[1], 1);
    EXPECT_EQ(dofs[2], 2);
    EXPECT_EQ(dofs[3], 3);
}

// ============================================================
// DoFHandler Tests
// ============================================================

class DoFHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<Mesh>();
        mesh_->set_dimension(3);
        mesh_->initialize_vertices(5);
        mesh_->set_vertex(0, Point<3>(0, 0, 0));
        mesh_->set_vertex(1, Point<3>(1, 0, 0));
        mesh_->set_vertex(2, Point<3>(0, 1, 0));
        mesh_->set_vertex(3, Point<3>(0, 0, 1));
        mesh_->set_vertex(4, Point<3>(1, 1, 0));

        auto* block = mesh_->add_cell_block(ElementType::Tetrahedron);
        Index tet1[] = {0, 1, 2, 3};
        Index tet2[] = {1, 4, 2, 3};
        block->add_element(tet1, 1);
        block->add_element(tet2, 1);

        auto* faces = mesh_->add_face_block(ElementType::Triangle);
        Index tri[] = {0, 1, 2};
        faces->add_element(tri, 1);  // boundary_id = 1

        fe_space_ = std::make_unique<FESpace>(mesh_.get(), "Lagrange1", 1);
        fe_space_->initialize();
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FESpace> fe_space_;
};

TEST_F(DoFHandlerTest, Initialize) {
    DoFHandler handler;
    handler.initialize(fe_space_.get());
    EXPECT_EQ(handler.fe_space(), fe_space_.get());
}

TEST_F(DoFHandlerTest, DistributeDofs) {
    DoFHandler handler;
    handler.initialize(fe_space_.get());
    handler.distribute_dofs();

    EXPECT_EQ(handler.n_dofs(), 5);
    EXPECT_EQ(handler.n_free_dofs(), 5);
}

TEST_F(DoFHandlerTest, CellDofMapping) {
    DoFHandler handler;
    handler.initialize(fe_space_.get());
    handler.distribute_dofs();

    std::vector<Index> dofs;
    handler.get_cell_dofs(0, dofs);

    ASSERT_EQ(dofs.size(), 4);
    EXPECT_EQ(dofs[0], 0);
    EXPECT_EQ(dofs[1], 1);
    EXPECT_EQ(dofs[2], 2);
    EXPECT_EQ(dofs[3], 3);
}

TEST_F(DoFHandlerTest, DirichletBC) {
    DoFHandler handler;
    handler.initialize(fe_space_.get());
    handler.distribute_dofs();

    handler.add_dirichlet_bc(1, 0.0);
    handler.apply_boundary_conditions();

    EXPECT_GT(handler.n_constrained_dofs(), 0);
    EXPECT_LT(handler.n_free_dofs(), handler.n_dofs());
}

// ============================================================
// Busbar Mesh Tests
// ============================================================

class BusbarTest : public ::testing::Test {
protected:
    void SetUp() override {
        MphtxtReader reader;
        mesh_ = reader.read("cases/busbar/mesh.mphtxt");

        if (!mesh_) {
            GTEST_SKIP() << "Busbar mesh not found";
        }
    }

    std::unique_ptr<Mesh> mesh_;
};

TEST_F(BusbarTest, MeshInfo) {
    EXPECT_EQ(mesh_->num_vertices(), 7360);

    // Should have 7 domains
    EXPECT_EQ(mesh_->num_domains(), 7);

    // Should have 43 boundaries
    EXPECT_EQ(mesh_->num_boundaries(), 43);

    // Total cells: 29456 tets + 256 hexes + 40 pyramids = 29752
    EXPECT_EQ(mesh_->num_cells(), 29752);
}

TEST_F(BusbarTest, FESpaceScalar) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 1);
    fe_space.initialize();

    EXPECT_EQ(fe_space.n_dofs(), 7360);
}

TEST_F(BusbarTest, FESpaceVector) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 3);
    fe_space.initialize();

    EXPECT_EQ(fe_space.n_dofs(), 7360 * 3);
}

TEST_F(BusbarTest, DoFHandler) {
    FESpace fe_space(mesh_.get(), "Lagrange1", 1);
    fe_space.initialize();

    DoFHandler handler;
    handler.initialize(&fe_space);
    handler.distribute_dofs();

    EXPECT_EQ(handler.n_dofs(), 7360);

    const auto& bnd_ids = mesh_->boundary_ids();
    ASSERT_GT(bnd_ids.size(), 0);

    handler.add_dirichlet_bc(bnd_ids[0], 100.0);
    handler.apply_boundary_conditions();

    EXPECT_GT(handler.n_constrained_dofs(), 0);
}

TEST_F(BusbarTest, DomainBoundaryGroups) {
    // Verify domain groups
    const auto& domain_ids = mesh_->domain_ids();
    EXPECT_EQ(domain_ids.size(), 7);

    for (Index id : domain_ids) {
        const Domain* domain = mesh_->geometry().get_domain(id);
        ASSERT_NE(domain, nullptr);
        EXPECT_GT(domain->cells.size(), 0);
    }

    // Verify boundary groups
    const auto& boundary_ids = mesh_->boundary_ids();
    EXPECT_EQ(boundary_ids.size(), 43);

    for (Index id : boundary_ids) {
        const Boundary* boundary = mesh_->geometry().get_boundary(id);
        ASSERT_NE(boundary, nullptr);
        EXPECT_GT(boundary->faces.size(), 0);
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
