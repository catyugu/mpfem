/**
 * @file test_assembly.cpp
 * @brief Tests for Assembly module using FieldSpace
 */

#include <gtest/gtest.h>
#include "assembly/assembly.hpp"
#include "dof/field_space.hpp"
#include "dof/field_registry.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mphtxt_reader.hpp"
#include "mesh/element.hpp"
#include "fem/fe_collection.hpp"
#include "linalg/direct_solver.hpp"
#include "core/logger.hpp"

using namespace mpfem;

// ============================================================
// FEValues Tests
// ============================================================

class FEValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple tetrahedron mesh
        mesh_ = std::make_unique<Mesh>();
        mesh_->set_dimension(3);
        mesh_->initialize_vertices(4);
        mesh_->set_vertex(0, Point<3>(0, 0, 0));
        mesh_->set_vertex(1, Point<3>(1, 0, 0));
        mesh_->set_vertex(2, Point<3>(0, 1, 0));
        mesh_->set_vertex(3, Point<3>(0, 0, 1));

        auto* block = mesh_->add_cell_block(ElementType::Tetrahedron);
        Index tet[] = {0, 1, 2, 3};
        block->add_element(tet, 1);
        
        mesh_->build_topology();

        // Create FieldSpace
        field_ = std::make_unique<FieldSpace>("test_field", mesh_.get(), 1, 1);
        
        auto fe = create_fe(GeometryType::Tetrahedron, 1, 1);
        fe_ = fe.get();
        fe_owner_ = std::move(fe);
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FieldSpace> field_;
    std::unique_ptr<FiniteElement> fe_owner_;
    const FiniteElement* fe_;
};

TEST_F(FEValuesTest, BasicConstruction) {
    FEValues fe_values(fe_);
    
    EXPECT_EQ(fe_values.dofs_per_cell(), 4);
    EXPECT_EQ(fe_values.n_quadrature_points(), fe_->n_quadrature_points());
}

TEST_F(FEValuesTest, ReinitCell) {
    FEValues fe_values(fe_, UpdateFlags::UpdateAll);
    fe_values.reinit(*field_, 0);
    
    // Check that quadrature points are computed
    for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
        EXPECT_GT(fe_values.JxW(q), 0);
    }
}

TEST_F(FEValuesTest, JxWComputation) {
    FEValues fe_values(fe_, UpdateFlags::UpdateJxW);
    fe_values.reinit(*field_, 0);
    
    // For unit tetrahedron, volume = 1/6
    // Sum of JxW should equal volume
    Scalar total_volume = 0;
    for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
        total_volume += fe_values.JxW(q);
    }
    
    // Expected volume: 1/6 ≈ 0.16667
    EXPECT_NEAR(total_volume, 1.0/6.0, 1e-10);
}

TEST_F(FEValuesTest, ShapeGradients) {
    FEValues fe_values(fe_, UpdateFlags::UpdateGradients | UpdateFlags::UpdateJxW);
    fe_values.reinit(*field_, 0);
    
    // Shape gradients should be computed
    for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
        for (int i = 0; i < fe_values.dofs_per_cell(); ++i) {
            const auto& grad = fe_values.shape_grad(i, q);
            // Gradient should have finite values
            EXPECT_TRUE(std::isfinite(grad.x()));
            EXPECT_TRUE(std::isfinite(grad.y()));
            EXPECT_TRUE(std::isfinite(grad.z()));
        }
    }
}

TEST_F(FEValuesTest, AssembleLocalToGlobal) {
    FEValues fe_values(fe_, UpdateFlags::UpdateDefault);
    fe_values.reinit(*field_, 0);
    
    // Create local matrix
    DynamicMatrix K_local(4, 4);
    K_local.setOnes();
    
    // Assemble to global
    SparseMatrix K(4, 4);
    K.setZero();
    fe_values.assemble_local_to_global(K, K_local);
    
    // Check that values were assembled
    EXPECT_GT(K.nonZeros(), 0);
}

// ============================================================
// BilinearForm Tests
// ============================================================

class BilinearFormTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple tetrahedron mesh
        mesh_ = std::make_unique<Mesh>();
        mesh_->set_dimension(3);
        mesh_->initialize_vertices(4);
        mesh_->set_vertex(0, Point<3>(0, 0, 0));
        mesh_->set_vertex(1, Point<3>(1, 0, 0));
        mesh_->set_vertex(2, Point<3>(0, 1, 0));
        mesh_->set_vertex(3, Point<3>(0, 0, 1));

        auto* block = mesh_->add_cell_block(ElementType::Tetrahedron);
        Index tet[] = {0, 1, 2, 3};
        block->add_element(tet, 1);
        
        mesh_->build_topology();

        field_ = std::make_unique<FieldSpace>("test", mesh_.get(), 1, 1);
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FieldSpace> field_;
};

TEST_F(BilinearFormTest, LaplacianAssembly) {
    BilinearForm form(field_.get());
    
    SparseMatrix K;
    form.assemble(BilinearForms::laplacian(1.0), K);
    
    // Check matrix properties
    EXPECT_EQ(K.rows(), 4);
    EXPECT_EQ(K.cols(), 4);
    
    // Matrix should be symmetric
    SparseMatrix Kt = K.transpose();
    EXPECT_TRUE(K.isApprox(Kt));
    
    // Matrix should be positive semi-definite (diagonal entries > 0)
    for (int i = 0; i < K.rows(); ++i) {
        EXPECT_GT(K.coeff(i, i), 0);
    }
}

TEST_F(BilinearFormTest, MassMatrix) {
    BilinearForm form(field_.get());
    
    SparseMatrix M;
    form.assemble(BilinearForms::mass(1.0), M);
    
    // Check matrix properties
    EXPECT_EQ(M.rows(), 4);
    EXPECT_EQ(M.cols(), 4);
    
    // Sum of mass matrix entries should equal volume
    Scalar total_mass = 0;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
            total_mass += it.value();
        }
    }
    
    // Volume of unit tetrahedron is 1/6
    EXPECT_NEAR(total_mass, 1.0/6.0, 1e-10);
}

TEST_F(BilinearFormTest, WithCoefficients) {
    BilinearForm form(field_.get());
    
    std::unordered_map<Index, Scalar> coeffs;
    coeffs[1] = 2.0;  // Domain 1 has coefficient 2.0
    
    SparseMatrix K;
    form.assemble_with_coefficients(BilinearForms::laplacian(1.0), K, coeffs);
    
    // Matrix should still be valid
    EXPECT_EQ(K.rows(), 4);
    EXPECT_GT(K.nonZeros(), 0);
}

// ============================================================
// LinearForm Tests
// ============================================================

class LinearFormTest : public ::testing::Test {
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
        block->add_element(tet, 1);
        
        mesh_->build_topology();

        field_ = std::make_unique<FieldSpace>("test", mesh_.get(), 1, 1);
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FieldSpace> field_;
};

TEST_F(LinearFormTest, SourceAssembly) {
    LinearForm form(field_.get());
    
    DynamicVector F;
    form.assemble(LinearForms::source(1.0), F);
    
    EXPECT_EQ(F.size(), 4);
    
    // Sum of load vector should equal source * volume
    Scalar total_load = F.sum();
    // Volume = 1/6, source = 1
    EXPECT_NEAR(total_load, 1.0/6.0, 1e-10);
}

TEST_F(LinearFormTest, WithSource) {
    LinearForm form(field_.get());
    
    std::unordered_map<Index, Scalar> sources;
    sources[1] = 5.0;  // Domain 1 has source 5.0
    
    DynamicVector F;
    form.assemble_with_source(LinearForms::source(1.0), F, sources);
    
    // Sum should be 5.0 * volume
    EXPECT_NEAR(F.sum(), 5.0/6.0, 1e-10);
}

// ============================================================
// Poisson Problem Test (Integration Test)
// ============================================================

class PoissonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple cube mesh (two tetrahedra)
        mesh_ = std::make_unique<Mesh>();
        mesh_->set_dimension(3);
        mesh_->initialize_vertices(5);
        
        mesh_->set_vertex(0, Point<3>(0, 0, 0));
        mesh_->set_vertex(1, Point<3>(1, 0, 0));
        mesh_->set_vertex(2, Point<3>(0, 1, 0));
        mesh_->set_vertex(3, Point<3>(1, 1, 0));
        mesh_->set_vertex(4, Point<3>(0.5, 0.5, 1));

        auto* block = mesh_->add_cell_block(ElementType::Tetrahedron);
        Index tet1[] = {0, 1, 2, 4};
        Index tet2[] = {1, 3, 2, 4};
        block->add_element(tet1, 1);
        block->add_element(tet2, 1);

        auto* faces = mesh_->add_face_block(ElementType::Triangle);
        Index tri[] = {0, 1, 2};
        faces->add_element(tri, 1);
        Index tri2[] = {1, 3, 2};
        faces->add_element(tri2, 1);
        
        mesh_->build_topology();

        field_ = std::make_unique<FieldSpace>("potential", mesh_.get(), 1, 1);
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FieldSpace> field_;
};

TEST_F(PoissonTest, SolveSimpleProblem) {
    // Assemble stiffness matrix
    BilinearForm bilinear(field_.get());
    SparseMatrix K;
    bilinear.assemble(BilinearForms::laplacian(1.0), K);
    
    ASSERT_GT(K.nonZeros(), 0);
    
    // Assemble load vector (no source)
    LinearForm linear(field_.get());
    DynamicVector F;
    linear.assemble(LinearForms::source(0.0), F);
    
    // Apply Dirichlet BC: u = 0 on boundary 1
    field_->add_dirichlet_bc(1, 0.0);
    
    // Apply BC to system
    field_->apply_bcs_to_system(K, F);
    
    ASSERT_GT(K.nonZeros(), 0);
    
    // Solve
    DirectSolver solver;
    DynamicVector u;
    auto status = solver.solve(K, F, u);
    
    EXPECT_EQ(status, SolverStatus::Success);
    
    // Check solution is finite
    for (int i = 0; i < u.size(); ++i) {
        EXPECT_TRUE(std::isfinite(u[i]));
    }
    
    // Update field solution
    field_->set_solution(u);
}

// ============================================================
// Busbar Assembly Tests
// ============================================================

class BusbarAssemblyTest : public ::testing::Test {
protected:
    void SetUp() override {
        MphtxtReader reader;
        mesh_ = reader.read("cases/busbar/mesh.mphtxt");

        if (!mesh_) {
            GTEST_SKIP() << "Busbar mesh not found";
        }
        
        field_ = std::make_unique<FieldSpace>("electric_potential", mesh_.get(), 1, 1);
    }

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<FieldSpace> field_;
};

TEST_F(BusbarAssemblyTest, AssembleLaplacian) {
    BilinearForm form(field_.get());
    
    SparseMatrix K;
    form.assemble(BilinearForms::laplacian(1.0), K);
    
    EXPECT_EQ(K.rows(), static_cast<Index>(mesh_->num_vertices()));
    EXPECT_EQ(K.cols(), static_cast<Index>(mesh_->num_vertices()));
    
    // Matrix should be symmetric
    SparseMatrix Kt = K.transpose();
    EXPECT_TRUE(K.isApprox(Kt, 1e-10));
    
    // Check for NaN/Inf
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            EXPECT_TRUE(std::isfinite(it.value()));
        }
    }
}

TEST_F(BusbarAssemblyTest, AssembleMassMatrix) {
    BilinearForm form(field_.get());
    
    SparseMatrix M;
    form.assemble(BilinearForms::mass(1.0), M);
    
    EXPECT_EQ(M.rows(), static_cast<Index>(mesh_->num_vertices()));
    
    int n_negative = 0;
    for (int i = 0; i < M.rows(); ++i) {
        if (M.coeff(i, i) < -1e-8) {
            n_negative++;
        }
    }
    EXPECT_EQ(n_negative, 0) << "Found " << n_negative << " significantly negative diagonal entries";
}

TEST_F(BusbarAssemblyTest, WithDomainCoefficients) {
    BilinearForm form(field_.get());
    
    std::unordered_map<Index, Scalar> conductivity;
    conductivity[1] = 5.998e7;      // Copper
    for (Index i = 2; i <= 7; ++i) {
        conductivity[i] = 7.407e5;  // Titanium
    }
    
    SparseMatrix K;
    form.assemble_with_coefficients(BilinearForms::laplacian(1.0), K, conductivity);
    
    EXPECT_EQ(K.rows(), static_cast<Index>(mesh_->num_vertices()));
    EXPECT_GT(K.nonZeros(), 0);
}

TEST_F(BusbarAssemblyTest, ElectrostaticsSetup) {
    // Add boundary conditions
    field_->add_dirichlet_bc(43, 0.02);  // Vtot = 20mV
    field_->add_dirichlet_bc(8, 0.0);
    field_->add_dirichlet_bc(15, 0.0);
    
    EXPECT_GT(field_->n_constrained_dofs(), 0);
    
    // Assemble system
    BilinearForm bilinear(field_.get());
    SparseMatrix K;
    
    std::unordered_map<Index, Scalar> conductivity;
    conductivity[1] = 5.998e7;
    for (Index i = 2; i <= 7; ++i) {
        conductivity[i] = 7.407e5;
    }
    
    bilinear.assemble_with_coefficients(BilinearForms::laplacian(1.0), K, conductivity);
    
    LinearForm linear(field_.get());
    DynamicVector F;
    linear.assemble(LinearForms::source(0.0), F);
    
    // Apply BC
    field_->apply_bcs_to_system(K, F);
    
    // Solve
    DirectSolver solver;
    DynamicVector V;
    solver.solve(K, F, V);
    
    // Check solution
    EXPECT_EQ(V.size(), static_cast<Index>(mesh_->num_vertices()));
    
    // Check for finite values
    for (int i = 0; i < V.size(); ++i) {
        EXPECT_TRUE(std::isfinite(V[i]));
    }
    
    // Solution should be bounded by BC values
    Scalar V_min = V.minCoeff();
    Scalar V_max = V.maxCoeff();
    EXPECT_GE(V_min, -0.01);
    EXPECT_LE(V_max, 0.03);
    
    // Update field solution
    field_->set_solution(V);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}