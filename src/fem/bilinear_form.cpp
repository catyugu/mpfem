#include "mpfem/fem/bilinear_form.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/core/logger.hpp"
#include <algorithm>
#include <set>

namespace mpfem {

// ============================================================================
// BilinearForm implementation
// ============================================================================

BilinearForm::BilinearForm(FiniteElementSpace* fes) : fes_(fes) {}

void BilinearForm::AddDomainIntegrator(
    std::unique_ptr<BilinearFormIntegrator> integrator,
    const std::vector<int>& domain_ids) {
  DomainIntegrator di;
  di.integrator = std::move(integrator);
  di.domain_ids = domain_ids;
  domain_integrators_.push_back(std::move(di));
}

void BilinearForm::AddBoundaryIntegrator(
    std::unique_ptr<BilinearFormIntegrator> integrator,
    const std::vector<int>& boundary_ids) {
  BoundaryIntegrator bi;
  bi.integrator = std::move(integrator);
  bi.boundary_ids = boundary_ids;
  boundary_integrators_.push_back(std::move(bi));
}

void BilinearForm::Assemble() {
  const Mesh* mesh = fes_->GetMesh();
  int ndofs = fes_->GetNDofs() * fes_->GetVDim();

  // Use triplet list for sparse matrix construction
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(ndofs * 10);  // Estimate: ~10 non-zeros per row

  ElementTransformation T;
  T.SetMesh(mesh);

  // Assemble domain integrators
  const auto& domain_groups = mesh->DomainElements();

  for (const auto& di : domain_integrators_) {
    for (size_t g = 0; g < domain_groups.size(); ++g) {
      const auto& group = domain_groups[g];
      const FiniteElement* fe = fes_->GetFE(group.type);
      if (!fe) continue;

      for (int e = 0; e < group.Count(); ++e) {
        int domain_id = group.entity_ids[e];

        // Check if integrator applies to this domain
        bool applies = di.domain_ids.empty() ||
                       std::find(di.domain_ids.begin(), di.domain_ids.end(),
                                 domain_id) != di.domain_ids.end();
        if (!applies) continue;

        // Get element DOFs
        std::vector<int> dofs;
        fes_->GetElementDofs(group, e, dofs);

        // Set up transformation
        T.SetElement(&group, e);

        // Assemble element matrix
        Eigen::MatrixXd elmat;
        di.integrator->AssembleElementMatrix(*fe, T, elmat);

        // Add to global matrix
        for (int i = 0; i < elmat.rows(); ++i) {
          for (int j = 0; j < elmat.cols(); ++j) {
            if (std::abs(elmat(i, j)) > 1e-16) {
              triplets.push_back(
                  Eigen::Triplet<double>(dofs[i], dofs[j], elmat(i, j)));
            }
          }
        }
      }
    }
  }

  // Assemble boundary integrators
  const auto& bdr_groups = mesh->BoundaryElements();

  for (const auto& bi : boundary_integrators_) {
    for (size_t g = 0; g < bdr_groups.size(); ++g) {
      const auto& group = bdr_groups[g];
      const FiniteElement* fe = fes_->GetFE(group.type);
      if (!fe) continue;

      // Only process external boundaries
      for (int e = 0; e < group.Count(); ++e) {
        int bdr_id = group.entity_ids[e];

        // Check if this is an external boundary
        if (!mesh->IsExternalBoundary(bdr_id)) continue;

        // Check if integrator applies to this boundary
        bool applies = bi.boundary_ids.empty() ||
                       std::find(bi.boundary_ids.begin(), bi.boundary_ids.end(),
                                 bdr_id) != bi.boundary_ids.end();
        if (!applies) continue;

        // Get boundary element DOFs
        std::vector<int> dofs;
        fes_->GetElementDofs(group, e, dofs);

        // Set up transformation
        T.SetElement(&group, e);

        // Assemble boundary element matrix
        Eigen::MatrixXd elmat;
        bi.integrator->AssembleFaceMatrix(*fe, T, elmat);

        // Add to global matrix
        for (int i = 0; i < elmat.rows(); ++i) {
          for (int j = 0; j < elmat.cols(); ++j) {
            if (std::abs(elmat(i, j)) > 1e-16) {
              triplets.push_back(
                  Eigen::Triplet<double>(dofs[i], dofs[j], elmat(i, j)));
            }
          }
        }
      }
    }
  }

  // Build sparse matrix
  mat_full_.resize(ndofs, ndofs);
  mat_full_.setFromTriplets(triplets.begin(), triplets.end());
  mat_full_.makeCompressed();

  // Copy to mat_ (will be modified by FormLinearSystem)
  mat_ = mat_full_;

  MPFEM_INFO("BilinearForm assembled: %dx%d matrix with %d non-zeros",
             static_cast<int>(mat_.rows()), static_cast<int>(mat_.cols()),
             static_cast<int>(mat_.nonZeros()));
}

void BilinearForm::FormLinearSystem(const std::vector<int>& ess_dofs,
                                     VectorXd& x, VectorXd& b,
                                     SparseMatrix& A, VectorXd& X, VectorXd& B) {
  int ndofs = fes_->GetNDofs() * fes_->GetVDim();
  int n_ess = static_cast<int>(ess_dofs.size());
  int n_free = ndofs - n_ess;

  // Build the free DOF map
  std::vector<int> free_dofs;
  std::vector<bool> is_essential(ndofs, false);

  for (int dof : ess_dofs) {
    if (dof >= 0 && dof < ndofs) {
      is_essential[dof] = true;
    }
  }

  for (int i = 0; i < ndofs; ++i) {
    if (!is_essential[i]) {
      free_dofs.push_back(i);
    }
  }

  n_free = static_cast<int>(free_dofs.size());

  // Create the reduced system
  // A = mat_(free, free)
  // B = b(free) - mat_(free, ess) * x(ess)

  // Build reduced matrix using triplets
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(mat_.nonZeros());

  for (int k = 0; k < mat_.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(mat_, k); it; ++it) {
      int row = it.row();
      int col = it.col();

      // Only include free DOFs
      if (is_essential[row] || is_essential[col]) continue;

      // Map to reduced indices
      int red_row = std::lower_bound(free_dofs.begin(), free_dofs.end(), row) -
                    free_dofs.begin();
      int red_col = std::lower_bound(free_dofs.begin(), free_dofs.end(), col) -
                    free_dofs.begin();

      triplets.push_back(Eigen::Triplet<double>(red_row, red_col, it.value()));
    }
  }

  A.resize(n_free, n_free);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();

  // Build RHS vector
  B.resize(n_free);

  for (int i = 0; i < n_free; ++i) {
    int dof = free_dofs[i];
    B(i) = b(dof);

    // Subtract contribution from essential DOFs
    for (SparseMatrix::InnerIterator it(mat_, dof); it; ++it) {
      int col = it.col();
      if (is_essential[col]) {
        B(i) -= it.value() * x(col);
      }
    }
  }

  // Build solution vector
  X.resize(n_free);
  for (int i = 0; i < n_free; ++i) {
    X(i) = x(free_dofs[i]);
  }

  MPFEM_INFO("FormLinearSystem: %d free DOFs, %d essential DOFs", n_free, n_ess);
}

// ============================================================================
// LinearForm implementation
// ============================================================================

LinearForm::LinearForm(FiniteElementSpace* fes) : fes_(fes) {
  vec_.resize(fes->GetVSize());
  vec_.setZero();
}

void LinearForm::AddDomainIntegrator(
    std::unique_ptr<LinearFormIntegrator> integrator,
    const std::vector<int>& domain_ids) {
  DomainLFIntegrator di;
  di.integrator = std::move(integrator);
  di.domain_ids = domain_ids;
  domain_integrators_.push_back(std::move(di));
}

void LinearForm::AddBoundaryIntegrator(
    std::unique_ptr<LinearFormIntegrator> integrator,
    const std::vector<int>& boundary_ids) {
  BoundaryLFIntegrator bi;
  bi.integrator = std::move(integrator);
  bi.boundary_ids = boundary_ids;
  boundary_integrators_.push_back(std::move(bi));
}

void LinearForm::Assemble() {
  const Mesh* mesh = fes_->GetMesh();
  vec_.setZero();

  ElementTransformation T;
  T.SetMesh(mesh);

  // Assemble domain integrators
  const auto& domain_groups = mesh->DomainElements();

  for (const auto& di : domain_integrators_) {
    for (size_t g = 0; g < domain_groups.size(); ++g) {
      const auto& group = domain_groups[g];
      const FiniteElement* fe = fes_->GetFE(group.type);
      if (!fe) continue;

      for (int e = 0; e < group.Count(); ++e) {
        int domain_id = group.entity_ids[e];

        bool applies = di.domain_ids.empty() ||
                       std::find(di.domain_ids.begin(), di.domain_ids.end(),
                                 domain_id) != di.domain_ids.end();
        if (!applies) continue;

        std::vector<int> dofs;
        fes_->GetElementDofs(group, e, dofs);

        T.SetElement(&group, e);

        Eigen::VectorXd elvec;
        di.integrator->AssembleElementVector(*fe, T, elvec);

        // Add to global vector
        for (int i = 0; i < elvec.size(); ++i) {
          vec_(dofs[i]) += elvec(i);
        }
      }
    }
  }

  // Assemble boundary integrators
  const auto& bdr_groups = mesh->BoundaryElements();

  for (const auto& bi : boundary_integrators_) {
    for (size_t g = 0; g < bdr_groups.size(); ++g) {
      const auto& group = bdr_groups[g];
      const FiniteElement* fe = fes_->GetFE(group.type);
      if (!fe) continue;

      for (int e = 0; e < group.Count(); ++e) {
        int bdr_id = group.entity_ids[e];

        // Only process external boundaries
        if (!mesh->IsExternalBoundary(bdr_id)) continue;

        bool applies = bi.boundary_ids.empty() ||
                       std::find(bi.boundary_ids.begin(), bi.boundary_ids.end(),
                                 bdr_id) != bi.boundary_ids.end();
        if (!applies) continue;

        std::vector<int> dofs;
        fes_->GetElementDofs(group, e, dofs);

        T.SetElement(&group, e);

        Eigen::VectorXd elvec;
        bi.integrator->AssembleBoundaryVector(*fe, T, elvec);

        for (int i = 0; i < elvec.size(); ++i) {
          vec_(dofs[i]) += elvec(i);
        }
      }
    }
  }

  MPFEM_INFO("LinearForm assembled: vector of size {}", vec_.size());
}

}  // namespace mpfem
