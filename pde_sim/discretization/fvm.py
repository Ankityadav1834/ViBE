"""
Finite Volume Method (FVM) discretization backend.

Uses cell-centered values with face-based gradient and cell-based divergence.
Harmonic mean interpolation for face coefficients.
"""

from __future__ import annotations

import torch

from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.mesh.mesh import CompositeMesh1D


class FiniteVolumeBackend(DiscretizationBackend):
    """
    1D Finite Volume discretization on a cell-centered mesh.

    Gradient:   cells(N) → faces(N+1)
    Divergence: faces(N+1) → cells(N)

    Boundary faces (0, -1) are padded with zero gradient by default.
    """

    def __init__(self, mesh: CompositeMesh1D):
        if mesh.method != "finite_volume":
            raise ValueError(
                "FiniteVolumeBackend requires a finite_volume mesh."
            )
        super().__init__(mesh)
        self._gradient_matrix = self._build_gradient_matrix()
        self._divergence_matrix = self._build_divergence_matrix()

    def _build_gradient_matrix(self) -> torch.Tensor:
        n_faces = self.mesh.n_faces
        n_nodes = self.mesh.n_nodes
        if n_faces != n_nodes + 1:
            raise ValueError(
                "FVM mesh must provide N+1 faces for N cell centers."
            )
        G = torch.zeros(
            (n_faces, n_nodes), device=self.device, dtype=self.dtype
        )
        x = self.mesh.nodes

        # Interior faces: centered difference
        for f in range(1, n_faces - 1):
            left, right = f - 1, f
            spacing = x[right] - x[left]
            G[f, left] = -1.0 / spacing
            G[f, right] = 1.0 / spacing
        # Boundary faces: zero row → zero-flux padding
        return G

    def _build_divergence_matrix(self) -> torch.Tensor:
        n_faces = self.mesh.n_faces
        n_nodes = self.mesh.n_nodes
        if n_faces != n_nodes + 1:
            raise ValueError(
                "FVM mesh must provide N+1 faces for N cell centers."
            )
        D = torch.zeros(
            (n_nodes, n_faces), device=self.device, dtype=self.dtype
        )
        dx = self.mesh.dx
        for i in range(n_nodes):
            D[i, i] = -1.0 / dx[i]
            D[i, i + 1] = 1.0 / dx[i]
        return D

    def gradient(self, values: torch.Tensor) -> torch.Tensor:
        return self._gradient_matrix @ values

    def divergence(self, values: torch.Tensor) -> torch.Tensor:
        return self._divergence_matrix @ values

    def face_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Harmonic mean interpolation to faces."""
        interior = self._harmonic_mean(coefficients)
        return torch.cat([
            coefficients[:1],
            interior,
            coefficients[-1:],
        ], dim=0)

    def apply_coefficient(
        self,
        coefficient: torch.Tensor,
        flux: torch.Tensor,
    ) -> torch.Tensor:
        """
        If coefficient is cell-centered, interpolate to faces first.
        """
        if coefficient.shape == flux.shape:
            return coefficient * flux
        # Cell-centered → face-centered via harmonic mean
        face_coeff = self.face_coefficients(coefficient)
        return face_coeff * flux

    def laplacian_matrix(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Assemble the full Laplacian matrix: D @ diag(coeff_faces) @ G."""
        face_coeff = self.face_coefficients(coefficients)
        return self._divergence_matrix @ (
            face_coeff.unsqueeze(1) * self._gradient_matrix
        )

    @staticmethod
    def _harmonic_mean(c: torch.Tensor) -> torch.Tensor:
        return 2.0 * c[:-1] * c[1:] / (c[:-1] + c[1:] + 1e-20)
