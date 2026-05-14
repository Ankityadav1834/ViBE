"""
Chebyshev / Spectral discretization backend.

Region-wise Chebyshev-Lobatto collocation operators. Each region in the
composite mesh gets its own differentiation matrix, and the global
gradient/divergence operators are block-diagonal over regions.
"""

from __future__ import annotations

from typing import Dict

import torch

from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.mesh.mesh import CompositeMesh1D


class ChebyshevBackend(DiscretizationBackend):
    """
    Region-wise Chebyshev collocation operators.

    For each region in the composite mesh, a Chebyshev-Lobatto
    differentiation matrix is computed on [0, L] and scaled to
    the physical region length.

    Gradient and divergence are block-diagonal over regions.
    """

    def __init__(self, mesh: CompositeMesh1D):
        if mesh.method != "chebyshev":
            raise ValueError(
                "ChebyshevBackend requires a chebyshev mesh."
            )
        super().__init__(mesh)

        self.region_D1: Dict[str, torch.Tensor] = {}
        self.region_D2: Dict[str, torch.Tensor] = {}
        self._build_region_matrices()

    def _build_region_matrices(self):
        """Build per-region Chebyshev differentiation matrices."""
        for name in self.mesh.region_names:
            s = self.mesh.region_slices[name]
            n = s.stop - s.start
            length = self.mesh.region_lengths[name]
            _, D_ref, D2_ref = self._cheb_matrices(n)
            self.region_D1[name] = D_ref / length
            self.region_D2[name] = D2_ref / (length ** 2)

    def _cheb_matrices(self, n: int):
        """
        Chebyshev-Lobatto differentiation matrices on [0, 1].

        Returns
        -------
        x_ref : Tensor, shape (n,)
            Reference nodes on [0, 1].
        D : Tensor, shape (n, n)
            First derivative matrix.
        D2 : Tensor, shape (n, n)
            Second derivative matrix (D @ D).
        """
        x_ref = 0.5 * (
            1.0 - torch.cos(
                torch.pi * torch.arange(n, device=self.device, dtype=self.dtype) / (n - 1)
            )
        )

        # Barycentric weights
        w = torch.ones(n, device=self.device, dtype=self.dtype)
        for i in range(n):
            diff = x_ref[i] - x_ref
            diff[i] = 1.0
            w[i] = 1.0 / torch.prod(diff)

        # Differentiation matrix
        D = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x_ref[i] - x_ref[j])
        for i in range(n):
            D[i, i] = -torch.sum(D[i, :]) + D[i, i]

        D2 = D @ D
        return x_ref, D, D2

    def _split(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.mesh.region_values(values)

    def _concat(self, region_vals: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mesh.concat_regions(region_vals)

    def gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Block-diagonal gradient: D1 applied per region."""
        rv = self._split(values)
        return self._concat({
            name: self.region_D1[name] @ rv[name]
            for name in self.mesh.region_names
        })

    def divergence(self, values: torch.Tensor) -> torch.Tensor:
        """
        Block-diagonal divergence: same as gradient in 1D Chebyshev.

        For 1D, Div(f) = df/dx, same operator as Grad.
        """
        rv = self._split(values)
        return self._concat({
            name: self.region_D1[name] @ rv[name]
            for name in self.mesh.region_names
        })

    def second_derivative(self, values: torch.Tensor) -> torch.Tensor:
        """Apply D2 per region."""
        rv = self._split(values)
        return self._concat({
            name: self.region_D2[name] @ rv[name]
            for name in self.mesh.region_names
        })
