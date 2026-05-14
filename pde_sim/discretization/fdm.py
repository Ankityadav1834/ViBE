"""
Finite Difference Method (FDM) discretization backend.

Uses node-centered values with second-order centered differences
for interior points and one-sided differences at boundaries.
"""

from __future__ import annotations

import torch

from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.mesh.mesh import CompositeMesh1D


class FiniteDifferenceBackend(DiscretizationBackend):
    """
    1D Finite Difference discretization on a node-centered mesh.

    Gradient:   nodes(N) → nodes(N)
    Divergence: nodes(N) → nodes(N)

    Uses central differences for interior and one-sided for boundaries.
    """

    def __init__(self, mesh: CompositeMesh1D):
        if mesh.method != "finite_difference":
            raise ValueError(
                "FiniteDifferenceBackend requires a finite_difference mesh."
            )
        super().__init__(mesh)
        self._gradient_matrix = self._build_gradient_matrix()
        self._divergence_matrix = self._gradient_matrix.clone()

    def _build_gradient_matrix(self) -> torch.Tensor:
        n = self.mesh.n_nodes
        G = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        x = self.mesh.nodes

        if n > 1:
            # Interior: central difference
            for i in range(1, n - 1):
                hl = x[i] - x[i - 1]
                hr = x[i + 1] - x[i]
                G[i, i - 1] = -hr / (hl * (hl + hr))
                G[i, i] = (hr - hl) / (hl * hr)
                G[i, i + 1] = hl / (hr * (hl + hr))

            # Left boundary: forward difference
            h = x[1] - x[0]
            G[0, 0] = -1.0 / h
            G[0, 1] = 1.0 / h

            # Right boundary: backward difference
            h = x[-1] - x[-2]
            G[-1, -2] = -1.0 / h
            G[-1, -1] = 1.0 / h

        return G

    def gradient(self, values: torch.Tensor) -> torch.Tensor:
        return self._gradient_matrix @ values

    def divergence(self, values: torch.Tensor) -> torch.Tensor:
        return self._divergence_matrix @ values
