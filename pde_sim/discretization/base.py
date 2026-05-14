"""
Abstract base class for discretization backends.

Every backend must implement gradient, divergence, and boundary application.
The assembly pipeline dispatches to these methods when it encounters
Grad, Div, Laplacian, etc. in the expression tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from pde_sim.mesh.mesh import CompositeMesh1D


class DiscretizationBackend(ABC):
    """
    Abstract 1D discretization backend.

    Concrete implementations (FVM, FDM, Chebyshev) provide the discrete
    operators needed to evaluate spatial derivatives on a given mesh.

    Parameters
    ----------
    mesh : CompositeMesh1D
        The mesh this backend operates on.
    """

    def __init__(self, mesh: CompositeMesh1D):
        self.mesh = mesh
        self.device = mesh.nodes.device
        self.dtype = mesh.nodes.dtype

    @abstractmethod
    def gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Discrete gradient operator: values(N) → gradient(N or N+1)."""
        ...

    @abstractmethod
    def divergence(self, values: torch.Tensor) -> torch.Tensor:
        """Discrete divergence operator: values(N or N+1) → divergence(N)."""
        ...

    def laplacian(
        self,
        values: torch.Tensor,
        coefficient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Discrete Laplacian:  Div(coeff * Grad(values)).

        Default implementation chains gradient + divergence.
        Backends may override for efficiency.
        """
        grad = self.gradient(values)
        if coefficient is not None:
            grad = self.apply_coefficient(coefficient, grad)
        return self.divergence(grad)

    def apply_coefficient(
        self,
        coefficient: torch.Tensor,
        flux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multiply a coefficient field onto a flux/gradient.

        For FVM, this interpolates cell-centered coefficients to faces.
        Default: pointwise multiplication.
        """
        return coefficient * flux

    def face_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Interpolate cell-centered coefficients to face values.

        Default: return as-is (identity for FDM/spectral).
        """
        return coefficients

    @property
    def n_nodes(self) -> int:
        return self.mesh.n_nodes

    def __repr__(self):
        return f"{self.__class__.__name__}(mesh={self.mesh!r})"
