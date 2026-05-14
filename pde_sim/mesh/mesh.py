"""
1D mesh generation — node coordinates, cell widths, face positions.

Supports uniform, non-uniform, and Chebyshev node distributions.
The mesh is a pure data structure consumed by discretization backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from pde_sim.mesh.domain import CompositeDomain, Region


@dataclass
class Mesh1D:
    """
    A single-region 1D mesh.

    Attributes
    ----------
    nodes : Tensor
        Node coordinates, shape (N,).
    dx : Tensor
        Cell widths, shape (N,).
    faces : Tensor or None
        Face positions for FVM, shape (N+1,) or None.
    n_nodes : int
        Number of nodes.
    length : float
        Domain length.
    """
    nodes: torch.Tensor
    dx: torch.Tensor
    faces: Optional[torch.Tensor] = None

    @property
    def n_nodes(self) -> int:
        return self.nodes.numel()

    @property
    def length(self) -> float:
        return float(self.nodes[-1] - self.nodes[0])

    @property
    def device(self):
        return self.nodes.device

    @property
    def dtype(self):
        return self.nodes.dtype


class CompositeMesh1D:
    """
    A multi-region 1D mesh assembled from a CompositeDomain.

    Supports finite-volume, finite-difference, and Chebyshev node placement.

    Parameters
    ----------
    domain : CompositeDomain
        The physical domain specification.
    method : str
        "finite_volume", "finite_difference", or "chebyshev".
    device : torch.device or None
        Target device.
    dtype : torch.dtype
        Floating-point precision.
    """

    def __init__(
        self,
        domain: CompositeDomain,
        method: str = "finite_volume",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.domain = domain
        self.method = method
        self.device = device
        self.dtype = dtype

        self.region_names: List[str] = domain.region_names
        self.region_slices: Dict[str, slice] = {}
        self.region_nodes: Dict[str, torch.Tensor] = {}
        self.region_dx: Dict[str, torch.Tensor] = {}
        self.region_lengths: Dict[str, float] = {}

        node_chunks = []
        dx_chunks = []
        face_chunks = []
        cursor = 0
        position = 0.0

        for region in domain.regions:
            name = region.name
            length = region.length
            n = region.n_nodes
            self.region_lengths[name] = length

            # Generate local nodes
            if method == "chebyshev":
                idx = torch.arange(n, device=device, dtype=dtype)
                local_nodes = 0.5 * length * (
                    1.0 - torch.cos(torch.pi * idx / (n - 1))
                )
            elif method == "finite_difference":
                local_nodes = torch.linspace(
                    0.0, length, n, device=device, dtype=dtype
                )
            else:  # finite_volume — cell-centered
                edges = torch.linspace(
                    0.0, length, n + 1, device=device, dtype=dtype
                )
                local_nodes = 0.5 * (edges[:-1] + edges[1:])

            # Cell widths
            if method == "chebyshev":
                dl = torch.empty_like(local_nodes)
                dl[0] = local_nodes[1] - local_nodes[0]
                dl[1:] = local_nodes[1:] - local_nodes[:-1]
                dr = torch.empty_like(local_nodes)
                dr[:-1] = local_nodes[1:] - local_nodes[:-1]
                dr[-1] = local_nodes[-1] - local_nodes[-2]
                local_dx = 0.5 * (dl + dr)
            elif method == "finite_difference":
                local_dx = torch.full_like(
                    local_nodes,
                    length / (n - 1) if n > 1 else length,
                )
            else:
                local_dx = torch.full_like(local_nodes, length / n)

            # Face positions (FVM)
            if method == "finite_volume":
                region_faces = torch.linspace(
                    0.0, length, n + 1, device=device, dtype=dtype
                ) + position
                if face_chunks:
                    region_faces = region_faces[1:]  # avoid duplicate at interface
                face_chunks.append(region_faces)

            # Register
            start = cursor
            stop = cursor + n
            self.region_slices[name] = slice(start, stop)
            self.region_nodes[name] = local_nodes + position
            self.region_dx[name] = local_dx

            node_chunks.append(local_nodes + position)
            dx_chunks.append(local_dx)

            cursor = stop
            position += length

        self.nodes = torch.cat(node_chunks)
        self.dx = torch.cat(dx_chunks)
        self.n_nodes = self.nodes.numel()
        self.total_length = position

        if method == "finite_volume":
            self.faces = torch.cat(face_chunks)
            self.n_faces = self.faces.numel()
        else:
            self.faces = None
            self.n_faces = self.n_nodes

    def region_values(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a global tensor by region."""
        return {
            name: values[self.region_slices[name]]
            for name in self.region_names
        }

    def concat_regions(self, region_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate per-region tensors into a global vector."""
        return torch.cat(
            [region_values[name] for name in self.region_names],
            dim=0,
        )

    def __repr__(self):
        parts = ", ".join(
            f"{name}({s.stop - s.start})"
            for name, s in self.region_slices.items()
        )
        return f"CompositeMesh1D([{parts}], method={self.method!r}, N={self.n_nodes})"
