"""
Domain specification — intervals, regions, and composite domains.

Domains define the physical extent of the problem *before* meshing.
They carry no discretization — that happens in the mesh layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Interval:
    """
    A single contiguous interval [x_left, x_right].

    Parameters
    ----------
    x_left : float
        Left boundary coordinate.
    x_right : float
        Right boundary coordinate.
    """
    x_left: float = 0.0
    x_right: float = 1.0

    @property
    def length(self) -> float:
        return self.x_right - self.x_left

    def __post_init__(self):
        if self.x_right <= self.x_left:
            raise ValueError(
                f"Interval must have x_right > x_left, "
                f"got [{self.x_left}, {self.x_right}]."
            )


@dataclass(frozen=True)
class Region:
    """
    A named region of a composite domain.

    Parameters
    ----------
    name : str
        Identifier (e.g. "anode", "separator", "cathode").
    length : float
        Physical extent.
    n_nodes : int
        Number of discretization points.
    """
    name: str
    length: float
    n_nodes: int

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"Region '{self.name}' must have positive length.")
        if self.n_nodes < 2:
            raise ValueError(f"Region '{self.name}' must have ≥ 2 nodes.")


@dataclass
class CompositeDomain:
    """
    An ordered sequence of adjacent regions.

    Used for multi-region 1D domains (e.g. electrode–separator–electrode).

    Parameters
    ----------
    regions : list of Region
        Ordered from left to right.
    coordinate : str
        Coordinate name (e.g. "x", "r").
    """
    regions: List[Region] = field(default_factory=list)
    coordinate: str = "x"

    @property
    def total_length(self) -> float:
        return sum(r.length for r in self.regions)

    @property
    def total_nodes(self) -> int:
        return sum(r.n_nodes for r in self.regions)

    @property
    def region_names(self) -> List[str]:
        return [r.name for r in self.regions]

    def add_region(self, name: str, length: float, n_nodes: int) -> "CompositeDomain":
        """Fluent API: add a region and return self."""
        self.regions.append(Region(name, length, n_nodes))
        return self

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        parts = ", ".join(f"{r.name}({r.n_nodes})" for r in self.regions)
        return f"CompositeDomain([{parts}], L={self.total_length:.4g})"
