"""
Boundary condition definitions.

All BC types store a ``value`` that can be:
- A float/int literal
- A PyTorch tensor
- A callable ``(context) -> tensor`` for state-dependent BCs

Supported types:
- DirichletBC: u = g  on the boundary
- NeumannBC:  du/dn = g  on the boundary
- RobinBC:    a*u + b*du/dn = g  on the boundary
- CustomBC:   user-defined residual function
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch


@dataclass
class DirichletBC:
    """
    Dirichlet boundary condition:  u(boundary) = value.

    Parameters
    ----------
    value : float, Tensor, or callable(context) -> Tensor
        The prescribed boundary value.
    location : str
        Boundary identifier (e.g. "left", "right", "surface").
    """
    value: Any = 0.0
    location: str = "left"

    @property
    def kind(self) -> str:
        return "dirichlet"

    def evaluate(self, context=None) -> Any:
        if callable(self.value):
            return self.value(context)
        return self.value


@dataclass
class NeumannBC:
    """
    Neumann boundary condition:  du/dn(boundary) = value.

    Parameters
    ----------
    value : float, Tensor, or callable(context) -> Tensor
        The prescribed normal derivative (flux).
    location : str
        Boundary identifier.
    """
    value: Any = 0.0
    location: str = "left"

    @property
    def kind(self) -> str:
        return "neumann"

    def evaluate(self, context=None) -> Any:
        if callable(self.value):
            return self.value(context)
        return self.value


@dataclass
class RobinBC:
    """
    Robin boundary condition:  alpha * u + beta * du/dn = value.

    Parameters
    ----------
    alpha : float
        Coefficient on the field value.
    beta : float
        Coefficient on the normal derivative.
    value : float, Tensor, or callable(context) -> Tensor
        The prescribed combined value.
    location : str
        Boundary identifier.
    """
    alpha: float = 1.0
    beta: float = 1.0
    value: Any = 0.0
    location: str = "left"

    @property
    def kind(self) -> str:
        return "robin"

    def evaluate(self, context=None) -> Any:
        if callable(self.value):
            return self.value(context)
        return self.value


@dataclass
class CustomBC:
    """
    Custom boundary condition defined by a residual function.

    Parameters
    ----------
    residual_fn : callable(values, gradient, context) -> Tensor
        Returns the residual that should be driven to zero.
    location : str
        Boundary identifier.
    """
    residual_fn: Callable = None
    location: str = "left"

    @property
    def kind(self) -> str:
        return "custom"

    def evaluate(self, values, gradient, context=None):
        return self.residual_fn(values, gradient, context)


class BoundarySet:
    """
    A collection of boundary conditions for one field.

    Usage
    -----
    ```python
    bc = BoundarySet()
    bc.add(NeumannBC(0.0, "left"))
    bc.add(DirichletBC(1.0, "right"))
    ```
    """

    def __init__(self, *conditions):
        self._conditions: Dict[str, Any] = {}
        for bc in conditions:
            self.add(bc)

    def add(self, bc):
        """Register a BC by its location."""
        self._conditions[bc.location] = bc
        return self

    def get(self, location: str, default=None):
        """Retrieve a BC by location name."""
        return self._conditions.get(location, default)

    def locations(self):
        return list(self._conditions.keys())

    def items(self):
        return self._conditions.items()

    def __contains__(self, location):
        return location in self._conditions

    def __len__(self):
        return len(self._conditions)

    def __repr__(self):
        parts = ", ".join(
            f"{loc}: {bc.kind}" for loc, bc in self._conditions.items()
        )
        return f"BoundarySet({{{parts}}})"
