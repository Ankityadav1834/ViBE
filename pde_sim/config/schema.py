"""
Simulation configuration schema.

One dataclass that captures everything needed to run a simulation:
- Domain/mesh geometry
- Discretization method
- Solver settings
- Time stepping
- Parameters
- Initial conditions
- Boundary conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pde_sim.boundary.conditions import BoundarySet
from pde_sim.solver.time_stepper import TimeStepConfig


@dataclass
class DomainConfig:
    """
    Specification for one mesh domain.

    Parameters
    ----------
    name : str
        Domain identifier.
    regions : list of dict
        Each dict has keys: "name", "length", "n_nodes".
    method : str
        Discretization method: "finite_volume", "finite_difference", "chebyshev".
    """
    name: str = "default"
    regions: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "finite_volume"


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    Parameters
    ----------
    domains : list of DomainConfig
        Mesh domain specifications.
    parameters : dict
        Physical parameters (scalars or tensors).
    initial_conditions : dict
        Maps field name → initial value (scalar, tensor, or callable).
    boundary_conditions : dict
        Maps field name → BoundarySet.
    time : TimeStepConfig
        Time stepping configuration.
    solver : dict
        Solver settings (tol, max_iter, damping, etc.).
    output : dict
        Output settings (filename, save_interval, derived quantities).
    device : str
        "cpu" or "cuda".
    dtype : str
        "float32" or "float64".
    """
    domains: List[DomainConfig] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    time: TimeStepConfig = field(default_factory=TimeStepConfig)
    solver: Dict[str, Any] = field(default_factory=lambda: {
        "tol": 1e-6,
        "max_iter": 15,
        "damping": 1e-12,
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        "filename": "results.csv",
        "save_interval": 1,
    })
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def t_end(self) -> float:
        return self.time.dt_max * 10000  # fallback; should be set by user

    def add_domain(
        self,
        name: str,
        regions: List[Dict[str, Any]],
        method: str = "finite_volume",
    ) -> "SimulationConfig":
        """Fluent API: add a domain config."""
        self.domains.append(DomainConfig(name, regions, method))
        return self
