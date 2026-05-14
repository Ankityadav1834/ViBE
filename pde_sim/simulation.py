"""
Simulation — the top-level orchestrator that wires everything together.

Usage
-----
```python
from pde_sim import *

# 1. Define equations
c = Field("c", domain="channel")
D = Param("D")
equations = System({"c": Dt(c) == Div(D * Grad(c))})

# 2. Configure
config = SimulationConfig(...)

# 3. Run
sim = Simulation(equations, config)
result = sim.run(t_end=100.0)
```
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import time as _time

import torch

from pde_sim.symbolic.equation import System, PDEEquation, AlgebraicEquation
from pde_sim.mesh.domain import CompositeDomain, Region
from pde_sim.mesh.mesh import CompositeMesh1D
from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.discretization.fvm import FiniteVolumeBackend
from pde_sim.discretization.fdm import FiniteDifferenceBackend
from pde_sim.discretization.spectral import ChebyshevBackend
from pde_sim.assembly.pipeline import AssemblyPipeline, DiscretizationContext
from pde_sim.state.layout import StateLayout
from pde_sim.solver.implicit import ImplicitSolver
from pde_sim.solver.time_stepper import AdaptiveTimeStepper, TimeStepConfig
from pde_sim.output.manager import OutputManager
from pde_sim.config.schema import SimulationConfig, DomainConfig
from pde_sim.boundary.conditions import BoundarySet


_BACKEND_MAP = {
    "finite_volume": FiniteVolumeBackend,
    "finite_difference": FiniteDifferenceBackend,
    "chebyshev": ChebyshevBackend,
}


class Simulation:
    """
    Top-level simulation orchestrator.

    Wires together: System → Meshes → Backends → Assembly → Solver → Output.

    Parameters
    ----------
    system : System
        The coupled PDE/algebraic equation system.
    config : SimulationConfig
        Domain, solver, time stepping, and parameter configuration.
    """

    def __init__(self, system: System, config: SimulationConfig):
        self.system = system
        self.config = config

        # Resolve device and dtype
        self.device = torch.device(config.device)
        self.dtype = torch.float64 if config.dtype == "float64" else torch.float32

        # Build meshes and backends
        self.meshes: Dict[str, CompositeMesh1D] = {}
        self.backends: Dict[str, DiscretizationBackend] = {}
        self._build_meshes_and_backends()

        # Build state layout
        self.layout = StateLayout()
        self._build_state_layout()

        # Build assembly pipeline
        self.pipeline = AssemblyPipeline(system, self.meshes, self.backends)

        # Build output manager
        self.output = OutputManager(self.layout)

        # Build solver (deferred — needs the RHS function)
        self._solver: Optional[ImplicitSolver] = None
        self._stepper: Optional[AdaptiveTimeStepper] = None

    def _build_meshes_and_backends(self):
        """Create meshes and discretization backends from config."""
        for domain_cfg in self.config.domains:
            domain = CompositeDomain(coordinate="x")
            for r in domain_cfg.regions:
                domain.add_region(r["name"], r["length"], r["n_nodes"])

            mesh = CompositeMesh1D(
                domain, method=domain_cfg.method,
                device=self.device, dtype=self.dtype,
            )
            self.meshes[domain_cfg.name] = mesh

            backend_cls = _BACKEND_MAP.get(domain_cfg.method)
            if backend_cls is None:
                raise ValueError(
                    f"Unknown discretization method: {domain_cfg.method!r}. "
                    f"Supported: {list(_BACKEND_MAP.keys())}"
                )
            self.backends[domain_cfg.name] = backend_cls(mesh)

        # Ensure at least a "default" mesh
        if not self.meshes:
            raise ValueError("No domains configured. Add at least one DomainConfig.")

    def _build_state_layout(self):
        """Register all fields in the state layout."""
        ic = self.config.initial_conditions

        for name in self.system.names:
            if name in self.system.pdes:
                eq = self.system.pdes[name]
                field = eq.field
            else:
                eq = self.system.algebraic[name]
                field = eq.field

            # Determine size
            domain_name = getattr(eq, 'domain', 'default')
            if field.size is not None:
                size = field.size
            elif domain_name in self.meshes:
                size = self.meshes[domain_name].n_nodes
            else:
                size = list(self.meshes.values())[0].n_nodes

            initial = ic.get(name, getattr(eq, 'initial_condition', 0.0))
            scale = getattr(eq, 'scale', 1.0)
            nonneg = getattr(eq, 'nonnegative', False)

            self.layout.register(name, size, initial=initial, scale=scale, nonnegative=nonneg)

    def _rhs_fn(
        self,
        y_flat: torch.Tensor,
        params: Dict[str, Any],
        y_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Compute the full RHS vector for the coupled system.

        This is the function passed to the Newton solver.
        """
        fields = self.layout.unpack(y_flat)

        # Build boundary conditions dict
        bc_dict = {}
        for name, bc_config in self.config.boundary_conditions.items():
            if isinstance(bc_config, BoundarySet):
                bc_dict[name] = bc_config

        y_old_dict = self.layout.unpack(y_old) if y_old is not None else None

        # Evaluate all RHS via the assembly pipeline
        rhs_dict = self.pipeline.evaluate_all(
            fields=fields,
            params=params,
            boundary_conditions=bc_dict,
            y_old=y_old_dict,
            dt=dt,
        )

        return self.layout.pack(rhs_dict, device=self.device, dtype=self.dtype)

    @property
    def solver(self) -> ImplicitSolver:
        """Lazy-build the solver on first access."""
        if self._solver is None:
            solver_cfg = self.config.solver
            self._solver = ImplicitSolver(
                state_layout=self.layout,
                rhs_fn=self._rhs_fn,
                device=self.device,
                dtype=self.dtype,
                tol=solver_cfg.get("tol", 1e-6),
                max_iter=solver_cfg.get("max_iter", 15),
                damping=solver_cfg.get("damping", 1e-12),
            )
        return self._solver

    @property
    def stepper(self) -> AdaptiveTimeStepper:
        """Lazy-build the time stepper."""
        if self._stepper is None:
            self._stepper = AdaptiveTimeStepper(self.solver, self.config.time)
        return self._stepper

    def run(
        self,
        t_end: float,
        callback: Optional[Callable] = None,
        print_interval: int = 10,
    ) -> dict:
        """
        Run the simulation from t=0 to t=t_end.

        Parameters
        ----------
        t_end : float
            Final simulation time.
        callback : callable or None
            Called after each accepted step: callback(step, t, y, dt, error).
            Return True to stop early.
        print_interval : int
            Print progress every N accepted steps.

        Returns
        -------
        result : dict
            Keys: "times", "states", "errors", "dts", "n_steps", "n_rejected".
        """
        # Build initial state
        y0 = self.layout.initial_state(
            params=self.config.parameters,
            device=self.device,
            dtype=self.dtype,
        )

        print(f"Starting simulation on {self.device}")
        print(f"  Equations: {self.system.names}")
        print(f"  State size: {self.layout.total_size}")
        print(f"  t_end: {t_end}")
        start = _time.time()

        def combined_callback(step, t, y, dt, error):
            if step % print_interval == 0:
                print(
                    f"  Step {step:5d} | t={t:.4g} | dt={dt:.4g} | err={error:.3g}"
                )
            if callback:
                return callback(step, t, y, dt, error)
            return False

        result = self.stepper.integrate(
            y0=y0,
            t_start=0.0,
            t_end=t_end,
            params=self.config.parameters,
            callback=combined_callback,
        )

        elapsed = _time.time() - start
        print(f"Simulation complete in {elapsed:.2f}s")
        print(f"  Steps: {result['n_steps']} accepted, {result['n_rejected']} rejected")

        # Save results
        filename = self.config.output.get("filename", "results.csv")
        self.output.save(
            result["times"], result["states"],
            self.config.parameters, filename,
        )
        print(f"  Saved: {filename}")

        return result
