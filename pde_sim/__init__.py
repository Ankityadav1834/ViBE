"""
pde_sim — A modular, physics-agnostic PDE simulation framework.

Equation definition style:

    from pde_sim import *

    c   = Field("c", domain="electrolyte")
    D   = Param("diffusivity")

    equations = System({
        "c": Dt(c) == Div(D * Grad(c)),
    })

    config = SimulationConfig(...)
    sim = Simulation(equations, config)
    sim.run()
"""

# ── Symbolic layer ──────────────────────────────────────────────────────────
from pde_sim.symbolic.expressions import (
    Expression,
    Constant,
    Field,
    Param,
    Grad,
    Div,
    Laplacian,
    Dt,
    Curl,
    SphericalDiv,
    Abs,
    Sqrt,
    Exp,
    Log,
    Tanh,
    Sinh,
    Cosh,
    Sign,
    Clamp,
    Conditional,
    ensure_expression,
)
from pde_sim.symbolic.equation import (
    PDEEquation,
    AlgebraicEquation,
    System,
)

# ── Mesh / Domain ──────────────────────────────────────────────────────────
from pde_sim.mesh.domain import Interval, Region, CompositeDomain
from pde_sim.mesh.mesh import Mesh1D, CompositeMesh1D

# ── Boundary conditions ────────────────────────────────────────────────────
from pde_sim.boundary.conditions import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    CustomBC,
    BoundarySet,
)

# ── Discretization backends ────────────────────────────────────────────────
from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.discretization.fvm import FiniteVolumeBackend
from pde_sim.discretization.fdm import FiniteDifferenceBackend
from pde_sim.discretization.spectral import ChebyshevBackend

# ── Assembly ────────────────────────────────────────────────────────────────
from pde_sim.assembly.pipeline import AssemblyPipeline, DiscretizationContext

# ── State management ────────────────────────────────────────────────────────
from pde_sim.state.layout import StateLayout, FieldSpec

# ── Solver ──────────────────────────────────────────────────────────────────
from pde_sim.solver.implicit import ImplicitSolver
from pde_sim.solver.time_stepper import AdaptiveTimeStepper

# ── Output ──────────────────────────────────────────────────────────────────
from pde_sim.output.manager import OutputManager, DerivedQuantity

# ── Config / Top-level ──────────────────────────────────────────────────────
from pde_sim.config.schema import SimulationConfig, DomainConfig
from pde_sim.solver.time_stepper import TimeStepConfig
from pde_sim.simulation import Simulation

__all__ = [
    # Symbolic
    "Expression", "Constant", "Field", "Param",
    "Grad", "Div", "Laplacian", "Dt", "Curl", "SphericalDiv",
    "Abs", "Sqrt", "Exp", "Log", "Tanh", "Sinh", "Cosh",
    "Sign", "Clamp", "Conditional",
    "ensure_expression",
    "PDEEquation", "AlgebraicEquation", "System",
    # Mesh
    "Interval", "Region", "CompositeDomain",
    "Mesh1D", "CompositeMesh1D",
    # BC
    "DirichletBC", "NeumannBC", "RobinBC", "CustomBC", "BoundarySet",
    # Discretization
    "DiscretizationBackend",
    "FiniteVolumeBackend", "FiniteDifferenceBackend", "ChebyshevBackend",
    # Assembly
    "AssemblyPipeline", "DiscretizationContext",
    # State
    "StateLayout", "FieldSpec",
    # Solver
    "ImplicitSolver", "AdaptiveTimeStepper",
    # Output
    "OutputManager", "DerivedQuantity",
    # Config
    "SimulationConfig", "DomainConfig", "TimeStepConfig", "Simulation",
]
