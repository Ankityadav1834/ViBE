"""
Example 1: 1D Heat Equation
============================

    ∂T/∂t = α ∇²T

Demonstrates the simplest possible usage of the pde_sim framework:
- One field, one equation
- Dirichlet boundary conditions
- Finite volume discretization
- Adaptive time stepping
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pde_sim import *

# ═══════════════════════════════════════════════════════════════════════════
#  1. Define the PDE symbolically
# ═══════════════════════════════════════════════════════════════════════════

T = Field("T", domain="rod", size=50)
alpha = Param("alpha")

equations = System({
    "T": Dt(T) == Div(alpha * Grad(T)),
}, metadata={
    "T": {
        "initial_condition": 300.0,
        "scale": 300.0,
        "nonnegative": True,
        "notes": "Temperature field on a 1D rod",
    }
})

print("Equation system:")
print(equations)
print()

# ═══════════════════════════════════════════════════════════════════════════
#  2. Configure the simulation
# ═══════════════════════════════════════════════════════════════════════════

config = SimulationConfig(
    domains=[
        DomainConfig(
            name="rod",
            regions=[{"name": "rod", "length": 1.0, "n_nodes": 50}],
            method="finite_volume",
        ),
    ],
    parameters={
        "alpha": 0.01,       # Thermal diffusivity [m²/s]
    },
    initial_conditions={
        "T": 300.0,          # Uniform initial temperature [K]
    },
    boundary_conditions={
        "T": BoundarySet(
            DirichletBC(400.0, "left"),   # Hot end
            DirichletBC(300.0, "right"),  # Cold end
        ),
    },
    time=TimeStepConfig(
        dt_init=0.01,
        dt_min=1e-6,
        dt_max=1.0,
    ),
    solver={"tol": 1e-8, "max_iter": 20},
    output={"filename": "heat_equation_results.csv"},
    device="cpu",
    dtype="float64",
)

# ═══════════════════════════════════════════════════════════════════════════
#  3. Run
# ═══════════════════════════════════════════════════════════════════════════

sim = Simulation(equations, config)
result = sim.run(t_end=10.0, print_interval=20)

print(f"\nFinal state at t={result['times'][-1]:.2f}:")
final = result["states"][-1]
T_final = sim.layout.get(final, "T")
print(f"  T_left  = {T_final[0]:.2f} K")
print(f"  T_mid   = {T_final[25]:.2f} K")
print(f"  T_right = {T_final[-1]:.2f} K")
