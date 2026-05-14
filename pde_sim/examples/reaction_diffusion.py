"""
Example 2: Coupled Diffusion-Reaction System
=============================================

    ∂c/∂t = D_c ∇²c  −  k c·v
    ∂v/∂t = D_v ∇²v  −  k c·v

Two species diffusing and reacting with each other (bimolecular reaction).
Demonstrates:
- Coupled nonlinear PDEs
- Nonlinear reaction terms via Param callables
- Multiple fields on the same domain
- Switching discretization method (FVM → Chebyshev) with one config change
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pde_sim import *

# ═══════════════════════════════════════════════════════════════════════════
#  1. Define the PDEs symbolically
# ═══════════════════════════════════════════════════════════════════════════

c = Field("c", domain="channel", size=40)
v = Field("v", domain="channel", size=40)

D_c = Param("D_c")
D_v = Param("D_v")
k   = Param("k")

# Reaction rate as a state-dependent parameter
def reaction_source_c(ctx):
    """Consumption of species c:  -k * c * v"""
    c_val = ctx.get_field("c")
    v_val = ctx.get_field("v")
    k_val = ctx.resolve_param("k")
    return -k_val * c_val * v_val

def reaction_source_v(ctx):
    """Consumption of species v:  -k * c * v"""
    c_val = ctx.get_field("c")
    v_val = ctx.get_field("v")
    k_val = ctx.resolve_param("k")
    return -k_val * c_val * v_val

equations = System({
    "c": Dt(c) == Div(D_c * Grad(c)) + Param("reaction_c"),
    "v": Dt(v) == Div(D_v * Grad(v)) + Param("reaction_v"),
}, metadata={
    "c": {
        "initial_condition": 1.0,
        "scale": 1.0,
        "nonnegative": True,
        "notes": "Concentration of species A",
    },
    "v": {
        "initial_condition": 0.5,
        "scale": 1.0,
        "nonnegative": True,
        "notes": "Concentration of species B",
    },
})

print("Coupled Reaction-Diffusion System:")
print(equations)
print()

# ═══════════════════════════════════════════════════════════════════════════
#  2. Configure  — try changing method to "chebyshev" or "finite_difference"
# ═══════════════════════════════════════════════════════════════════════════

DISCRETIZATION_METHOD = "finite_volume"   # Change this to switch backends

config = SimulationConfig(
    domains=[
        DomainConfig(
            name="channel",
            regions=[{"name": "channel", "length": 1.0, "n_nodes": 40}],
            method=DISCRETIZATION_METHOD,
        ),
    ],
    parameters={
        "D_c": 0.01,
        "D_v": 0.005,
        "k": 0.5,
        "reaction_c": reaction_source_c,   # State-dependent parameter!
        "reaction_v": reaction_source_v,
    },
    initial_conditions={
        "c": 1.0,
        "v": 0.5,
    },
    boundary_conditions={
        "c": BoundarySet(
            NeumannBC(0.0, "left"),    # No-flux walls
            NeumannBC(0.0, "right"),
        ),
        "v": BoundarySet(
            DirichletBC(1.0, "left"),  # Constant supply of v at left
            NeumannBC(0.0, "right"),
        ),
    },
    time=TimeStepConfig(
        dt_init=0.01,
        dt_min=1e-6,
        dt_max=2.0,
    ),
    solver={"tol": 1e-7, "max_iter": 20},
    output={"filename": "reaction_diffusion_results.csv"},
    device="cpu",
    dtype="float64",
)

# ═══════════════════════════════════════════════════════════════════════════
#  3. Run
# ═══════════════════════════════════════════════════════════════════════════

sim = Simulation(equations, config)

# Register a derived output: total species mass
sim.output.register(DerivedQuantity(
    name="total_c",
    fn=lambda fields, params, t: fields["c"].sum(),
    requires=("c",),
    description="Total mass of species c",
))
sim.output.register(DerivedQuantity(
    name="total_v",
    fn=lambda fields, params, t: fields["v"].sum(),
    requires=("v",),
    description="Total mass of species v",
))

result = sim.run(t_end=5.0, print_interval=20)

print(f"\nFinal state at t={result['times'][-1]:.2f}:")
final = result["states"][-1]
c_final = sim.layout.get(final, "c")
v_final = sim.layout.get(final, "v")
print(f"  c: min={c_final.min():.4f}, max={c_final.max():.4f}, mean={c_final.mean():.4f}")
print(f"  v: min={v_final.min():.4f}, max={v_final.max():.4f}, mean={v_final.mean():.4f}")
