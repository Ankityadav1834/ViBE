from states.registry import StateEquationSpec, physics_attr, parameter_item
from pde_framework import OperatorEquationSpec, BoundaryCondition, spherical_diffusion_expression

def cs_n_surface_gradient(physics, y_flat, I_app, p, context):
    # The flux at the surface is calculated in electrochemistry.py
    # because it depends on side reactions and Butler-Volmer kinetics
    return context['flux_n'] / p['Ds_n']

spec = StateEquationSpec(
    order=10, 
    size=physics_attr("Nr_n"), 
    initial=29866.0, 
    scale=30000.0, 
    nonnegative=True,
    
    # Mathematical definition
    equation="dcs_n/dt = Ds_n * (d2cs_n/dr2 + 2/r * dcs_n/dr)",
    
    operator=OperatorEquationSpec(
        state_name="cs_n",
        variable_name="cs_n",
        domain="negative_particle",
        evaluator="spherical_particle",
        method="chebyshev",
        # Using exact spherical Laplacian expression
        rhs=spherical_diffusion_expression("cs_n"),
        parameters={
            "diffusivity": parameter_item("Ds_n"), 
            "particle_radius": parameter_item("Rs_n")
        },
        boundary_conditions={
            "center": BoundaryCondition("neumann", 0.0),
            "surface": BoundaryCondition("neumann", cs_n_surface_gradient),
        },
    )
)
