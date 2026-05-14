from states.registry import StateEquationSpec, physics_attr, parameter_item
from pde_framework import OperatorEquationSpec, BoundaryCondition, spherical_diffusion_expression

def cs_p_surface_gradient(physics, y_flat, I_app, p, context):
    return context['flux_p'] / p['Ds_p']

spec = StateEquationSpec(
    order=20, 
    size=physics_attr("Nr_p"), 
    initial=17038.0, 
    scale=30000.0, 
    nonnegative=True,
    
    equation="dcs_p/dt = Ds_p * (d2cs_p/dr2 + 2/r * dcs_p/dr)",
    
    operator=OperatorEquationSpec(
        state_name="cs_p",
        variable_name="cs_p",
        domain="positive_particle",
        evaluator="spherical_particle",
        method="chebyshev",
        rhs=spherical_diffusion_expression("cs_p"),
        parameters={
            "diffusivity": parameter_item("Ds_p"), 
            "particle_radius": parameter_item("Rs_p")
        },
        boundary_conditions={
            "center": BoundaryCondition("neumann", 0.0),
            "surface": BoundaryCondition("neumann", cs_p_surface_gradient),
        },
    )
)
