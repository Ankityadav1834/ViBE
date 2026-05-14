from states.registry import StateEquationSpec, electrolyte_initial, physics_attr, context_item, state_item
from pde_framework import OperatorEquationSpec, BoundaryCondition, Div, Grad, Parameter, Variable

def electrolyte_source(physics, y_flat, I_app, p, context):
    return context['src_state']

def electrolyte_flux_coeff(physics, y_flat, I_app, p, context):
    if physics.discretization_methods['electrolyte'] == 'finite_volume':
        return lambda ctx, coeff=context['deff_state']: ctx.operators.face_coefficients(coeff)
    return context['deff_state']

spec = StateEquationSpec(
    order=30, 
    size=physics_attr("Nel"), 
    initial=electrolyte_initial, 
    scale=1000.0, 
    nonnegative=True,
    dependencies=("cs_n", "cs_p"),
    
    equation="delectrolyte/dt = Div(Deff * Grad(ce)) + reaction_source",
    
    operator=OperatorEquationSpec(
        state_name="electrolyte",
        variable_name="ce",
        domain="through_cell",
        evaluator="conservative",
        method="config['electrolyte_spatial_method']",
        
        # Explicit Mathematics: Div(Deff * Grad(ce)) + Source
        rhs=Div(Parameter("flux_coefficient") * Grad(Variable("ce"))) + Parameter("source"),
        flux=Parameter("flux_coefficient") * Grad(Variable("ce")),
        source=Parameter("source"),
        
        values=context_item("ce_state"), 
        time_values=state_item("electrolyte"),
        parameters={
            "flux_coefficient": electrolyte_flux_coeff, 
            "source": electrolyte_source
        },
        boundary_conditions={
            "left": BoundaryCondition("neumann", 0.0),
            "right": BoundaryCondition("neumann", 0.0),
        },
    )
)
