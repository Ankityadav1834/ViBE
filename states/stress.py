from states.registry import StateEquationSpec, option_value, physics_attr
from pde_framework import OperatorEquationSpec, BoundaryCondition, Div, Grad, Parameter, Variable
import torch

def stress_enabled(config): 
    return bool(config.get("stress_options", {}).get("enabled", True))

def stress_flux_coefficient(physics, y_flat, I_app, p, context):
    options = physics.state_equation_options.get("stress", {})
    stress = physics.state(y_flat, "stress")
    diffusivity = torch.ones_like(stress) * float(options.get("diffusivity", 1e-12))
    return physics.through_cell_flux_coefficient(diffusivity)

def stress_source(physics, y_flat, I_app, p, context):
    options = physics.state_equation_options.get("stress", {})
    ce_state = context["ce_real"]
    
    relaxation = float(options.get("relaxation", 1e-4))
    coupling = float(options.get("coupling", 1.0))
    stress = physics.state(y_flat, "stress")
    
    # Source is proportional to concentration deviation, minus relaxation
    return coupling * (ce_state - p["ce_0"]) - relaxation * stress

spec = StateEquationSpec(
    order=40, 
    size=physics_attr("Nel"), 
    initial=option_value("initial", 0.0), 
    scale=option_value("scale", 1e6), 
    nonnegative=False,
    enabled=stress_enabled, 
    options_key="stress_options",
    dependencies=("electrolyte",),
    
    equation="dstress/dt = Div(D_stress * Grad(stress)) + coupling*(ce - ce_0) - relaxation*stress",
    
    operator=OperatorEquationSpec(
        state_name="stress",
        variable_name="stress",
        domain="through_cell",
        evaluator="conservative",
        method="config['electrolyte_spatial_method']",
        
        # Explicit Mathematics: Div(D_stress * Grad(stress)) + Source
        rhs=Div(Parameter("flux_coefficient") * Grad(Variable("stress"))) + Parameter("source"),
        flux=Parameter("flux_coefficient") * Grad(Variable("stress")),
        source=Parameter("source"),
        
        parameters={
            "flux_coefficient": stress_flux_coefficient, 
            "source": stress_source
        },
        boundary_conditions={
            "left": BoundaryCondition("neumann", 0.0),
            "right": BoundaryCondition("neumann", 0.0),
        },
    )
)
