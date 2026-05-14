from states.registry import StateEquationSpec, physics_attr
import torch

def sei_enabled(config): 
    return bool(config.get("sei_options", {}).get("enabled", True))

def sei_rhs(physics, y_flat, I_app, p, context):
    """
    Algebraic/Dummy state for SEI nodes if spatial discretization is used.
    Currently hardcoded to 0 change as main dynamics are in Lsei.
    """
    return torch.zeros(physics.Nsei, device=physics.device, dtype=y_flat.dtype)

def lsei_rhs(physics, y_flat, I_app, p, context):
    """
    SEI Thickness ODE: dLsei/dt = i_side * Msei / (2 * F * rho_sei)
    
    The side reaction current `i_side` is calculated in `electrochemistry.py`
    because it couples with stress and OCV.
    """
    return context["dLsei_dt"]

spec_sei = StateEquationSpec(
    order=80, 
    size=physics_attr("Nsei"), 
    initial=0.0, 
    scale=1.0, 
    nonnegative=True,
    enabled=sei_enabled, 
    options_key="sei_options", 
    rhs=sei_rhs, 
    equation="dsei/dt = 0"
)

spec_lsei = StateEquationSpec(
    order=90, 
    size=1, 
    initial="Lsei_0", 
    scale=1e-8, 
    nonnegative=True,
    enabled=sei_enabled, 
    options_key="sei_options", 
    dependencies=("cs_n",),
    rhs=lsei_rhs, 
    equation="dLsei/dt = i_side * Msei / (2 * F * rho_sei)"
)
