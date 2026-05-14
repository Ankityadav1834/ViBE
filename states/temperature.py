from states.registry import StateEquationSpec

def temperature_rhs(physics, y_flat, I_app, p, context):
    """
    Temperature ODE: dT/dt = heat_generation / (rho_Cp * Vol_cell)
    
    The heat generation is calculated in `electrochemistry.py` (which runs once 
    per step) because it relies on the same overpotentials and OCVs that the 
    diffusion PDEs use.
    """
    return context["dT_dt"]

spec = StateEquationSpec(
    order=1000, 
    size=1, 
    initial="T_amb", 
    scale=300.0, 
    nonnegative=True,
    
    equation="dT/dt = I_app * (OCV - V_cell) / (rho_Cp * Vol_cell)",
    rhs=temperature_rhs, 
)
