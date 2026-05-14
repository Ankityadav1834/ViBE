from states.registry import StateEquationSpec
import torch

def lithium_plating_rhs(physics, y_flat, I_app, p, context):
    """
    Lithium Plating ODE: dn_Li/dt = -i_plating / F
    """
    # 1. Extract exactly what we need
    T_cell = physics.state(y_flat, 'temperature')
    eta_Li = context['phi_s_n'] - 0.0 # vs Li/Li+ reference = 0V
    
    # 2. Explicit Math for Butler-Volmer kinetics
    i_plating = 1e-6 * (torch.exp(-0.5*p['F']*eta_Li/(p['R_g']*T_cell)) 
                      - torch.exp(0.5*p['F']*eta_Li/(p['R_g']*T_cell)))
                      
    return -i_plating / p['F']

spec = StateEquationSpec(
    order=95, 
    size=1, 
    initial=0.0, 
    scale=1e-6, 
    nonnegative=True,
    rhs=lithium_plating_rhs, 
    equation="dn_Li/dt = -i_plating / F"
)
