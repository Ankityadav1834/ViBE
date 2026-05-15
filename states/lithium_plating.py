"""
states/lithium_plating.py
=========================
Lithium plating ODE state equation.

The plating model is selected via `lithium_plating_options.model` in config.
Each model receives the corrected phi_s_n (normalized by i_n) from context.
"""

from states.registry import StateEquationSpec
import torch

def _plating_enabled(config):
    return bool(config.get('lithium_plating_options', {}).get('enabled', False))

def lithium_plating_rhs(physics, y_flat, I_app, p, context):
    """
    dn_Li/dt = model(phi_s_n, i_n, T_cell, p)

    phi_s_n and i_n are passed via context (computed in electrochemistry.py).
    The model is selected from lithium_plating_models.py.
    """
    phi_s_n = context['phi_s_n']   # already normalized with i_n film correction
    i_n     = context.get('i_n', torch.zeros_like(phi_s_n))
    T_cell  = physics.state(y_flat, 'temperature')

    model_name = getattr(physics, 'plating_model_name', 'builtin')

    if model_name == 'builtin':
        # Original symmetric Butler-Volmer
        eta_Li = phi_s_n  # vs Li/Li+ reference = 0 V
        i_plating = 1e-6 * (torch.exp(-0.5*p['F']*eta_Li/(p['R_g']*T_cell))
                           - torch.exp( 0.5*p['F']*eta_Li/(p['R_g']*T_cell)))
        return -i_plating / p['F']
    else:
        from lithium_plating_models import get_plating_model
        return get_plating_model(model_name)(phi_s_n, i_n, T_cell, p, physics.device)


spec = StateEquationSpec(
    order=95,
    size=1,
    initial=0.0,
    scale=1e-6,
    nonnegative=True,
    enabled=_plating_enabled,
    options_key='lithium_plating_options',
    rhs=lithium_plating_rhs,
    equation="dn_Li/dt = -i_plating / F  (model selectable via lithium_plating_options.model)"
)
