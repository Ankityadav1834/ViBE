"""
temperature_models.py — Modular Thermal Model Library
======================================================
Each model computes `dT_dt` [K/s] — the rate of temperature change.

Config key: `thermal_options.model`
Valid values: 'builtin', 'isothermal', 'entropic'

Note: The thermal_options.strategy ('ambient', 'adiabatic') is still respected
by the 'builtin' model. Set `model` to 'isothermal' to bypass all heat generation.
"""

import torch


def thermal_isothermal(I_app, OCV, V_cell, T_cell, p, device):
    """
    Isothermal model — temperature stays constant.
    dT/dt = 0 regardless of current or heat generation.
    Useful for isolating electrochemical effects from thermal.
    """
    return torch.zeros_like(T_cell)


def thermal_builtin(I_app, OCV, V_cell, T_cell, p, device):
    """
    Lumped thermal model (default).
    Joule + reaction heat balanced by convective cooling.

        Q_gen = I * (OCV - V_cell)       [W]
        Q_cool = hA * (T - T_amb)        [W]
        dT/dt = (Q_gen - Q_cool) / (rho_Cp * Vol)   [K/s]
    """
    Q_gen  = I_app * (OCV - V_cell)
    Q_cool = p['hA'] * (T_cell - p['T_amb'])
    return (Q_gen - Q_cool) / (p['rho_Cp'] * p['Vol_cell'])


def thermal_entropic(I_app, OCV, V_cell, T_cell, p, device):
    """
    Extended thermal model with entropic heat correction (Bernardi 1985).
    Adds the reversible heat term: Q_rev = I * T * dOCV/dT

    dOCV/dT is approximated as -0.0003 V/K (typical graphite/NMC value,
    can be overridden via `p['dOCV_dT']`).

        Q_irr = I * (OCV - V_cell)
        Q_rev = I * T * dOCV_dT
        Q_cool = hA * (T - T_amb)
        dT/dt = (Q_irr + Q_rev - Q_cool) / (rho_Cp * Vol)
    """
    dOCV_dT = p.get('dOCV_dT', -3.0e-4)   # [V/K], negative for graphite anode
    Q_irr  = I_app * (OCV - V_cell)
    Q_rev  = I_app * T_cell * dOCV_dT
    Q_cool = p['hA'] * (T_cell - p['T_amb'])
    return (Q_irr + Q_rev - Q_cool) / (p['rho_Cp'] * p['Vol_cell'])


_REGISTRY = {
    "builtin":    thermal_builtin,
    "isothermal": thermal_isothermal,
    "entropic":   thermal_entropic,
}

VALID_THERMAL_MODELS = list(_REGISTRY.keys())

def get_thermal_model(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown thermal model '{name}'. Choose from: {VALID_THERMAL_MODELS}")
    return _REGISTRY[name]
