"""
lithium_plating_models.py — Modular Lithium Plating Model Library
=================================================================
Each model computes `dn_Li_dt` [mol/m³/s] — the molar rate of lithium
deposition per unit electrode volume.

Config key: `lithium_plating_options.model`
Valid values: 'builtin', 'none', 'irreversible', 'yang2017'

Physical background:
    Lithium plating occurs when the anode potential drops below 0 V vs Li/Li+.
    All models use: eta_Li = phi_s_n  (vs Li/Li+ reference = 0 V)
    The current convention: i_plating > 0 = plating (deposition).
"""

import torch

# Default plating kinetic parameters (Chen2020 / Yang2017 inspired)
PLATING_PARAMS = {
    "j0_Li":    1.0e-6,    # Exchange current density for Li plating [A/m²]
    "alpha_Li": 0.5,       # Symmetry factor [-]
    "j0_Li_strip": 1.0e-4, # Exchange current density for stripping [A/m²]
}

def _pp(p):
    merged = dict(PLATING_PARAMS)
    merged.update({k: p[k] for k in PLATING_PARAMS if k in p})
    return merged


def plating_none(phi_s_n, i_n, T_cell, p, device):
    """No lithium plating. dn_Li/dt = 0"""
    return torch.zeros(1, device=device, dtype=phi_s_n.dtype)


def plating_builtin(phi_s_n, i_n, T_cell, p, device):
    """
    Built-in symmetric Butler-Volmer plating (current default).
    Activates when eta_Li < 0 (anode below Li/Li+ reference).

        i_plating = j0_Li * (exp(-alpha*F*eta/RT) - exp(alpha*F*eta/RT))
        dn_Li/dt  = -i_plating / F     [mol/m²/s → stored as total mol]
    """
    pp = _pp(p)
    F, R_g = p['F'], p['R_g']
    eta_Li = phi_s_n  # vs Li/Li+ reference = 0 V
    arg = pp["alpha_Li"] * F * eta_Li / (R_g * T_cell)
    arg = torch.clamp(arg, -50.0, 50.0)
    i_plating = pp["j0_Li"] * (torch.exp(-arg) - torch.exp(arg))
    return -i_plating / F


def plating_irreversible(phi_s_n, i_n, T_cell, p, device):
    """
    Irreversible plating model — lithium only deposits, never strips.
    Plating current activates only when eta_Li < 0.
    Stripping (eta_Li > 0) is suppressed.

        i_plating = j0_Li * exp(-alpha*F*eta/RT)  if eta < 0 else 0
    """
    pp = _pp(p)
    F, R_g = p['F'], p['R_g']
    eta_Li = phi_s_n
    arg = torch.clamp(pp["alpha_Li"] * F * eta_Li / (R_g * T_cell), max=50.0)
    # Only apply when overpotential drives deposition (eta < 0)
    i_plating = pp["j0_Li"] * torch.exp(-arg) * (eta_Li < 0).float()
    return -i_plating / F


def plating_yang2017(phi_s_n, i_n, T_cell, p, device):
    """
    Yang et al. (2017) asymmetric plating/stripping model.
    Uses different exchange current densities for deposition vs stripping,
    capturing the irreversibility observed experimentally.

    Plating  (eta < 0): i = -j0_Li    * exp(-alpha*F*eta/RT)
    Stripping (eta ≥ 0): i =  j0_strip * exp( alpha*F*eta/RT)
    """
    pp = _pp(p)
    F, R_g = p['F'], p['R_g']
    eta_Li = phi_s_n
    arg = torch.clamp(pp["alpha_Li"] * F * eta_Li / (R_g * T_cell), -50.0, 50.0)

    plating_side  = -pp["j0_Li"] * torch.exp(-arg)
    stripping_side =  pp["j0_Li_strip"] * torch.exp(arg)

    # Smooth switch using tanh (avoids discontinuous gradient)
    switch = 0.5 * (1.0 - torch.tanh(eta_Li / 0.001))  # 1 when plating, 0 when stripping
    i_plating = switch * plating_side + (1.0 - switch) * stripping_side
    return -i_plating / F


_REGISTRY = {
    "builtin":      plating_builtin,
    "none":         plating_none,
    "irreversible": plating_irreversible,
    "yang2017":     plating_yang2017,
}

VALID_PLATING_MODELS = list(_REGISTRY.keys())

def get_plating_model(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown plating model '{name}'. Choose from: {VALID_PLATING_MODELS}")
    return _REGISTRY[name]
