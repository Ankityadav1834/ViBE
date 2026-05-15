"""
stress_models.py — Modular Mechanical Stress Model Library
==========================================================
Each model computes `(total_surf_stress, stress_enhancement)` from the
particle concentration field.

`stress_enhancement` multiplies the SEI side reaction current:
    i_side = i_side_base * stress_enhancement

`total_surf_stress` is logged to the output for analysis.

Config key: `stress_options.model`
Valid values: 'builtin', 'linear_elastic', 'none'
"""

import torch
import math


def stress_none(cs_n, Lsei, physics, p, device):
    """
    No stress effects. Stress enhancement = 1.0 (no amplification).
    Use this when you want to isolate electrochemical effects from mechanics.
    """
    zero = torch.zeros(1, device=device, dtype=cs_n.dtype)
    return zero, torch.ones(1, device=device, dtype=cs_n.dtype)


def stress_linear_elastic(cs_n, Lsei, physics, p, device):
    """
    Simplified linear elastic model (no crack detection).
    Computes diffusion-induced stress at the particle surface only.
    Stress enhancement is a smooth function of surface stress amplitude.

    Equations:
        sigma_th = E*Omega / (3*(1-nu)) * (c_bar - c_surf)   [Pa]
        stress_enhancement = 1 + gamma * |sigma_th| / sigma_ref
    """
    E_g, nu_g, Omega = 15e9, 0.3, 3.17e-6
    pf = E_g * Omega / (3.0 * (1.0 - nu_g))

    r_ref = physics.r_n_ref
    r_faces = torch.zeros(physics.Nr_n + 1, device=device, dtype=cs_n.dtype)
    r_faces[1:-1] = 0.5 * (r_ref[:-1] + r_ref[1:])
    r_faces[-1] = 1.0
    v = r_faces[1:]**3 - r_faces[:-1]**3
    c_bar = torch.sum(cs_n * v) / torch.sum(v)
    c_surf = cs_n[-1]

    sth_surf = 3.0 * pf * (c_bar - c_surf)

    sigma_ref = 50e6  # 50 MPa reference scale
    gamma = 1.5       # enhancement coefficient
    stress_enhancement = 1.0 + gamma * torch.abs(sth_surf) / sigma_ref
    return sth_surf, stress_enhancement


def stress_builtin(cs_n, Lsei, physics, p, device):
    """
    Full built-in model: linear elastic DIS + SEI mismatch stress + crack detection.
    Crack opening amplifies the SEI side reaction up to 4x.

    Equations (after Reniers 2019 / Zhao 2011):
        sigma_th   = E_g*Omega/(3*(1-nu_g)) * (c_bar - c_surf)
        sigma_sei  = E_sei/(1-nu_sei) * Omega/3 * (c_surf - c_ref) + sigma_intr
        sigma_tot  = sigma_th + sigma_sei
        crack_flag = smooth_step(sigma_tot - K_Ic/sqrt(pi*L))
        enhancement = 1 + 3 * crack_flag    (up to 4x)
    """
    E_sei, nu_sei, Omega = 10e9, 0.25, 3.17e-6
    sigma_intr = -0.5e9
    E_g, nu_g = 15e9, 0.3
    E_sei_b = E_sei / (1.0 - nu_sei)
    pf = E_g * Omega / (3.0 * (1.0 - nu_g))

    c_surf = cs_n[-1]
    c_surf_ref = 0.8 * p['cs_max_n']

    r_ref = physics.r_n_ref
    r_faces = torch.zeros(physics.Nr_n + 1, device=device, dtype=cs_n.dtype)
    r_faces[1:-1] = 0.5 * (r_ref[:-1] + r_ref[1:])
    r_faces[-1] = 1.0
    v = r_faces[1:]**3 - r_faces[:-1]**3
    c_bar = torch.sum(cs_n * v) / torch.sum(v)

    sth_surf = 3.0 * pf * (c_bar - c_surf)
    sigma_sei = E_sei_b * (Omega / 3.0) * (c_surf - c_surf_ref) + sigma_intr
    total_surf_stress = sth_surf + sigma_sei

    K_Ic = 0.3e6
    sigma_crack_threshold = K_Ic / torch.sqrt(math.pi * torch.clamp(Lsei, min=1e-10))
    crack_flag = 0.5 * (1.0 + torch.tanh((total_surf_stress - sigma_crack_threshold) / 1e5))
    stress_enhancement = 1.0 + 3.0 * crack_flag
    return total_surf_stress, stress_enhancement


_REGISTRY = {
    "builtin":       stress_builtin,
    "linear_elastic": stress_linear_elastic,
    "none":          stress_none,
}

VALID_STRESS_MODELS = list(_REGISTRY.keys())

def get_stress_model(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown stress model '{name}'. Choose from: {VALID_STRESS_MODELS}")
    return _REGISTRY[name]
