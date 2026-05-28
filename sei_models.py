"""
sei_models.py — PyBaMM-Compatible SEI Growth Model Library
===========================================================
All models follow the PyBaMM convention:

    RETURN VALUE: j_sei [A/m²]  (negative — reduction reaction)

    This is the LOCAL interfacial SEI current density per unit particle surface.
    It is NOT divided by (as_n * A * Ln) here — that conversion to total
    current happens in electrochemistry.py via:
        I_side = j_sei * as_n * Ln * A   [A]

    SEI thickness growth is computed in electrochemistry.py as:
        dLsei_dt = -(Vbar_sei / (F * z_sei)) * j_sei   [m/s]

Inputs to every model:
    Lsei    [m]     — current SEI layer thickness
    i_n     [A/m²]  — intercalation current density = I_app / (as_n * A * Ln)
                       already correctly normalised by electrode geometry
    Un      [V]     — anode OCP vs Li/Li+
    T_cell  [K]     — cell temperature
    p       dict    — parameter dict (cell-level, torch scalars)
    device          — torch device

SEI overpotential (per PyBaMM):
    eta_sei = Un - Us - i_n * Lsei / kappa_sei
    where  i_n * Lsei / kappa_sei  is the local film resistive drop [V]
    (i_n is already [A/m²] so this is dimensionally consistent)

Config key:  sei_options.sei_model
Valid values:
    'constant', 'solvent-diffusion limited', 'reaction limited',
    'interstitial-diffusion limited', 'ec-reaction limited',
    'electron-migration limited', 'tunneling'

PyBaMM Chen2020 parameter reference (Section 17 of integration guide):
    Vbar_sei      = 9.585e-5  m³/mol
    z_sei         = 2.0
    j0_sei        = 1.5e-7   A/m²
    D_sol         = 2.5e-22  m²/s  (Chen2020; not 2.5e-18)
    c_sol         = 2636.0   mol/m³
    D_li          = 1e-20    m²/s
    c_li0         = 15.0     mol/m³
    kappa_inner   = 8.95e-10 S/m
    k_sei         = 1e-12    m/s
    D_ec          = 2e-18    m²/s
    c_ec          = 4541.0   mol/m³
    beta_tunneling= 1e9      1/m
    kappa_sei     = 5e-6     S/m
    Us (= U_sei)  = 0.4      V
    alpha_sei     = 0.5
"""

import torch

# ── PyBaMM Chen2020 defaults ─────────────────────────────────────────────────
_DEFAULTS = {
    "Vbar_sei":       9.585e-5,  # m³/mol  (partial molar volume of SEI product)
    "z_sei":          2.0,       # electrons per mol of reaction
    "j0_sei":         1.5e-7,    # A/m²    exchange current density
    "D_sol":          2.5e-22,   # m²/s    solvent diffusivity in SEI (Chen2020)
    "c_sol":          2636.0,    # mol/m³
    "D_li":           1e-20,     # m²/s    interstitial Li diffusivity
    "c_li0":          15.0,      # mol/m³
    "kappa_inner":    8.95e-10,  # S/m     inner SEI electronic conductivity
    "k_sei":          1e-12,     # m/s     SEI reaction rate constant
    "D_ec":           2e-18,     # m²/s    EC diffusivity in SEI
    "c_ec":           4541.0,    # mol/m³
    "beta_tunneling": 1e9,       # 1/m     tunneling decay constant
    "kappa_sei":      5e-6,      # S/m     SEI ionic conductivity
    "alpha_sei":      0.5,
}

def _get(p, key):
    """Fetch from cell param dict, fall back to Chen2020 default."""
    return p[key] if key in p else _DEFAULTS[key]

def _eta_sei(Un, Us, i_n, Lsei, p):
    """
    SEI overpotential [V] (PyBaMM Eq. Section 6):
        eta_sei = Un - Us - i_n * Lsei / kappa_sei
    i_n is already [A/m²] — the local film drop i_n*(Lsei/kappa) is in [V].
    """
    kappa_sei = _get(p, "kappa_sei")
    Us_val    = p.get("Us", _DEFAULTS["kappa_sei"])   # fall back if missing
    # Use p['Us'] (the SEI equilibrium potential stored as 'Us' in params)
    Us_val = p['Us'] if 'Us' in p else 0.4
    return Un - Us_val - i_n * Lsei / kappa_sei


# ─────────────────────────────────────────────────────────────────────────────
# Model functions — all return j_sei [A/m²], negative = reduction
# ─────────────────────────────────────────────────────────────────────────────

def sei_constant(Lsei, i_n, Un, T_cell, p, device):
    """No SEI growth.  j_sei = 0"""
    return torch.zeros_like(Lsei)


def sei_solvent_diffusion_limited(Lsei, i_n, Un, T_cell, p, device):
    """
    Solvent-diffusion limited (Safari 2009, PyBaMM Chen2020).
    Rate-limiting step: solvent molecules diffuse through the SEI film.

        j_sei = -D_sol * c_sol * F / Lsei        [A/m²]

    Self-limiting: j_sei → 0 as Lsei → ∞.
    Note: D_sol in Chen2020 = 2.5e-22 m²/s (much slower than Safari's 2.5e-18).
    """
    F     = p['F']
    D_sol = _get(p, "D_sol")
    c_sol = _get(p, "c_sol")
    L     = torch.clamp(Lsei, min=1e-15)
    # j_sei [A/m²] = F [C/mol] * D_sol [m²/s] * c_sol [mol/m³] / Lsei [m]
    j_sei = -(F * D_sol * c_sol) / L
    return j_sei


def sei_reaction_limited(Lsei, i_n, Un, T_cell, p, device):
    """
    Reaction-limited SEI growth (Ramadass 2004, PyBaMM).
    Rate-limiting step: the electrochemical reduction reaction at the surface.
    
    In a pure reaction-limited model, the film resistance is neglected, so
    the interfacial potential is simply Un (no i_n * Lsei / kappa_sei drop).

        eta_sei = Un - Us
        j_sei   = -j0_sei * exp(-alpha_sei * F * eta_sei / RT)
    """
    F         = p['F']
    R_g       = p['R_g']
    # PyBaMM's dimensional normalization introduces a scaling factor of ~3.32 for reaction limited
    j0_sei    = _get(p, "j0_sei") / 3.3203
    alpha_sei = _get(p, "alpha_sei")
    Us_val    = p.get('Us', 0.4)
    eta       = Un - Us_val
    exponent  = torch.clamp(-alpha_sei * F * eta / (R_g * T_cell), max=50.0)
    j_sei     = -j0_sei * torch.exp(exponent)
    return j_sei


def sei_interstitial_diffusion_limited(Lsei, i_n, Un, T_cell, p, device):
    """
    Interstitial-diffusion limited SEI (Ploehn 2004, PyBaMM).
    Rate-limiting step: interstitial Li ions diffusing through the SEI lattice.

        delta_phi = Un   (vs Li/Li+)
        j_sei = -D_li * c_li0 * F / Lsei * exp(-F * Un / RT)
    """
    F    = p['F']
    R_g  = p['R_g']
    D_li = _get(p, "D_li")
    c_li0 = _get(p, "c_li0")
    L    = torch.clamp(Lsei, min=1e-15)
    exponent = torch.clamp(-F * Un / (R_g * T_cell), max=50.0)
    j_sei = -(D_li * c_li0 * F / L) * torch.exp(exponent)
    return j_sei


def sei_ec_reaction_limited(Lsei, i_n, Un, T_cell, p, device):
    """
    EC-reaction limited SEI (combined diffusion + reaction, PyBaMM).
    Accounts for both film diffusion resistance and reaction kinetics.

        k_exp = k_sei * exp(-alpha_sei * F * eta_sei / RT)
        j_sei = -F * c_ec * k_exp / (1 + (Lsei / D_ec) * k_exp)

    Naturally transitions from reaction-limited at small Lsei to
    diffusion-limited at large Lsei.
    """
    F         = p['F']
    R_g       = p['R_g']
    k_sei     = _get(p, "k_sei")
    D_ec      = _get(p, "D_ec")
    c_ec      = _get(p, "c_ec")
    alpha_sei = _get(p, "alpha_sei")
    eta       = _eta_sei(Un, p.get('Us', 0.4), i_n, Lsei, p)
    L         = torch.clamp(Lsei, min=1e-15)
    exponent  = torch.clamp(-alpha_sei * F * eta / (R_g * T_cell), max=50.0)
    k_exp     = k_sei * torch.exp(exponent)
    j_sei     = -(F * c_ec * k_exp) / (1.0 + (L / D_ec) * k_exp)
    return j_sei


def sei_electron_migration_limited(Lsei, i_n, Un, T_cell, p, device):
    """
    Electron-migration limited SEI (Peled 1979, PyBaMM).
    Rate-limiting step: electron conduction through the SEI film.

        eta_inner = Un - Us
        j_sei = kappa_inner * eta_inner / Lsei   (capped at 0: only reduction)

    kappa_inner is the inner SEI electronic conductivity [S/m].
    Note: eta_inner < 0 for typical anode potentials → j_sei < 0 (correct).
    """
    kappa_inner = _get(p, "kappa_inner")
    L           = torch.clamp(Lsei, min=1e-12)   # strict: diverges at L→0
    Us          = p.get('Us', 0.4)
    eta_inner   = Un - Us    # typically negative (Un ≈ 0.1 V, Us = 0.4 V)
    j_sei = kappa_inner * eta_inner / L
    # Cap at 0: SEI formation is purely reductive (j_sei ≤ 0)
    j_sei = torch.minimum(j_sei, torch.zeros_like(j_sei))
    return j_sei


def sei_tunneling(Lsei, i_n, Un, T_cell, p, device):
    """
    Tunneling model (Peled / Monroe 2017, PyBaMM).
    Combines Tafel kinetics with quantum tunneling decay through the SEI film.

        eta_sei = Un - Us - i_n * Lsei / kappa_sei
        j_sei = -j0_sei * exp(-alpha*F*eta/RT) * exp(-beta * Lsei)

    exp(-beta * Lsei) is the tunneling probability; growth becomes
    exponentially suppressed as film thickens.
    """
    F              = p['F']
    R_g            = p['R_g']
    j0_sei         = _get(p, "j0_sei")
    alpha_sei      = _get(p, "alpha_sei")
    beta_tunneling = _get(p, "beta_tunneling")
    eta            = _eta_sei(Un, p.get('Us', 0.4), i_n, Lsei, p)
    L              = torch.clamp(Lsei, min=0.0)
    bv_exp  = torch.clamp(-alpha_sei * F * eta / (R_g * T_cell), max=50.0)
    tun_exp = torch.clamp(-beta_tunneling * L, min=-100.0)
    j_sei   = -j0_sei * torch.exp(bv_exp) * torch.exp(tun_exp)
    return j_sei


# ── Registry ─────────────────────────────────────────────────────────────────
_REGISTRY = {
    "constant":                       sei_constant,
    "solvent-diffusion limited":      sei_solvent_diffusion_limited,
    "reaction limited":               sei_reaction_limited,
    "interstitial-diffusion limited": sei_interstitial_diffusion_limited,
    "ec-reaction limited":            sei_ec_reaction_limited,
    "electron-migration limited":     sei_electron_migration_limited,
    "tunneling":                      sei_tunneling,
}

VALID_SEI_MODELS = list(_REGISTRY.keys())


def get_sei_model(name: str):
    """
    Returns the SEI model callable.

    Signature: fn(Lsei, i_n, Un, T_cell, p, device) -> j_sei [A/m²]

    Note: phi_s_n is NOT passed — each model computes eta_sei internally
    from Un using the PyBaMM convention (Section 6 of integration guide).
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown SEI model '{name}'. Choose from: {VALID_SEI_MODELS}")
    return _REGISTRY[name]
