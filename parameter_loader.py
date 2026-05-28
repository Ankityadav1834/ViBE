"""
parameter_loader.py
-------------------
Utilities for loading battery chemistry parameters from a folder of CSV files
and a scalar-parameter definition file (params.json).

Usage
-----
    from parameter_loader import load_chemistry_from_folder

    chem = load_chemistry_from_folder("parameters/na_ion_chayambuka2022")
    solver = ImplicitBatterySolver(config, disc, overrides, chemistry=chem)

Folder layout expected
----------------------
    parameters/
        my_chemistry/
            params.json          # scalar / constant parameters
            U_n.csv              # OCP of negative electrode  (stoich vs V)
            U_p.csv              # OCP of positive electrode  (stoich vs V)
            D_n.csv              # Solid diffusivity – anode  (conc vs m^2/s)
            D_p.csv              # Solid diffusivity – cathode (conc vs m^2/s)
            k_n.csv              # Reaction rate constant – anode (conc vs m/s)
            k_p.csv              # Reaction rate constant – cathode (conc vs m/s)
            D_e.csv              # Electrolyte diffusivity   (conc vs m^2/s)
            sigma_e.csv          # Electrolyte conductivity  (conc vs S/m)

Any CSV file that is absent falls back gracefully (uniform zero / constant).

params.json keys
----------------
All parameters that would normally appear in get_standard_parameters() can
be listed here.  Function-valued parameters (OCP, Ds, k) are automatically
assembled from the corresponding CSV files — you do NOT need to list them in
params.json (they will be overwritten by the CSV-loaded functions).
"""

import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Low-level interpolation helper
# ---------------------------------------------------------------------------

class TorchInterpolant:
    """
    A 1-D piecewise-linear interpolator backed by PyTorch tensors.

    Inputs are clamped to the knot range, so extrapolation is handled via
    constant extension (flat extrapolation).

    Parameters
    ----------
    x : array-like
        Independent variable knots (need not be pre-sorted).
    y : array-like
        Dependent variable values at the knots.
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        self.x = torch.as_tensor(x[order])
        self.y = torch.as_tensor(y[order])

    # ------------------------------------------------------------------
    def _ensure_device(self, x_new):
        if self.x.device != x_new.device:
            self.x = self.x.to(x_new.device)
            self.y = self.y.to(x_new.device)

    # ------------------------------------------------------------------
    def __call__(self, x_new):
        """
        Evaluate the interpolant at *x_new*.

        Works with scalars, NumPy arrays, and PyTorch tensors.
        Preserves the input type on return (tensor in → tensor out, etc.).
        """
        is_tensor = torch.is_tensor(x_new)
        if not is_tensor:
            x_new = torch.as_tensor(x_new, dtype=self.x.dtype)

        self._ensure_device(x_new)

        clamped = torch.clamp(x_new, self.x[0], self.x[-1])
        idx = torch.clamp(
            torch.searchsorted(self.x.contiguous(), clamped.contiguous()) - 1,
            0,
            len(self.x) - 2,
        )

        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]
        t = (clamped - x0) / (x1 - x0 + 1e-20)
        out = y0 + t * (y1 - y0)

        if is_tensor:
            return out
        result = out.detach().cpu().numpy()
        return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# 2. CSV reader
# ---------------------------------------------------------------------------

def load_csv_data(filepath):
    """
    Read a two-column CSV file (with an optional header row) and return
    ``(x_array, y_array)`` as NumPy float64 arrays.

    If the file does not exist, returns two dummy arrays of length 10
    spanning [0, 1] with zero y-values so the rest of the code won't crash.
    """
    if not os.path.exists(filepath):
        return np.linspace(0.0, 1.0, 10, dtype=np.float64), np.zeros(10, dtype=np.float64)

    import pandas as pd
    df = pd.read_csv(filepath, header=None).apply(pd.to_numeric, errors="coerce").dropna()
    data = df.values.astype(np.float64)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# 3. Chemistry folder loader
# ---------------------------------------------------------------------------

def load_chemistry_from_folder(folder_path):
    """
    Load a complete chemistry parameter set from *folder_path*.

    Returns
    -------
    dict
        A chemistry dict compatible with ``ImplicitBatterySolver``:
        ``{"name": ..., "params": {...}, "ocp_n": fn, "ocp_p": fn,
           "cond_e": fn, "diff_e": fn}``

    Notes
    -----
    * ``params.json`` in the folder must contain all scalar parameters.
    * Any CSV files (U_n, U_p, D_n, D_p, k_n, k_p, D_e, sigma_e) found in
      the folder automatically replace the corresponding param-dict entries
      with interpolating functions.
    * All function-valued parameters are *closures* over a ``TorchInterpolant``,
      so they are fully compatible with ``torch.func.jacrev`` and batched
      derivatives.
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Chemistry folder not found: {folder_path!r}\n"
            "Make sure the folder exists and contains params.json."
        )

    # ---- 3a. Load scalar parameters from params.json -------------------
    params_file = os.path.join(folder_path, "params.json")
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"params.json not found in {folder_path!r}.\n"
            "Every chemistry folder must contain a params.json file."
        )
    with open(params_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Preserve numeric types exactly (no silent coercion)
    params = {k: float(v) if isinstance(v, (int, float)) else v for k, v in raw.items()}

    # Convenience shorthand
    def csv(name):
        return load_csv_data(os.path.join(folder_path, name))

    # ---- 3b. Build interpolants for any CSV files that exist -----------
    F  = float(params.get("F", 96485.33))
    ce0 = float(params.get("ce_0", 1000.0))

    # --- OCP functions ---
    ocp_n_interp, ocp_p_interp = None, None
    if os.path.exists(os.path.join(folder_path, "U_n.csv")):
        x, y = csv("U_n.csv")
        ocp_n_interp = TorchInterpolant(x, y)

    if os.path.exists(os.path.join(folder_path, "U_p.csv")):
        x, y = csv("U_p.csv")
        ocp_p_interp = TorchInterpolant(x, y)

    # --- Solid diffusivity (concentration-dependent) ---
    Ds_n_interp, Ds_p_interp = None, None
    if os.path.exists(os.path.join(folder_path, "D_n.csv")):
        x, y = csv("D_n.csv")
        Ds_n_interp = TorchInterpolant(x, y)

    if os.path.exists(os.path.join(folder_path, "D_p.csv")):
        x, y = csv("D_p.csv")
        Ds_p_interp = TorchInterpolant(x, y)

    # --- Reaction rate constants (concentration-dependent) ---
    k_n_interp, k_p_interp = None, None
    if os.path.exists(os.path.join(folder_path, "k_n.csv")):
        x, y = csv("k_n.csv")
        k_n_interp = TorchInterpolant(x, y)

    if os.path.exists(os.path.join(folder_path, "k_p.csv")):
        x, y = csv("k_p.csv")
        k_p_interp = TorchInterpolant(x, y)

    # --- Electrolyte diffusivity / conductivity ---
    De_interp, sigma_e_interp = None, None
    if os.path.exists(os.path.join(folder_path, "D_e.csv")):
        x, y = csv("D_e.csv")
        De_interp = TorchInterpolant(x, y)

    if os.path.exists(os.path.join(folder_path, "sigma_e.csv")):
        x, y = csv("sigma_e.csv")
        sigma_e_interp = TorchInterpolant(x, y)

    # ---- 3c. Build callable wrappers compatible with BatteryPhysics ----
    cs_max_n = float(params.get("cs_max_n", 33133.0))
    cs_max_p = float(params.get("cs_max_p", 63104.0))

    # OCP callables  (stoichiometry in, voltage out)
    if ocp_n_interp is not None:
        def ocp_n(sto, _interp=ocp_n_interp):
            return _interp(sto)
    else:
        def ocp_n(sto):
            # Fallback: Li-ion Chen2020 anode OCP
            s = torch.clamp(sto, 0.001, 0.999)
            return (1.9793 * torch.exp(-39.3631 * s) + 0.2482
                    - 0.0909 * torch.tanh(29.8538 * (s - 0.1234))
                    - 0.04478 * torch.tanh(14.9159 * (s - 0.2769))
                    - 0.0205 * torch.tanh(30.4444 * (s - 0.6103)))

    if ocp_p_interp is not None:
        def ocp_p(sto, _interp=ocp_p_interp):
            return _interp(sto)
    else:
        def ocp_p(sto):
            # Fallback: Li-ion Chen2020 cathode OCP
            s = torch.clamp(sto, 0.001, 0.999)
            return (-0.8090 * s + 4.4875
                    - 0.0428 * torch.tanh(18.5138 * (s - 0.5542))
                    - 17.7326 * torch.tanh(15.7890 * (s - 0.3117))
                    + 17.5842 * torch.tanh(15.9308 * (s - 0.3120)))

    # Electrolyte property callables  (ce [mol/m³], T → value)
    if sigma_e_interp is not None:
        def cond_e(ce, T, _interp=sigma_e_interp):
            return _interp(ce)
    else:
        def cond_e(ce, T):
            c_k = ce / 1000.0
            return 0.1297 * c_k**3 - 2.51 * c_k**1.5 + 3.329 * c_k

    if De_interp is not None:
        def diff_e(ce, T, _interp=De_interp):
            return _interp(ce)
    else:
        def diff_e(ce, T):
            c_k = ce / 1000.0
            return 8.794e-11 * c_k**2 - 3.972e-10 * c_k + 4.862e-10

    # Solid diffusivity callables  (sto, T → Ds [m²/s])
    # The CSV for D_n/D_p stores *concentration* as x-axis, not stoichiometry.
    # We convert sto → concentration using cs_max before interpolating.
    if Ds_n_interp is not None:
        _cs_max_n_f = cs_max_n
        def Ds_n_func(sto, T, _interp=Ds_n_interp, _cm=_cs_max_n_f):
            return _interp(sto * _cm)
        params["Ds_n"] = Ds_n_func
    # else: Ds_n stays as the scalar already in params.json

    if Ds_p_interp is not None:
        _cs_max_p_f = cs_max_p
        def Ds_p_func(sto, T, _interp=Ds_p_interp, _cm=_cs_max_p_f):
            return _interp(sto * _cm)
        params["Ds_p"] = Ds_p_func

    # Reaction rate callables  (c_s_surf → m_ref [A·m / mol^1.5])
    # The CSV stores exchange-current-density rate [m/s] vs. concentration.
    # Convention: m_ref_n(c_s_surf) returns F/(2*ce0^0.5) * k(c_s_surf)
    if k_n_interp is not None:
        _F_f, _ce0_f = F, ce0
        def k_n_func(c_s_surf, _interp=k_n_interp, _F=_F_f, _ce0=_ce0_f):
            return (_F / (2.0 * _ce0 ** 0.5)) * _interp(c_s_surf)
        params["m_ref_n"] = k_n_func

    if k_p_interp is not None:
        _F_f, _ce0_f = F, ce0
        def k_p_func(c_s_surf, _interp=k_p_interp, _F=_F_f, _ce0=_ce0_f):
            return (_F / (2.0 * _ce0 ** 0.5)) * _interp(c_s_surf)
        params["m_ref_p"] = k_p_func

    # ---- 3d. Assemble and return the chemistry dict --------------------
    chemistry_name = os.path.basename(folder_path)
    return {
        "name": chemistry_name,
        "params": params,
        "ocp_n": ocp_n,
        "ocp_p": ocp_p,
        "cond_e": cond_e,
        "diff_e": diff_e,
    }


# ---------------------------------------------------------------------------
# 4. Convenience: list available chemistries in the default parameters/ dir
# ---------------------------------------------------------------------------

def list_available_chemistries(parameters_root=None):
    """
    Print all chemistry folders found under *parameters_root*
    (defaults to a ``parameters/`` directory next to this file).

    Returns a list of folder names.
    """
    if parameters_root is None:
        parameters_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters")

    if not os.path.isdir(parameters_root):
        print(f"[parameter_loader] No 'parameters/' directory found at {parameters_root!r}")
        return []

    entries = [
        d for d in os.listdir(parameters_root)
        if os.path.isdir(os.path.join(parameters_root, d))
        and os.path.exists(os.path.join(parameters_root, d, "params.json"))
    ]

    print(f"Available chemistries in {parameters_root!r}:")
    for name in sorted(entries):
        print(f"  • {name}")
    return sorted(entries)
