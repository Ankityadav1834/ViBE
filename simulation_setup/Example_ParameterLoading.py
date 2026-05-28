"""
Example: Loading a battery chemistry from the parameters/ folder.

This script shows three ways to run a simulation with different chemistries:
  1. Default Li-ion (no chemistry argument needed — backward compatible)
  2. Na-ion loaded via chemistry_folder path argument
  3. Na-ion loaded manually with load_chemistry_from_folder() for inspection

Run from the project root:
    python simulation_setup/Example_ParameterLoading.py
"""

import os
import sys

# ── Make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ImplicitBatterySolver
from parameter_loader import load_chemistry_from_folder, list_available_chemistries

# ── Common solver settings ────────────────────────────────────────────────────
config = {
    'device': 'cpu',
    'n_series': 1,
    'n_parallel': 1,
    'electrolyte_spatial_method': 'finite_volume',
    'solid_spatial_method': 'chebyshev',
    'sei_options': {'enabled': False},
    'stress_options': {'enabled': False},
}
discretization = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n': 10, 'Nx_s': 10, 'Nx_p': 10,
    'Nsei': 1,
}

# ── 0. List what's available ──────────────────────────────────────────────────
print("\n=== Available chemistries ===")
list_available_chemistries()   # prints all folders in parameters/

# ── 1. Default Li-ion (no argument needed) ────────────────────────────────────
print("\n=== 1. Default Li-ion (backward compatible) ===")
solver_liion = ImplicitBatterySolver(config, discretization, {})
v_li = solver_liion.get_exact_terminal_voltages(solver_liion.y, 5.0)
print(f"Li-ion OC terminal voltage: {v_li.item():.4f} V")

# ── 2. Na-ion via chemistry_folder ────────────────────────────────────────────
print("\n=== 2. Na-ion loaded from parameters/na_ion_chayambuka2022/ ===")

# Point to the folder (relative to project root, or use an absolute path)
na_ion_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'parameters', 'na_ion_chayambuka2022'
)

solver_naion = ImplicitBatterySolver(
    config, discretization, {},
    chemistry_folder=na_ion_folder   # <-- just pass the folder path!
)
v_na = solver_naion.get_exact_terminal_voltages(solver_naion.y, 0.001)
print(f"Na-ion OC terminal voltage: {v_na.item():.4f} V")

# ── 3. Manual chemistry dict (inspect before using) ──────────────────────────
print("\n=== 3. Manual load for inspection ===")
chem = load_chemistry_from_folder(na_ion_folder)
print(f"  Chemistry name : {chem['name']}")
print(f"  OCP anode type : {'CSV interpolant' if chem['ocp_n'].__name__ == 'ocp_n' else 'analytic'}")
print(f"  Ds_n callable  : {chem['params'].get('Ds_n') is not None and callable(chem['params']['Ds_n'])}")
print(f"  m_ref_n callable: {chem['params'].get('m_ref_n') is not None and callable(chem['params']['m_ref_n'])}")
print(f"  Ln = {chem['params']['Ln']:.2e} m")
print(f"  cs_max_n = {chem['params']['cs_max_n']:.1f} mol/m³")

# Pass the pre-loaded dict instead of the folder path
solver_naion2 = ImplicitBatterySolver(
    config, discretization, {},
    chemistry=chem   # <-- pre-loaded dict also works
)
v_na2 = solver_naion2.get_exact_terminal_voltages(solver_naion2.y, 0.001)
print(f"Na-ion (dict) OC terminal voltage: {v_na2.item():.4f} V")

print("\n[OK] All three loading methods work correctly.")
