"""
Fast Charging Comparison: MPC vs CC-CV
=======================================
Runs two charging simulations from a fully-discharged state:

  Run A – Baseline CC-CV  (1C constant charge → CV hold)
  Run B – MPC Fast Charge  (up to 3C, constrained by voltage / temperature /
                             SEI growth rate / surface stress)

Both runs use identical battery config and discretisation.
Results are saved to simulation_result/<script_name>/results/ and a comparison
plot is saved as fast_charge_comparison.png in the same directory.

Usage
-----
    python test_fast_charging.py

Dependencies
------------
    scipy   (for MPC optimisation)
    matplotlib, numpy, pandas (standard)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend so the script works on servers
import matplotlib.pyplot as plt

# ── battery framework ────────────────────────────────────────────────────────
from main import ImplicitBatterySolver
from controllers import build_controller

# ── simulation settings ──────────────────────────────────────────────────────
BATTERY_CONFIG = {
    'n_series': 1,        # single cell for clarity
    'n_parallel': 1,
    'electrolyte_spatial_method': 'finite_volume',
    'solid_spatial_method': 'chebyshev',
    'stress_options': {'enabled': True, 'initial': 0.0, 'scale': 1e6,
                       'diffusivity': 1e-12, 'relaxation': 1e-4,
                       'coupling': 1.0, 'force_area': 0.1027},
    'sei_options': {'enabled': True},
}

DISCRETISATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n': 10, 'Nx_s': 10, 'Nx_p': 10,
    'Nsei': 1,
}

INITIAL_STATE_OPTIONS = {
    'cutoff_voltage': 2.5,
    'discharge_current': 10.0,
    'dt': 2.0,
    'max_time': 6000.0,
    'coarse_dt': 15.0,
    'refine_margin': 0.10,
}

# Nominal cell capacity: ~3 Ah (set to match standard parameters)
Q_NOM_AH = 3.0
I_1C      = Q_NOM_AH        # A

MAX_CHARGE_TIME = 5 * 3600   # 5 hours hard cap
DT_INIT         = 1.0        # initial time step [s]
DT_MAX          = 30.0       # max time step [s]

# ── CC-CV baseline parameters ────────────────────────────────────────────────
CCCV_CONFIG = {
    'cc_current':      -1.0 * I_1C,    # 1C charge
    'cv_voltage':       4.2,
    'cutoff_current':   0.1 * I_1C,   # C/10 cutoff
    'min_voltage':      2.5,
    'max_voltage':      4.2,
}

# ── MPC fast-charge parameters ───────────────────────────────────────────────
MPC_CONFIG = {
    'n_parallel':    1,
    'Q_nom_cell':    Q_NOM_AH,
    'I_max_rate':    3.0,        # up to 3C
    'I_min_rate':    0.1,        # at least 0.1C to prevent stall
    'v_max':         4.2,
    't_max_degC':    45.0,
    'sigma_max_MPa': 180.0,      # tighter stress limit for demo
    'soc_target':    0.95,
    'Np':            10,         # 10-step horizon
    'dt_mpc':        15.0,       # 15 s prediction step
    'min_voltage':   2.5,
    'max_voltage':   4.2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract comparison metrics from CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_run_metrics(results_dir: str) -> dict:
    df_all  = pd.read_csv(os.path.join(results_dir, 'all_results.csv'))
    df_sim  = pd.read_csv(os.path.join(results_dir, 'sim_results.csv'))
    cell_id = 0

    def col(name, cell=cell_id):
        return df_all[(df_all['name'] == name) & (df_all['cell'] == cell)].set_index('time')['value']

    return {
        'time':        df_sim['time'].values,
        'pack_voltage': df_sim['pack_voltage'].values,
        'pack_current': df_sim['pack_current'].values,
        'soc':         col('soc').reindex(df_sim['time']).ffill().values,
        'temperature': col('temperature').reindex(df_sim['time']).ffill().values - 273.15,
        'sei_nm':      col('sei_thickness_nm').reindex(df_sim['time']).ffill().values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run A: CC-CV baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_cccv(label='run_cccv') -> str:
    print("=" * 60)
    print(f"  RUN A — CC-CV Baseline  (script: {label})")
    print("=" * 60)
    t0 = time.time()

    # Patch argv so the solver saves results under label/
    sys.argv = [label + '.py']

    solver = ImplicitBatterySolver(
        BATTERY_CONFIG, DISCRETISATION,
        overrides={},
        initial_state_mode='fully_discharged',
        initial_state_options=INITIAL_STATE_OPTIONS,
    )

    controller = build_controller('cc_cv', **CCCV_CONFIG)
    solver.simulate(
        t_end=MAX_CHARGE_TIME,
        dt_init=DT_INIT,
        controller=controller,
        dt_max=DT_MAX,
    )

    elapsed = time.time() - t0
    print(f"  CC-CV wall-clock time: {elapsed:.1f}s\n")

    # Restore results path
    results_dir = os.path.join('simulation_result', label, 'results')
    return results_dir


# ─────────────────────────────────────────────────────────────────────────────
# Run B: MPC fast charge
# ─────────────────────────────────────────────────────────────────────────────

def run_mpc(label='run_mpc') -> str:
    print("=" * 60)
    print(f"  RUN B — MPC Fast Charge  (script: {label})")
    print("=" * 60)
    t0 = time.time()

    sys.argv = [label + '.py']

    solver = ImplicitBatterySolver(
        BATTERY_CONFIG, DISCRETISATION,
        overrides={},
        initial_state_mode='fully_discharged',
        initial_state_options=INITIAL_STATE_OPTIONS,
    )

    controller = build_controller('fast_charge_mpc', **MPC_CONFIG)
    solver.simulate(
        t_end=MAX_CHARGE_TIME,
        dt_init=DT_INIT,
        controller=controller,
        dt_max=DT_MAX,
    )

    elapsed = time.time() - t0
    print(f"  MPC wall-clock time: {elapsed:.1f}s\n")

    results_dir = os.path.join('simulation_result', label, 'results')
    return results_dir


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(metrics_cccv: dict, metrics_mpc: dict, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle('Fast Charging Comparison: MPC vs CC-CV Baseline',
                 fontsize=14, fontweight='bold')

    t_cc  = metrics_cccv['time'] / 60.0
    t_mpc = metrics_mpc['time']  / 60.0

    panels = [
        ('pack_voltage', 'Pack Voltage [V]',       [[2.5, 4.25]], True),
        ('pack_current', 'Charging Current [A]',   [[-I_1C*3.2, 0.2]], False),
        ('soc',          'State of Charge [–]',    [[0.0, 1.05]], False),
        ('temperature',  'Temperature [°C]',       [[20, 50]], True),
        ('sei_nm',       'SEI Thickness [nm]',     [[0, None]], False),
    ]

    ax_flat = axes.flatten()
    for idx, (key, ylabel, ylims, draw_limit) in enumerate(panels):
        ax = ax_flat[idx]

        y_cc  = metrics_cccv.get(key)
        y_mpc = metrics_mpc.get(key)

        if y_cc is not None:
            ax.plot(t_cc,  y_cc,  lw=1.8, color='steelblue',  label='CC-CV (1C)')
        if y_mpc is not None:
            ax.plot(t_mpc, y_mpc, lw=1.8, color='darkorange', label='MPC (≤3C)')

        ax.set_xlabel('Time [min]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35, linestyle='--')

        if draw_limit:
            if key == 'pack_voltage':
                ax.axhline(4.2, color='red', lw=0.9, ls='--', label='V_max = 4.2 V')
            elif key == 'temperature':
                ax.axhline(45.0, color='red', lw=0.9, ls='--', label='T_max = 45 °C')

        if ylims[0][0] is not None:
            ax.set_ylim(bottom=ylims[0][0])
        if ylims[0][1] is not None:
            ax.set_ylim(top=ylims[0][1])

    # Summary table in the 6th panel
    ax_sum = ax_flat[5]
    ax_sum.axis('off')

    def charge_time_min(metrics):
        """Time to reach SOC ≥ 0.95, or total sim time."""
        soc = metrics.get('soc', np.zeros(1))
        if soc is None or len(soc) == 0:
            return np.nan
        idx = np.searchsorted(soc, 0.95)
        if idx >= len(metrics['time']):
            return metrics['time'][-1] / 60.0
        return metrics['time'][idx] / 60.0

    def max_temp(metrics):
        t = metrics.get('temperature')
        return float(np.nanmax(t)) if t is not None and len(t) else np.nan

    def final_sei(metrics):
        s = metrics.get('sei_nm')
        return float(s[-1]) if s is not None and len(s) else np.nan

    rows = [
        ['Metric', 'CC-CV (1C)', 'MPC (≤3C)'],
        ['Time to 95% SOC [min]',
         f'{charge_time_min(metrics_cccv):.1f}',
         f'{charge_time_min(metrics_mpc):.1f}'],
        ['Peak Temperature [°C]',
         f'{max_temp(metrics_cccv):.1f}',
         f'{max_temp(metrics_mpc):.1f}'],
        ['Final SEI thickness [nm]',
         f'{final_sei(metrics_cccv):.3f}',
         f'{final_sei(metrics_mpc):.3f}'],
    ]
    table = ax_sum.table(cellText=rows[1:], colLabels=rows[0],
                         cellLoc='center', loc='center',
                         bbox=[0.0, 0.3, 1.0, 0.55])
    table.auto_set_font_size(True)
    ax_sum.set_title('Summary', fontsize=10, pad=4)

    fig.savefig(out_path, dpi=150)
    print(f"\n✓  Comparison plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run both simulations
    results_cccv = run_cccv(label='fc_cccv_baseline')
    results_mpc  = run_mpc(label='fc_mpc_fast')

    # Load results
    print("\nLoading results …")
    try:
        m_cccv = load_run_metrics(results_cccv)
    except Exception as e:
        print(f"  [WARNING] Could not load CC-CV metrics: {e}")
        m_cccv = {}

    try:
        m_mpc = load_run_metrics(results_mpc)
    except Exception as e:
        print(f"  [WARNING] Could not load MPC metrics: {e}")
        m_mpc = {}

    # Save comparison plot next to this script
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fast_charge_comparison.png')
    plot_comparison(m_cccv, m_mpc, plot_path)

    print("\nDone.")
