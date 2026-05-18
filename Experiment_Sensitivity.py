"""
Experiment_Sensitivity.py
=========================
Local (one-at-a-time, OAT) Sensitivity Analysis of the VIBE pack simulation.

Physical motivation
-------------------
Battery pack behaviour is a nonlinear function of many electrochemical,
thermal, and electrical parameters.  A small mismatch in one parameter
(e.g. SEI exchange current) can amplify into a large current/temperature
divergence over many cycles ? a phenomenon called *heterogeneity propagation*.

This script quantifies that effect using the normalised sensitivity index:

    S_i = (Dy / y_0) / (Dp / p_0)              [dimensionless]

        = % change in output / % change in parameter

Interpretation:
    |S_i| > 1  ->  amplifying (the output is more sensitive than the perturbation)
    |S_i| < 1  ->  damping    (output is less sensitive than the perturbation)
    |S_i| = 2  ->  1% param change -> 2% output change

Parameters tested
-----------------
Chosen to cover all three physics layers of the VIBE DFN model:

  Electrochemical   : Ds_n, Ds_p, m_ref_n, m_ref_p, eps_e_n
  Aging             : kappa_sei  (SEI ionic conductivity -> SEI kinetics)
  Thermal           : hA
  Electrical network: R_contact, R_bus

Outputs measured
----------------
After N_CYCLES on a 2S2P series-first pack:

  capacity_fade_pct   ? pack-average capacity fade [%]
  max_temp_K          ? maximum cell temperature across all timesteps [K]
  current_imbalance   ? std(I_cells) at final step [A] (0 for S-P topology)
  sei_thickness_nm    ? pack-average SEI thickness [nm]

Reference
---------
Saltelli, A. et al. (2008) Global Sensitivity Analysis: The Primer.
Wiley, Chichester.

Tranter, T.G. et al. (2022) liionpack: A Python package for simulating
packs of batteries with PyBaMM. JOSS 7(70), 4051.

Usage
-----
    python Experiment_Sensitivity.py

Results are saved to:
    sensitivity_results/
        baseline_results.npz
        perturbed_<param>_<direction>.npz
        sensitivity_matrix.csv
        Sensitivity_Heatmap.png
        Sensitivity_Time_Curves.png
"""

import os
import gc
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend ? safe for long runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ?? Suppress interactive plt.show during simulation ?????????????????????????
_plt_show_orig = plt.show
plt.show = lambda: None

from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ???????????????????????????????????????????????????????????????????????????????
# USER CONFIGURATION
# ???????????????????????????????????????????????????????????????????????????????

N_CYCLES  = 25          # number of charge/discharge cycles per run
PERTURBATION = 0.05     # fractional perturbation: 5 %  ->  Dp/p = 0.05

# Pack layout
PACK_CONFIG = {
    'n_series':   2,
    'n_parallel': 2,
    'topology':   'series_first',   # ensures same current per series branch
    'device':     'cpu',
    'stress_options': {'enabled': False},
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n':  5, 'Nx_s':  5, 'Nx_p': 5,
    'Nsei': 1,
}

# CC-CV cycling protocol
CYCLE_CONFIG = {
    'cc_current':       -5.0,    # 1C charge (5 Ah cell)
    'cv_voltage':        4.2,
    'cutoff_current':    0.5,
    'discharge_current': 5.0,    # 1C discharge
    'min_voltage':       2.5,
    'max_voltage':       4.2,
    'n_cycles':          N_CYCLES,
}

# OutputSpec: save only what we need  (liionpack-inspired selective output)
SA_SPEC = OutputSpec([
    "cell_current",
    "soc",
    "temperature",
    "sei_thickness",
    "capacity_fade",
    "pack_voltage",
    "pack_current",
])

# ?? Parameters to perturb ????????????????????????????????????????????????????
# Each entry:  param_key -> (display_label, physical_unit)
# Only scalar parameters from get_standard_parameters() that have a direct
# physical effect on the outputs above are listed.

PARAMETERS = {
    # Electrochemical transport
    'Ds_n':    ('Solid diff. (anode)',    'm2/s'),
    'Ds_p':    ('Solid diff. (cathode)',  'm2/s'),
    'eps_e_n': ('Electrolyte porosity n', '-'),
    # Reaction kinetics
    'm_ref_n': ('Rxn rate const. (anode)',   'A m-2?(mol m-3)-1??'),
    'm_ref_p': ('Rxn rate const. (cathode)', 'A m-2?(mol m-3)-1??'),
    # Aging
    'kappa_sei': ('SEI ionic conductivity', 'S/m'),
    # Thermal
    'hA':        ('Convective coefficient', 'W/K'),
    # Electrical network
    'R_contact': ('Contact resistance',  'Ohm'),
    'R_bus':     ('Busbar resistance',   'Ohm'),
}

# Outputs to track
OUTPUTS = [
    'capacity_fade_pct',
    'max_temp_K',
    'current_imbalance_A',
    'sei_thickness_nm',
]

OUTPUT_LABELS = {
    'capacity_fade_pct':    'Capacity Fade [%]',
    'max_temp_K':           'Max Temperature [K]',
    'current_imbalance_A':  'Current Imbalance std [A]',
    'sei_thickness_nm':     'Avg SEI Thickness [nm]',
}

RESULTS_DIR = 'sensitivity_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ???????????????????????????????????????????????????????????????????????????????
# HELPER FUNCTIONS
# ???????????????????????????????????????????????????????????????????????????????

def _run_simulation(param_overrides: dict, run_tag: str) -> dict:
    """
    Run a full N_CYCLES simulation with the given global parameter overrides
    applied to every cell (uniform perturbation for OAT sensitivity).

    Returns a dict of scalar output metrics extracted from the .npz file.
    """
    n_s = PACK_CONFIG['n_series']
    n_p = PACK_CONFIG['n_parallel']

    # Build per-cell overrides dictionary: {(s,p): param_overrides}
    cell_overrides = {
        (s, p): dict(param_overrides)
        for s in range(n_s)
        for p in range(n_p)
    }

    t0 = time.perf_counter()
    solver = ImplicitBatterySolver(
        PACK_CONFIG,
        DISCRETIZATION,
        cell_overrides,
        initial_state_mode='fully_charged',
    )

    controller = build_controller('cycle_cccv', **CYCLE_CONFIG)

    # Resolve the output path BEFORE simulate() (which does an internal chdir/restore)
    _root = os.path.abspath(os.getcwd())
    solver.simulate(
        t_end=36_000_000,
        dt_init=1.0,
        controller=controller,
        dt_max=50.0,
        output_spec=SA_SPEC,
        run_name=run_tag,
    )

    elapsed = time.perf_counter() - t0

    # ?? Load .npz  (use absolute path ? simulate() does an internal chdir) ??
    npz_path = os.path.join(_root, 'simulation_result', run_tag, 'results',
                            'simulation_results.npz')
    data = np.load(npz_path)

    # Shapes: (n_steps, n_cells)
    n_cells = n_s * n_p

    # 1. Capacity fade: last step, average across cells
    fade = data['CapFade']                       # [n_steps, n_cells]
    capacity_fade_pct = float(np.mean(fade[-1])) # pack average at end

    # 2. Max temperature: global max across all steps & cells
    temp = data['Temp']                           # [n_steps, n_cells]
    max_temp_K = float(np.max(temp))

    # 3. Current imbalance: std of cell currents at the final timestep
    curr = data['Curr']                                  # [n_steps, n_cells]
    current_imbalance_A = float(np.std(curr[-1, :]))

    # 4. SEI thickness: last step, average across cells
    sei  = data['SEI_Thick']                             # [n_steps, n_cells]
    sei_thickness_nm = float(np.mean(sei[-1, :]))

    # 5. Save a compact per-run npz for time-series plots
    times_s = data['times']
    np.savez_compressed(
        os.path.join(_root, RESULTS_DIR, f'{run_tag}.npz'),
        times_h           = times_s / 3600.0,
        capacity_fade_pct = np.mean(fade, axis=1),
        max_temp_K        = np.max(temp,  axis=1),
        current_std_A     = np.std(curr,  axis=1),
        sei_avg_nm        = np.mean(sei,  axis=1),
    )

    del solver, controller, data
    gc.collect()

    metrics = {
        'capacity_fade_pct':   capacity_fade_pct,
        'max_temp_K':          max_temp_K,
        'current_imbalance_A': current_imbalance_A,
        'sei_thickness_nm':    sei_thickness_nm,
        'elapsed_s':           elapsed,
    }
    print(f"    -> fade={capacity_fade_pct:.3f}%  T_max={max_temp_K:.2f}K  "
          f"I_std={current_imbalance_A:.4f}A  SEI={sei_thickness_nm:.3f}nm  "
          f"[{elapsed:.0f}s]")
    return metrics


def _normalised_sensitivity(y_pos, y_neg, y_0, delta_frac):
    """
    Central-difference normalised sensitivity:

        S = (Dy / y_0) / (2 * Dp/p)
          = (y_pos - y_neg) / (2 * delta_frac * y_0)

    Falls back to forward-difference relative change if y_0 ? 0.
    """
    dy = y_pos - y_neg
    if abs(y_0) < 1e-15:
        return float(dy) / (2 * delta_frac + 1e-30)
    return float(dy) / (2 * delta_frac * abs(y_0))


# ???????????????????????????????????????????????????????????????????????????????
# MAIN SENSITIVITY LOOP
# ???????????????????????????????????????????????????????????????????????????????

def main():
    print("=" * 72)
    print(f" VIBE Pack Sensitivity Analysis  ?  {N_CYCLES} cycles, "
          f"+-{PERTURBATION*100:.0f}% OAT perturbation")
    print(f" Pack: {PACK_CONFIG['n_series']}Sx{PACK_CONFIG['n_parallel']}P  "
          f"topology={PACK_CONFIG['topology']}")
    print("=" * 72)

    # ?? STEP 1: Baseline run ?????????????????????????????????????????????????
    print("\n[1/3] Running BASELINE simulation ?")
    baseline = _run_simulation({}, 'baseline')
    print(f"  Baseline: {json.dumps({k: round(v,4) for k,v in baseline.items() if k != 'elapsed_s'})}")

    # ?? STEP 2: Perturbed runs ???????????????????????????????????????????????
    sensitivity_matrix = {}    # param_key -> {output_key -> S_i}
    perturbed_runs = {}        # param_key -> {'pos': metrics, 'neg': metrics}

    n_params = len(PARAMETERS)
    print(f"\n[2/3] Running {2*n_params} perturbed simulations ({n_params} params x 2 directions) ?\n")

    for idx, (param_key, (label, unit)) in enumerate(PARAMETERS.items()):
        print(f"  [{idx+1}/{n_params}]  Perturbing '{param_key}' ({label}) +-{PERTURBATION*100:.0f}% ?")

        # Get baseline value from standard parameters
        base_val = ImplicitBatterySolver.get_standard_parameters()[param_key]

        # Positive perturbation: p = p_0 * (1 + ?)
        pos_val = base_val * (1 + PERTURBATION)
        print(f"    (+) {param_key} = {pos_val:.4e} [{unit}]")
        metrics_pos = _run_simulation({param_key: pos_val}, f'perturb_{param_key}_pos')

        # Negative perturbation: p = p_0 * (1 - ?)
        neg_val = base_val * (1 - PERTURBATION)
        print(f"    (-) {param_key} = {neg_val:.4e} [{unit}]")
        metrics_neg = _run_simulation({param_key: neg_val}, f'perturb_{param_key}_neg')

        perturbed_runs[param_key] = {'pos': metrics_pos, 'neg': metrics_neg}

        # Compute normalised sensitivity for each output
        sensitivities = {}
        for out_key in OUTPUTS:
            S = _normalised_sensitivity(
                metrics_pos[out_key],
                metrics_neg[out_key],
                baseline[out_key],
                PERTURBATION,
            )
            sensitivities[out_key] = round(S, 4)
        sensitivity_matrix[param_key] = sensitivities
        print(f"    Sensitivities: {sensitivities}")

    # ?? STEP 3: Save results & plots ?????????????????????????????????????????
    print("\n[3/3] Saving results and generating plots ?")

    # 3a. Sensitivity matrix as CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'label'] + OUTPUTS)
        for param_key, sensitivities in sensitivity_matrix.items():
            label = PARAMETERS[param_key][0]
            row = [param_key, label] + [sensitivities[o] for o in OUTPUTS]
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # 3b. Save full sensitivity data as .npz for later analysis
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'sensitivity_matrix.npz'),
        param_keys   = np.array(list(sensitivity_matrix.keys())),
        output_keys  = np.array(OUTPUTS),
        S_matrix     = np.array([[sensitivity_matrix[p][o] for o in OUTPUTS]
                                  for p in sensitivity_matrix]),
        baseline     = np.array([baseline[o] for o in OUTPUTS]),
    )

    # 3c. Heatmap of |S_i| for all params x outputs
    plt.show = _plt_show_orig

    param_labels  = [PARAMETERS[k][0] for k in sensitivity_matrix]
    output_labels = [OUTPUT_LABELS[o] for o in OUTPUTS]
    S_matrix = np.array([[sensitivity_matrix[p][o] for o in OUTPUTS]
                          for p in sensitivity_matrix])

    fig, ax = plt.subplots(figsize=(max(7, len(OUTPUTS)*2.5), max(5, len(PARAMETERS)*0.6)))

    # Use diverging colormap centred at 0
    vmax = max(1.0, np.max(np.abs(S_matrix)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(S_matrix, cmap='RdBu_r', norm=norm, aspect='auto')

    ax.set_xticks(range(len(OUTPUTS)))
    ax.set_xticklabels(output_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels, fontsize=9)

    for i, p in enumerate(sensitivity_matrix):
        for j, o in enumerate(OUTPUTS):
            val = sensitivity_matrix[p][o]
            color = 'white' if abs(val) > 0.5 * vmax else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalised sensitivity  S_i = (Dy/y?) / (Dp/p?)')
    ax.set_title(f'Pack Sensitivity Matrix ? {N_CYCLES} cycles, +-{PERTURBATION*100:.0f}% OAT\n'
                 f'Pack: {PACK_CONFIG["n_series"]}Sx{PACK_CONFIG["n_parallel"]}P  '
                 f'[topology={PACK_CONFIG["topology"]}]',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, 'Sensitivity_Heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {heatmap_path}")
    plt.close()

    # 3d. Time-series plots: how each output evolves for the most sensitive param
    #     Find the parameter with the largest |S| for capacity fade
    top_param = max(sensitivity_matrix,
                    key=lambda p: abs(sensitivity_matrix[p]['capacity_fade_pct']))
    print(f"  Most sensitive parameter for capacity fade: {top_param} "
          f"(S={sensitivity_matrix[top_param]['capacity_fade_pct']:.3f})")

    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle(
        f'Sensitivity Time Curves ? most sensitive param: '
        f'{PARAMETERS[top_param][0]}\n'
        f'(+-{PERTURBATION*100:.0f}% perturbation, {N_CYCLES} cycles)',
        fontsize=12
    )

    pairs = [
        ('capacity_fade_pct', 'Capacity Fade [%]'),
        ('max_temp_K',        'Max Temperature [K]'),
        ('current_std_A',     'Current Std [A]'),
        ('sei_avg_nm',        'Avg SEI Thickness [nm]'),
    ]

    for ax, (key, ylabel) in zip(axs2.flatten(), pairs):
        for tag, lbl, ls in [
            ('baseline',                        'Baseline',  '-'),
            (f'perturb_{top_param}_pos',        f'+{PERTURBATION*100:.0f}%', '--'),
            (f'perturb_{top_param}_neg',        f'-{PERTURBATION*100:.0f}%', ':'),
        ]:
            npz_path = os.path.join(RESULTS_DIR, f'{tag}.npz')
            if not os.path.exists(npz_path):
                continue
            d = np.load(npz_path)
            arr_key = {
                'capacity_fade_pct': 'capacity_fade_pct',
                'max_temp_K':        'max_temp_K',
                'current_std_A':     'current_std_A',
                'sei_avg_nm':        'sei_avg_nm',
            }[key]
            ax.plot(d['times_h'], d[arr_key], linestyle=ls, label=lbl)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time [h]')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, 'Sensitivity_Time_Curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {curves_path}")
    plt.close()

    # 3e. Print ranked summary
    print("\n" + "=" * 72)
    print(f"  SENSITIVITY SUMMARY  (output: capacity_fade_pct, {N_CYCLES} cycles)")
    print("=" * 72)
    ranked = sorted(sensitivity_matrix.items(),
                    key=lambda x: abs(x[1]['capacity_fade_pct']),
                    reverse=True)
    for rank, (p, s) in enumerate(ranked, 1):
        lbl = PARAMETERS[p][0]
        print(f"  #{rank:2d}  {lbl:<35s}  S = {s['capacity_fade_pct']:+.3f}")
    print("=" * 72)
    print("\nSensitivity analysis complete!")
    print(f"  Results directory : {os.path.abspath(RESULTS_DIR)}")
    print(f"  Sensitivity matrix: {csv_path}")


# ???????????????????????????????????????????????????????????????????????????????

if __name__ == '__main__':
    main()
