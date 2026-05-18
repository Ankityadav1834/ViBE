"""
Experiment_Heterogeneity.py
============================
Pack-level heterogeneity propagation study for the VIBE framework.

Scientific question
-------------------
"If one cell in a pack has a manufacturing defect (e.g. higher contact
resistance, lower cooling, slower diffusivity), how does that defect
propagate to affect neighbouring cells and the pack as a whole?"

This is fundamentally different from Experiment_Sensitivity.py, which
perturbed ALL cells uniformly. Here we perturb ONE cell and measure:

  1. Current redistribution across the pack
  2. Temperature divergence between cells
  3. SEI growth divergence (differential aging)
  4. Capacity fade divergence
  5. Propagation index P_i = (max divergence) / (initial mismatch)

Physical mechanism of propagation (series-first topology)
----------------------------------------------------------
In a series-first pack (strings in parallel), each string must carry the
same voltage (KVL). If cell (0,p*) has higher internal resistance:

  - It produces a larger voltage drop at the same current
  - KCL forces the parallel strings to re-balance
  - String p* carries LESS current, string p!=p* carries MORE
  - The cells in the neighbouring string run hotter
  - Hotter cells age faster -> further resistance increase
  - Positive feedback loop: small initial mismatch amplifies over cycles

This experiment QUANTIFIES that amplification.

Cell ID mapping (2S2P series-first pack)
-----------------------------------------
       Pack terminal +
            |
    String 0          String 1
 [Cell (0,0)]      [Cell (0,1)]   <- Row 0
 [Cell (1,0)]      [Cell (1,1)]   <- Row 1
            |
       Pack terminal -

Cell indices in output arrays: (n_series * n_parallel)
  cell 0 = (0,0) = string 0, row 0
  cell 1 = (0,1) = string 1, row 0
  cell 2 = (1,0) = string 0, row 1
  cell 3 = (1,1) = string 1, row 1

Defect cell is always (0,1) = cell index 1 (string 1, row 0).
Healthy cells: 0, 2, 3

Scenarios tested
----------------
Each scenario introduces a single defect in cell (0,1):

  R_contact  +30%  : higher weld resistance (common manufacturing defect)
  hA         -50%  : impaired cooling (blocked airflow / thermal interface)
  Ds_n       -20%  : lower anode diffusivity (particle cracking / aging)
  m_ref_n    -20%  : lower exchange current density (surface contamination)

Reference
---------
Baumhofer, T. et al. (2014). Production caused variation in capacity aging
trend. Journal of Power Sources, 247, 332-338. DOI:10.1016/j.jpowsour.2013.08.108

Gogoana, R. et al. (2014). Internal resistance matching for parallel-connected
lithium-ion cells. Journal of Power Sources, 252, 8-13.
DOI:10.1016/j.jpowsour.2013.11.101

Usage
-----
    python Experiment_Heterogeneity.py

Outputs
-------
    heterogeneity_results/
        baseline.npz
        scenario_<name>.npz
        Heterogeneity_Divergence.png   -- per-scenario divergence plots
        Propagation_Index.png          -- bar chart of propagation indices
        heterogeneity_summary.csv      -- tabular results
"""

import os
import gc
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

N_CYCLES = 10      # number of cycles (increase to 25 for thesis runs)
N_SERIES  = 2
N_PARALLEL = 2

PACK_CONFIG = {
    'n_series':   N_SERIES,
    'n_parallel': N_PARALLEL,
    'topology':   'series_first',
    'device':     'cpu',
    'stress_options': {'enabled': False},
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n':  5, 'Nx_s':  5, 'Nx_p': 5,
    'Nsei': 1,
}

CYCLE_CONFIG = {
    'cc_current':       -5.0,   # 1C charge
    'cv_voltage':        4.2,
    'cutoff_current':    0.5,
    'discharge_current': 5.0,   # 1C discharge
    'min_voltage':       2.5,
    'max_voltage':       4.2,
    'n_cycles':          N_CYCLES,
}

# Defect cell: (series_row, parallel_string)
# For a 2S2P pack, cell (0,1) is the top cell in string 1
DEFECT_CELL = (0, 1)

# Scenarios: each defines one parameter override on the defect cell only
# Format: scenario_name -> (param_key, multiplier, description)
SCENARIOS = {
    'R_contact_high': ('R_contact', 1.30, '+30% contact resistance (weld defect)'),
    'hA_low':         ('hA',        0.50, '-50% cooling coefficient (thermal blockage)'),
    'Ds_n_low':       ('Ds_n',      0.80, '-20% anode solid diffusivity (particle cracking)'),
    'm_ref_n_low':    ('m_ref_n',   0.80, '-20% exchange current density (surface film)'),
}

# Which outputs to save
HETERO_SPEC = OutputSpec([
    'cell_current',
    'temperature',
    'sei_thickness',
    'capacity_fade',
    'soc',
    'pack_voltage',
    'pack_current',
])

RESULTS_DIR = 'heterogeneity_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cell names for labelling
N_CELLS = N_SERIES * N_PARALLEL
CELL_NAMES = [f'({s},{p})' for s in range(N_SERIES) for p in range(N_PARALLEL)]
DEFECT_IDX = DEFECT_CELL[0] * N_PARALLEL + DEFECT_CELL[1]
HEALTHY_IDX = [i for i in range(N_CELLS) if i != DEFECT_IDX]

# ============================================================================
# SIMULATION HELPER
# ============================================================================

def run_scenario(overrides, run_tag):
    """
    Run N_CYCLES with the given per-cell overrides.
    Returns the compact metrics dict extracted from the .npz.
    """
    _root = os.path.abspath(os.getcwd())

    solver = ImplicitBatterySolver(
        PACK_CONFIG, DISCRETIZATION, overrides,
        initial_state_mode='fully_charged',
    )
    ctrl = build_controller('cycle_cccv', **CYCLE_CONFIG)

    t0 = time.perf_counter()
    solver.simulate(
        t_end=36_000_000,
        dt_init=1.0,
        controller=ctrl,
        dt_max=50.0,
        output_spec=HETERO_SPEC,
        run_name=run_tag,
    )
    elapsed = time.perf_counter() - t0

    npz_path = os.path.join(_root, 'simulation_result', run_tag, 'results',
                            'simulation_results.npz')
    data = np.load(npz_path)

    times_s  = data['times']                      # [n_steps]
    curr     = data['Curr']                       # [n_steps, n_cells]
    temp     = data['Temp']                       # [n_steps, n_cells]
    sei      = data['SEI_Thick']                  # [n_steps, n_cells]
    fade     = data['CapFade']                    # [n_steps, n_cells]
    soc      = data['SOC']                        # [n_steps, n_cells]

    # Save compact per-scenario npz
    compact_path = os.path.join(RESULTS_DIR, f'{run_tag}.npz')
    np.savez_compressed(
        compact_path,
        times_h  = times_s / 3600.0,
        curr     = curr,
        temp     = temp,
        sei      = sei,
        fade     = fade,
        soc      = soc,
    )

    del solver, ctrl, data
    gc.collect()

    return {
        'times_h': times_s / 3600.0,
        'curr':    curr,
        'temp':    temp,
        'sei':     sei,
        'fade':    fade,
        'soc':     soc,
        'elapsed': elapsed,
    }


def compute_divergence_metrics(base, scen):
    """
    For a given scenario result vs baseline, compute:

    current_imbalance_end  : std(I_cells) at final timestep [A]
    temp_divergence_end    : max(T) - min(T) at final timestep [K]
    sei_divergence_end     : max(SEI) - min(SEI) at final timestep [nm]
    fade_divergence_end    : max(fade) - min(fade) at final timestep [%]
    defect_excess_temp     : T_defect - mean(T_healthy) at peak [K]
    defect_excess_sei      : SEI_defect - mean(SEI_healthy) at end [nm]
    current_shift_defect   : mean(I_defect_scen) - mean(I_defect_base) [A]
    """
    def _last(arr): return arr[-1]

    metrics = {}

    # At final step
    I_end_base = _last(base['curr'])   # [n_cells]
    I_end_scen = _last(scen['curr'])
    T_end_base = _last(base['temp'])
    T_end_scen = _last(scen['temp'])
    S_end_base = _last(base['sei'])
    S_end_scen = _last(scen['sei'])
    F_end_base = _last(base['fade'])
    F_end_scen = _last(scen['fade'])

    # Current imbalance (scenario only -- baseline should be zero for series-first)
    metrics['current_imbalance_base_A'] = float(np.std(I_end_base))
    metrics['current_imbalance_scen_A'] = float(np.std(I_end_scen))

    # Temperature spread
    metrics['temp_spread_base_K']  = float(np.max(T_end_base) - np.min(T_end_base))
    metrics['temp_spread_scen_K']  = float(np.max(T_end_scen) - np.min(T_end_scen))

    # SEI spread
    metrics['sei_spread_base_nm']  = float(np.max(S_end_base) - np.min(S_end_base))
    metrics['sei_spread_scen_nm']  = float(np.max(S_end_scen) - np.min(S_end_scen))

    # Capacity fade spread
    metrics['fade_spread_base_pct'] = float(np.max(F_end_base) - np.min(F_end_base))
    metrics['fade_spread_scen_pct'] = float(np.max(F_end_scen) - np.min(F_end_scen))

    # Defect cell vs healthy cells
    T_defect_scen    = scen['temp'][:, DEFECT_IDX]      # [n_steps]
    T_healthy_scen   = scen['temp'][:, HEALTHY_IDX]     # [n_steps, 3]
    metrics['defect_excess_temp_K'] = float(
        np.max(T_defect_scen - T_healthy_scen.mean(axis=1))
    )

    sei_defect_end   = scen['sei'][-1, DEFECT_IDX]
    sei_healthy_end  = scen['sei'][-1, HEALTHY_IDX].mean()
    metrics['defect_excess_sei_nm']  = float(sei_defect_end - sei_healthy_end)

    fade_defect_end  = scen['fade'][-1, DEFECT_IDX]
    fade_healthy_end = scen['fade'][-1, HEALTHY_IDX].mean()
    metrics['defect_excess_fade_pct'] = float(fade_defect_end - fade_healthy_end)

    # Current shift in defect cell
    I_defect_base = base['curr'][:, DEFECT_IDX].mean()
    I_defect_scen_val = scen['curr'][:, DEFECT_IDX].mean()
    metrics['current_shift_A'] = float(I_defect_scen_val - I_defect_base)

    # Propagation index: (excess fade in defect cell) / (param mismatch fraction)
    # Note: filled in by caller who knows the mismatch fraction
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 72)
    print(f"  VIBE Pack Heterogeneity Propagation Study -- {N_CYCLES} cycles")
    print(f"  Pack: {N_SERIES}S x {N_PARALLEL}P (series-first)")
    print(f"  Defect cell: {DEFECT_CELL} (cell index {DEFECT_IDX})")
    print("=" * 72)

    # ---- STEP 1: Baseline (all cells identical) ----
    print(f"\n[1/{len(SCENARIOS)+1}] Running BASELINE (all cells identical) ...")
    base = run_scenario({}, 'baseline')
    print(f"  Done in {base['elapsed']:.0f}s")
    print(f"  End-of-run current distribution: {base['curr'][-1]}")
    print(f"  End-of-run temperatures [C]:    {base['temp'][-1] - 273.15}")
    print(f"  End-of-run SEI thickness [nm]:  {base['sei'][-1]}")

    # ---- STEP 2: Defect scenarios ----
    all_scenario_results = {}
    all_metrics = {}

    for idx, (scenario_name, (param_key, multiplier, description)) in enumerate(SCENARIOS.items()):
        print(f"\n[{idx+2}/{len(SCENARIOS)+1}] Scenario: {scenario_name}")
        print(f"  Description: {description}")

        baseline_val = ImplicitBatterySolver.get_standard_parameters()[param_key]
        defect_val   = baseline_val * multiplier
        mismatch_pct = (multiplier - 1.0) * 100

        print(f"  {param_key}: {baseline_val:.4e} -> {defect_val:.4e}  "
              f"({mismatch_pct:+.0f}% on cell {DEFECT_CELL} only)")

        overrides = {DEFECT_CELL: {param_key: defect_val}}
        scen = run_scenario(overrides, f'scenario_{scenario_name}')
        all_scenario_results[scenario_name] = scen

        print(f"  Done in {scen['elapsed']:.0f}s")
        print(f"  End-of-run currents [A]:       {scen['curr'][-1]}")
        print(f"  End-of-run temperatures [C]:   {scen['temp'][-1] - 273.15}")
        print(f"  End-of-run SEI [nm]:           {scen['sei'][-1]}")

        # Compute divergence metrics
        metrics = compute_divergence_metrics(base, scen)
        mismatch_frac = abs(multiplier - 1.0)
        if mismatch_frac > 0 and abs(metrics['defect_excess_fade_pct']) > 1e-12:
            metrics['propagation_index'] = metrics['defect_excess_fade_pct'] / mismatch_frac
        else:
            metrics['propagation_index'] = 0.0

        all_metrics[scenario_name] = metrics

        print(f"  Defect cell excess temp:       {metrics['defect_excess_temp_K']:+.3f} K")
        print(f"  Defect cell excess SEI:        {metrics['defect_excess_sei_nm']:+.4f} nm")
        print(f"  Temperature spread (scenario): {metrics['temp_spread_scen_K']:.3f} K")
        print(f"  Current imbalance (scenario):  {metrics['current_imbalance_scen_A']:.4f} A")
        print(f"  Propagation index:             {metrics['propagation_index']:.4f}")

    # ---- STEP 3: Save summary CSV ----
    csv_path = os.path.join(RESULTS_DIR, 'heterogeneity_summary.csv')
    metric_keys = list(next(iter(all_metrics.values())).keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'description'] + metric_keys)
        for sname, metrics in all_metrics.items():
            desc = SCENARIOS[sname][2]
            writer.writerow([sname, desc] + [round(metrics[k], 6) for k in metric_keys])
    print(f"\n  Saved: {csv_path}")

    # ---- STEP 4: Divergence time-series plots ----
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'Pack Heterogeneity Propagation -- {N_CYCLES} Cycles\n'
        f'Defect in cell {DEFECT_CELL} only  |  Healthy cells: '
        f'{[CELL_NAMES[i] for i in HEALTHY_IDX]}',
        fontsize=13, fontweight='bold'
    )

    cell_colors  = ['steelblue', 'firebrick', 'seagreen', 'darkorange']
    cell_styles  = ['-', '--', '-.', ':']

    n_scen = len(SCENARIOS)
    gs = gridspec.GridSpec(3, n_scen, hspace=0.45, wspace=0.35)

    for col, (scenario_name, (param_key, multiplier, description)) in enumerate(SCENARIOS.items()):
        scen = all_scenario_results[scenario_name]
        th   = scen['times_h']
        th_b = base['times_h']

        # Row 0: Current per cell (scenario)
        ax0 = fig.add_subplot(gs[0, col])
        for cell_idx in range(N_CELLS):
            lw = 2.5 if cell_idx == DEFECT_IDX else 1.0
            ax0.plot(th, scen['curr'][:, cell_idx],
                     color=cell_colors[cell_idx],
                     linestyle=cell_styles[cell_idx],
                     linewidth=lw,
                     label=f'Cell{CELL_NAMES[cell_idx]}'
                           + (' [DEFECT]' if cell_idx == DEFECT_IDX else ''))
        ax0.set_title(f'{scenario_name}\n({param_key} x{multiplier})', fontsize=8.5)
        ax0.set_ylabel('Current [A]' if col == 0 else '')
        ax0.legend(fontsize=6, loc='upper right')
        ax0.grid(True, linestyle='--', alpha=0.3)

        # Row 1: Temperature per cell
        ax1 = fig.add_subplot(gs[1, col])
        for cell_idx in range(N_CELLS):
            lw = 2.5 if cell_idx == DEFECT_IDX else 1.0
            ax1.plot(th, scen['temp'][:, cell_idx] - 273.15,
                     color=cell_colors[cell_idx],
                     linestyle=cell_styles[cell_idx],
                     linewidth=lw)
        # Also plot baseline as thin dashed grey
        for cell_idx in range(N_CELLS):
            ax1.plot(th_b, base['temp'][:, cell_idx] - 273.15,
                     color='grey', linestyle=':', linewidth=0.7, alpha=0.5)
        ax1.set_ylabel('Temperature [C]' if col == 0 else '')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Row 2: SEI thickness divergence (defect cell minus mean healthy)
        ax2 = fig.add_subplot(gs[2, col])
        t_min_len = min(scen['sei'].shape[0], base['sei'].shape[0])
        sei_defect  = scen['sei'][:t_min_len, DEFECT_IDX]
        sei_healthy = scen['sei'][:t_min_len, HEALTHY_IDX].mean(axis=1)
        sei_base_d  = base['sei'][:t_min_len, DEFECT_IDX]
        sei_base_h  = base['sei'][:t_min_len, HEALTHY_IDX].mean(axis=1)
        ax2.plot(th[:t_min_len], (sei_defect - sei_healthy),
                 color='firebrick', linewidth=1.8, label='Defect - Healthy (scenario)')
        ax2.plot(th_b[:t_min_len], (sei_base_d - sei_base_h),
                 color='grey', linewidth=0.8, linestyle='--', label='Baseline')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('DSEI_defect - DSEI_healthy [nm]' if col == 0 else '')
        ax2.set_xlabel('Time [h]')
        ax2.legend(fontsize=6)
        ax2.grid(True, linestyle='--', alpha=0.3)

    plot_path = os.path.join(RESULTS_DIR, 'Heterogeneity_Divergence.png')
    plt.savefig(plot_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ---- STEP 5: Propagation index bar chart ----
    fig2, axs2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle(
        f'Heterogeneity Propagation Summary -- {N_CYCLES} Cycles\n'
        f'All defects applied to cell {DEFECT_CELL} only',
        fontsize=12, fontweight='bold'
    )

    snames      = list(all_metrics.keys())
    descs       = [SCENARIOS[s][2][:30] + '...' if len(SCENARIOS[s][2]) > 30
                   else SCENARIOS[s][2] for s in snames]
    excess_temp = [all_metrics[s]['defect_excess_temp_K']  for s in snames]
    excess_sei  = [all_metrics[s]['defect_excess_sei_nm']  for s in snames]
    curr_shift  = [all_metrics[s]['current_shift_A']       for s in snames]

    x = np.arange(len(snames))

    axs2[0].bar(x, excess_temp, color='firebrick', alpha=0.8)
    axs2[0].set_xticks(x); axs2[0].set_xticklabels(snames, rotation=25, ha='right', fontsize=9)
    axs2[0].set_ylabel('Defect cell excess temperature [K]')
    axs2[0].set_title('Thermal Penalty on Defect Cell')
    axs2[0].grid(axis='y', linestyle='--', alpha=0.4)
    axs2[0].axhline(0, color='black', linewidth=0.8)

    axs2[1].bar(x, excess_sei, color='steelblue', alpha=0.8)
    axs2[1].set_xticks(x); axs2[1].set_xticklabels(snames, rotation=25, ha='right', fontsize=9)
    axs2[1].set_ylabel('Defect cell excess SEI growth [nm]')
    axs2[1].set_title('Accelerated Aging in Defect Cell')
    axs2[1].grid(axis='y', linestyle='--', alpha=0.4)
    axs2[1].axhline(0, color='black', linewidth=0.8)

    axs2[2].bar(x, curr_shift, color='darkorange', alpha=0.8)
    axs2[2].set_xticks(x); axs2[2].set_xticklabels(snames, rotation=25, ha='right', fontsize=9)
    axs2[2].set_ylabel('Mean current shift in defect cell [A]')
    axs2[2].set_title('Current Redistribution (Defect Cell)')
    axs2[2].grid(axis='y', linestyle='--', alpha=0.4)
    axs2[2].axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'Propagation_Index.png')
    plt.savefig(bar_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {bar_path}")

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print(f"  HETEROGENEITY SUMMARY  ({N_CYCLES} cycles, defect on cell {DEFECT_CELL})")
    print("=" * 72)
    header = f"{'Scenario':<22} {'dT_defect[K]':>14} {'dSEI_defect[nm]':>16} {'dI_shift[A]':>13} {'P_index':>9}"
    print(header)
    print("-" * 72)
    for sname in snames:
        m = all_metrics[sname]
        print(f"  {sname:<20}  {m['defect_excess_temp_K']:>12.3f}  "
              f"{m['defect_excess_sei_nm']:>14.4f}  "
              f"{m['current_shift_A']:>11.4f}  "
              f"{m['propagation_index']:>8.4f}")
    print("=" * 72)
    print(f"\n  Results directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"\nPhysical interpretation:")
    print(f"  - dT_defect > 0 : defect cell runs hotter than healthy cells")
    print(f"  - dSEI > 0      : defect cell ages faster (more SEI growth)")
    print(f"  - dI_shift < 0  : defect cell draws LESS current (higher impedance)")
    print(f"  - P_index       : excess aging / mismatch fraction (amplification)")


if __name__ == '__main__':
    main()
