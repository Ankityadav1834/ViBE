"""
Experiment_Heterogeneity.py
============================
Pack-level heterogeneity propagation study for the VIBE framework.
"""

import os
import gc
import sys
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent folder to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

N_CYCLES = 10      # number of cycles
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

DEFECT_CELL = (0, 1)

SCENARIOS = {
    'R_contact_high': ('R_contact', 1.30, '+30% contact resistance (weld defect)'),
    'hA_low':         ('hA',        0.50, '-50% cooling coefficient (thermal blockage)'),
    'Ds_n_low':       ('Ds_n',      0.80, '-20% anode solid diffusivity (particle cracking)'),
    'm_ref_n_low':    ('m_ref_n',   0.80, '-20% exchange current density (surface film)'),
}

HETERO_SPEC = OutputSpec([
    'cell_current',
    'temperature',
    'sei_thickness',
    'capacity_fade',
    'soc',
    'pack_voltage',
    'pack_current',
])

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'simulation_result', 'heterogeneity_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CELLS = N_SERIES * N_PARALLEL
CELL_NAMES = [f'({s},{p})' for s in range(N_SERIES) for p in range(N_PARALLEL)]
DEFECT_IDX = DEFECT_CELL[0] * N_PARALLEL + DEFECT_CELL[1]
HEALTHY_IDX = [i for i in range(N_CELLS) if i != DEFECT_IDX]

# ============================================================================
# SIMULATION HELPER
# ============================================================================

def run_scenario(overrides, run_tag):
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    times_s  = data['times']
    curr     = data['Curr']
    temp     = data['Temp']
    sei      = data['SEI_Thick']
    fade     = data['CapFade']
    soc      = data['SOC']

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
    def _last(arr): return arr[-1]

    metrics = {}

    I_end_base = _last(base['curr'])
    I_end_scen = _last(scen['curr'])
    T_end_base = _last(base['temp'])
    T_end_scen = _last(scen['temp'])
    S_end_base = _last(base['sei'])
    S_end_scen = _last(scen['sei'])
    F_end_base = _last(base['fade'])
    F_end_scen = _last(scen['fade'])

    metrics['current_imbalance_base_A'] = float(np.std(I_end_base))
    metrics['current_imbalance_scen_A'] = float(np.std(I_end_scen))

    metrics['temp_spread_base_K']  = float(np.max(T_end_base) - np.min(T_end_base))
    metrics['temp_spread_scen_K']  = float(np.max(T_end_scen) - np.min(T_end_scen))

    metrics['sei_spread_base_nm']  = float(np.max(S_end_base) - np.min(S_end_base))
    metrics['sei_spread_scen_nm']  = float(np.max(S_end_scen) - np.min(S_end_scen))

    metrics['fade_spread_base_pct'] = float(np.max(F_end_base) - np.min(F_end_base))
    metrics['fade_spread_scen_pct'] = float(np.max(F_end_scen) - np.min(F_end_scen))

    T_defect_scen    = scen['temp'][:, DEFECT_IDX]
    T_healthy_scen   = scen['temp'][:, HEALTHY_IDX]
    metrics['defect_excess_temp_K'] = float(
        np.max(T_defect_scen - T_healthy_scen.mean(axis=1))
    )

    sei_defect_end   = scen['sei'][-1, DEFECT_IDX]
    sei_healthy_end  = scen['sei'][-1, HEALTHY_IDX].mean()
    metrics['defect_excess_sei_nm']  = float(sei_defect_end - sei_healthy_end)

    fade_defect_end  = scen['fade'][-1, DEFECT_IDX]
    fade_healthy_end = scen['fade'][-1, HEALTHY_IDX].mean()
    metrics['defect_excess_fade_pct'] = float(fade_defect_end - fade_healthy_end)

    I_defect_base = base['curr'][:, DEFECT_IDX].mean()
    I_defect_scen_val = scen['curr'][:, DEFECT_IDX].mean()
    metrics['current_shift_A'] = float(I_defect_scen_val - I_defect_base)

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

    print(f"\n[1/{len(SCENARIOS)+1}] Running BASELINE (all cells identical) ...")
    base = run_scenario({}, 'baseline')
    print(f"  Done in {base['elapsed']:.0f}s")
    print(f"  End-of-run current distribution: {base['curr'][-1]}")
    print(f"  End-of-run temperatures [C]:    {base['temp'][-1] - 273.15}")
    print(f"  End-of-run SEI thickness [nm]:  {base['sei'][-1]}")

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

    csv_path = os.path.join(RESULTS_DIR, 'heterogeneity_summary.csv')
    metric_keys = list(next(iter(all_metrics.values())).keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'description'] + metric_keys)
        for sname, metrics in all_metrics.items():
            desc = SCENARIOS[sname][2]
            writer.writerow([sname, desc] + [round(metrics[k], 6) for k in metric_keys])
    print(f"\n  Saved: {csv_path}")

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

        ax1 = fig.add_subplot(gs[1, col])
        for cell_idx in range(N_CELLS):
            lw = 2.5 if cell_idx == DEFECT_IDX else 1.0
            ax1.plot(th, scen['temp'][:, cell_idx] - 273.15,
                     color=cell_colors[cell_idx],
                     linestyle=cell_styles[cell_idx],
                     linewidth=lw)
        for cell_idx in range(N_CELLS):
            ax1.plot(th_b, base['temp'][:, cell_idx] - 273.15,
                     color='grey', linestyle=':', linewidth=0.7, alpha=0.5)
        ax1.set_ylabel('Temperature [C]' if col == 0 else '')
        ax1.grid(True, linestyle='--', alpha=0.3)

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

    fig2, axs2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle(
        f'Heterogeneity Propagation Summary -- {N_CYCLES} Cycles\n'
        f'All defects applied to cell {DEFECT_CELL} only',
        fontsize=12, fontweight='bold'
    )

    snames      = list(all_metrics.keys())
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


if __name__ == '__main__':
    main()
