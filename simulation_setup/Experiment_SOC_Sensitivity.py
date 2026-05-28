"""
Experiment_SOC_Sensitivity.py
==============================
SOC-varying instantaneous sensitivity of SEI growth rate using
PyTorch autograd during a plain 1C CC discharge.
"""

import os
import csv
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# Add parent folder to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import ImplicitBatterySolver

# ============================================================================
# CONFIGURATION
# ============================================================================

PACK_CONFIG = {
    'n_series':   1,
    'n_parallel': 1,
    'topology':   'series_first',
    'device':     'cpu',
    'stress_options': {'enabled': False},
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n':  5, 'Nx_s':  5, 'Nx_p': 5,
    'Nsei': 1,
}

I_DISCHARGE = 5.0     # 1C constant current [A]
DT          = 5.0     # timestep [s]
DT_MAX      = 30.0    # adaptive cap [s]
V_CUTOFF    = 2.5     # stop discharge at this voltage [V]

SOC_CHECKPOINTS = [0.90, 0.70, 0.50, 0.30, 0.10]

DIFF_PARAMS = {
    'Ds_n':      'Anode solid diffusivity',
    'm_ref_n':   'Anode rxn rate constant',
    'eps_e_n':   'Electrolyte porosity (anode)',
    'R_contact': 'Contact resistance',
    'hA':        'Cooling coefficient',
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'simulation_result', 'soc_temperature_sensitivity')
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAM_COLORS = {
    'Ds_n':      '#457b9d',
    'm_ref_n':   '#2a9d8f',
    'eps_e_n':   '#f4a261',
    'R_contact': '#e63946',
    'hA':        '#8338ec',
}

# ============================================================================
# HELPERS
# ============================================================================

def compute_soc(y, physics):
    cs_n  = physics.state(y, 'cs_n')
    cs_max = physics.params['cs_max_n']
    return float((cs_n.mean(dim=1, keepdim=True) / cs_max).mean().item())


def gradient_snapshot(battery, I_pack, param_keys):
    physics  = battery.physics
    y_snap   = battery.y.detach().clone()

    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(y_snap, I_pack)

    for k in param_keys:
        t = physics.params[k]
        t.requires_grad_(True)
        if t.grad is not None:
            t.grad.zero_()

    dY = physics.batched_derivatives(y_snap, I_cells)

    T_sl  = physics.state_layout.slice('temperature')
    dT_dt = dY[:, T_sl].mean()
    dT_val = dT_dt.item()

    dT_dt.backward()

    sensitivities = {}
    for k in param_keys:
        g = physics.params[k].grad
        if g is None:
            sensitivities[k] = 0.0
        else:
            g_val = float(g.mean().item())
            p_val = float(physics.params[k].data.mean().item())
            if abs(dT_val) > 1e-30:
                sensitivities[k] = g_val * p_val / abs(dT_val)
            else:
                sensitivities[k] = 0.0

    for k in param_keys:
        physics.params[k].requires_grad_(False)
        physics.params[k].grad = None

    return sensitivities, dT_val


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 66)
    print("  VIBE -- SOC-dependent temperature-rise sensitivity  (1C CC discharge)")
    print("  Output  : dT/dt  [K/s]  (instantaneous temperature rise rate)")
    print("  Params  : " + ", ".join(DIFF_PARAMS.keys()))
    print("  Checkpts: " + "  ".join(f"{int(s*100)}%" for s in SOC_CHECKPOINTS))
    print("=" * 66)

    print("\n[1/3] Initialising fully-charged cell ...")
    battery = ImplicitBatterySolver(
        PACK_CONFIG, DISCRETIZATION, {},
        initial_state_mode='fully_charged',
    )
    physics    = battery.physics
    param_keys = list(DIFF_PARAMS.keys())

    soc_init = compute_soc(battery.y, physics)
    sei_init = physics.state(battery.y, 'Lsei').mean().item() * 1e9
    print(f"    SOC = {soc_init:.3f},  Lsei = {sei_init:.3f} nm")

    print(f"\n[2/3] 1C CC discharge at {I_DISCHARGE} A  (no controller) ...")

    checkpoints = list(SOC_CHECKPOINTS)
    results     = {}
    prev_soc    = soc_init
    dt          = DT
    t           = 0.0
    step        = 0
    t0          = time.perf_counter()

    while True:
        with torch.no_grad():
            I_cells = battery.compute_effective_cell_currents(battery.y, I_DISCHARGE)
            y_new, ok = battery.basic_solver.newton_step(battery.y, dt, I_cells)

        if not ok:
            dt = max(dt * 0.5, 0.2)
            continue

        battery.y = y_new
        t        += dt
        step     += 1
        dt        = min(dt * 1.05, DT_MAX)

        soc = compute_soc(battery.y, physics)

        while (checkpoints
               and prev_soc > checkpoints[0]
               and soc     <= checkpoints[0]):

            target = checkpoints.pop(0)
            print(f"\n  [SOC={soc:.3f}, t={t:.0f}s]  -> checkpoint {target:.0%}")

            S, dL = gradient_snapshot(battery, I_DISCHARGE, param_keys)
            results[target] = {'S': S, 'dLsei': dL, 'soc': soc, 't': t}

            print(f"    dT/dt = {dL:.4e} K/s")
            for k, sv in S.items():
                print(f"    S({k:12s}) = {sv:+.4f}   [{DIFF_PARAMS[k]}]")

        prev_soc = soc

        with torch.no_grad():
            I_ck = battery.compute_effective_cell_currents(battery.y, I_DISCHARGE)
            _, _, v_min, _ = battery.check_voltage_limits(battery.y, I_ck)
        if v_min is not None and v_min <= V_CUTOFF:
            print(f"\n  Cutoff voltage reached ({v_min:.4f} V) at t={t:.0f}s")
            break

        if not checkpoints:
            print("\n  All checkpoints captured.")
            break

    wall = time.perf_counter() - t0
    print(f"\n  Discharge: {step} steps, {t/3600:.2f} h simulated, {wall:.1f} s wall-clock")

    if not results:
        print("ERROR: No checkpoints captured. Initial SOC may already be below targets.")
        return

    print("\n[3/3] Saving results and generating plots ...")

    soc_targets = sorted(results.keys(), reverse=True)
    soc_labels  = [f"SOC {int(s*100)}%" for s in soc_targets]
    dLsei_vals  = [results[s]['dLsei']   for s in soc_targets]
    S_matrix    = np.array([[results[st]['S'][pk] for st in soc_targets]
                             for pk in param_keys])

    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_vs_soc.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'description'] + soc_labels)
        for i, pk in enumerate(param_keys):
            w.writerow([pk, DIFF_PARAMS[pk]] +
                       [round(S_matrix[i, j], 6) for j in range(len(soc_targets))])
        w.writerow(['dT_dt [K/s]', ''] +
                   [f'{v:.4e}' for v in dLsei_vals])
    print(f"  Saved: {csv_path}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'sensitivity_vs_soc.npz'),
        param_keys  = np.array(param_keys),
        soc_targets = np.array(soc_targets),
        S_matrix    = S_matrix,
        dLsei_vals  = np.array(dLsei_vals),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    soc_pct = [s * 100 for s in soc_targets]

    for i, pk in enumerate(param_keys):
        ax.plot(soc_pct, S_matrix[i],
                marker='o', markersize=9, linewidth=2.2,
                color=PARAM_COLORS[pk],
                label=f'{pk}  ({DIFF_PARAMS[pk]})')
        for j, (xp, yp) in enumerate(zip(soc_pct, S_matrix[i])):
            if abs(yp) > 1e-4:
                ax.annotate(f'{yp:.3f}',
                            xy=(xp, yp), xytext=(0, 9),
                            textcoords='offset points',
                            ha='center', fontsize=7.5,
                            color=PARAM_COLORS[pk])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.invert_xaxis()
    ax.set_xlabel('SOC [%]   (discharge  90 -> 10)', fontsize=12)
    ax.set_ylabel(r'Normalised sensitivity  $S_i = \frac{\partial(dT/dt)}{\partial p_i}'
                  r'\cdot \frac{p_i}{|dT/dt|}$', fontsize=11)
    ax.set_title('SOC-Dependent Temperature-Rise Sensitivity\n'
                 'DFN Electrochemical-Thermal Model  |  1C CC discharge  |  PyTorch autograd',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(soc_pct)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Temp_Sensitivity_vs_SOC.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Temp_Sensitivity_vs_SOC.png")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    vmax = max(1e-3, float(np.max(np.abs(S_matrix))))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax2.imshow(S_matrix, cmap='RdBu_r', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax2, label='Normalised sensitivity S_i')
    ax2.set_xticks(range(len(soc_labels)))
    ax2.set_xticklabels(soc_labels, fontsize=10)
    ax2.set_yticks(range(len(param_keys)))
    ax2.set_yticklabels([DIFF_PARAMS[k] for k in param_keys], fontsize=9)
    for i in range(len(param_keys)):
        for j in range(len(soc_targets)):
            v = S_matrix[i, j]
            c = 'white' if abs(v) > 0.4 * vmax else 'black'
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center',
                     fontsize=9, color=c, fontweight='bold')
    ax2.set_title('dT/dt Sensitivity Heatmap  (params x SOC checkpoints)',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Temp_Sensitivity_Heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Temp_Sensitivity_Heatmap.png")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.semilogy(soc_pct, [abs(v) for v in dLsei_vals],
                 marker='s', markersize=9, linewidth=2, color='firebrick')
    for j, (xp, yp) in enumerate(zip(soc_pct, dLsei_vals)):
        ax3.annotate(f'{yp:.2e}', xy=(xp, abs(yp)),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8)
    ax3.invert_xaxis()
    ax3.set_xlabel('SOC [%]', fontsize=12)
    ax3.set_ylabel('|dT/dt|  [K/s]', fontsize=11)
    ax3.set_title('Temperature Rise Rate During 1C Discharge', fontsize=12, fontweight='bold')
    ax3.set_xticks(soc_pct)
    ax3.grid(True, linestyle='--', alpha=0.4, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Temp_RiseRate_vs_SOC.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Temp_RiseRate_vs_SOC.png")

    print("\n" + "=" * 66)
    print("  SENSITIVITY SUMMARY  (dT/dt output, 1C discharge)")
    print("=" * 66)
    print(f"  {'Parameter':<26}" +
          "".join(f"  {lb:>10}" for lb in soc_labels))
    print("-" * 66)
    for i, pk in enumerate(param_keys):
        print(f"  {pk:<26}" +
              "".join(f"  {S_matrix[i,j]:>+10.4f}" for j in range(len(soc_targets))))
    print("-" * 66)
    print(f"  {'dT/dt [K/s]':<26}" +
          "".join(f"  {v:>10.4e}" for v in dLsei_vals))
    print("=" * 66)
    print(f"\n  Results: {os.path.abspath(RESULTS_DIR)}")


if __name__ == '__main__':
    main()
