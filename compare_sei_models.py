"""
compare_sei_models.py
=====================
Runs a 1C constant-current discharge (3600 s, 1 cell) for every available
SEI growth model and plots the SEI layer thickness over time.

Usage:
    python compare_sei_models.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from sei_models import VALID_SEI_MODELS
from main import ImplicitBatterySolver

# ─────────────────────────────────────────────────────────────────────────────
# Simulation settings (shared across all runs)
# ─────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    'n_series': 1,
    'n_parallel': 1,
    'electrolyte_spatial_method': 'finite_volume',
    'solid_spatial_method': 'chebyshev',
    'stress_options': {'enabled': False},   # disable stress for cleaner comparison
    'sei_options': {'enabled': True, 'sei_model': 'builtin'},
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n': 10, 'Nx_s': 10, 'Nx_p': 10,
    'Nsei': 1,
}

T_END  = 3600.0   # 1 hour, 1C discharge
DT_INIT = 1.0
DT_MAX  = 50.0
I_PACK  = 5.0     # 1C for a nominal 5 Ah cell

# Run builtin + every model in registry
ALL_MODELS = ['builtin'] + VALID_SEI_MODELS

# Auto-generate labels — fall back to model name if not in the map
_LABEL_MAP = {
    'builtin':                        'Built-in (stress-enhanced)',
    'constant':                       'Constant (0 growth)',
    'solvent-diffusion limited':      'Solvent-diffusion limited',
    'reaction limited':               'Reaction limited',
    'interstitial-diffusion limited': 'Interstitial-diffusion limited',
    'ec-reaction limited':            'EC-reaction limited',
    'electron-migration limited':     'Electron-migration limited',
    'tunneling':                      'Tunneling',
}
LABELS = {m: _LABEL_MAP.get(m, m) for m in ALL_MODELS}

# ─────────────────────────────────────────────────────────────────────────────
# Run simulations
# ─────────────────────────────────────────────────────────────────────────────
results    = {}   # model_name → (times [s], lsei [nm])
skipped    = {}   # model_name → reason string

for model_name in ALL_MODELS:
    print(f"\n{'─'*60}")
    print(f"  SEI model: {LABELS[model_name]}")
    print(f"{'─'*60}")

    cfg = dict(BASE_CONFIG)
    cfg['sei_options'] = {'enabled': True, 'sei_model': model_name}

    try:
        solver = ImplicitBatterySolver(
            cfg,
            DISCRETIZATION,
            overrides={},
            initial_state_mode='fully_charged',
        )

        # Monkey-patch BasicSolver to capture state history in memory
        _times_hist, _y_hist = [], []
        _basic = solver.basic_solver
        _orig  = _basic.process_results

        def _capture(times, y_list, I_pack, _orig=_orig):
            _times_hist.clear(); _times_hist.extend(times)
            _y_hist.clear();     _y_hist.extend(y_list)
            _orig(times, y_list, I_pack)

        _basic.process_results = _capture

        solver.simulate(
            t_end=T_END,
            dt_init=DT_INIT,
            I_pack=I_PACK,
            dt_max=DT_MAX,
        )

        # ── Extract Lsei from in-memory history ───────────────────────────
        if not _y_hist:
            skipped[model_name] = "No state history recorded (simulation may have crashed immediately)"
            print(f"  SKIPPED: {skipped[model_name]}")
            continue

        phys      = solver.physics
        times_arr = np.array(_times_hist)
        lsei_vals = []
        for y in _y_hist:
            y_batch = y[0:1] if y.dim() == 2 else y.unsqueeze(0)
            lsei_vals.append(phys.state(y_batch, 'Lsei')[0].item())

        lsei_nm = np.array(lsei_vals) * 1e9   # m → nm
        results[model_name] = (times_arr, lsei_nm)
        print(f"  Final SEI = {lsei_nm[-1]:.3f} nm  at t = {times_arr[-1]:.0f} s")

    except Exception as e:
        skipped[model_name] = str(e)
        print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  Summary")
print(f"{'═'*60}")
for m in ALL_MODELS:
    if m in results:
        t, L = results[m]
        print(f"  ✓ {LABELS[m]:40s}  {L[-1]:.3f} nm  ({t[-1]:.0f} s)")
    else:
        print(f"  ✗ {LABELS[m]:40s}  SKIPPED — {skipped.get(m, '?')}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
if not results:
    print("\nNo results to plot.")
    sys.exit(1)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   12,
    'axes.titlesize':  15,
    'axes.labelsize':  13,
    'legend.fontsize': 10,
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'text.color':  '#e0e0e0',
    'grid.color':  '#2a2d3a',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'lines.linewidth': 2.2,
})

n = len(results)
colors = cm.turbo(np.linspace(0.05, 0.95, n))

# ── Single-panel: all models together ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

for (model_name, (t_s, lsei_nm)), color in zip(results.items(), colors):
    ls = '--' if model_name == 'constant' else '-'
    ax.plot(t_s / 60.0, lsei_nm, label=LABELS[model_name],
            color=color, linestyle=ls)

ax.set_xlabel('Time [min]')
ax.set_ylabel('SEI Layer Thickness [nm]')
ax.set_title(f'SEI Growth Model Comparison — 1C Discharge, 3600 s, 1 Cell  ({n} models)')
ax.legend(loc='upper left', framealpha=0.25, ncol=2)
ax.grid(True)

# Annotate final values (stagger vertically to avoid overlap)
sorted_items = sorted(results.items(), key=lambda kv: kv[1][1][-1])
for idx, (model_name, (t_s, lsei_nm)) in enumerate(sorted_items):
    color = colors[list(results.keys()).index(model_name)]
    ax.annotate(
        f'{lsei_nm[-1]:.2f} nm',
        xy=(t_s[-1] / 60.0, lsei_nm[-1]),
        xytext=(6, idx * 4 - (len(sorted_items) * 2)),
        textcoords='offset points',
        color=color, fontsize=8.5, va='center',
        arrowprops=dict(arrowstyle='-', color=color, alpha=0.4, lw=0.8),
    )

plt.savefig('sei_model_comparison.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nPlot saved → sei_model_comparison.png")
plt.show()
