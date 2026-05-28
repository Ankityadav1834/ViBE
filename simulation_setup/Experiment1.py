import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import sys

# Add the parent directory to path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ---------------------------------------------------------------------------
# Disable plt.show() so the solver's internal plots don't block the loop.
# We will restore it at the end.
# ---------------------------------------------------------------------------
plt_show_orig = plt.show
plt.show = lambda: None

# ── Experiment Configuration ────────────────────────────────────────────────

# C-rates to compare
C_RATES = [0.5, 1.0, 2.0, 3.0]

# Number of charge/discharge cycles.
# Note: 200 cycles of a full PDE model can take many hours.
# 20 cycles is a reasonable starting point for a test run.
N_CYCLES = 20

# Use a single cell to isolate C-rate effects (no pack heterogeneity)
config = {
    'n_series': 1,
    'n_parallel': 1,
    'stress_options': {'enabled': False},  # skip mechanical stress for speed
}

discretization = {
    'Nr_n': 10, 'Nr_p': 10, 'Nx_n': 5, 'Nx_s': 5, 'Nx_p': 5, 'Nsei': 1
}

# ── OutputSpec: save only what this experiment needs ────────────────────────
experiment_spec = OutputSpec([
    "terminal_voltage",         # TermV  — per-cell [V]
    "cell_current",             # Curr   — per-cell [A]
    "soc",                      # SOC    — per-cell [-]
    "temperature",              # Temp   — per-cell [K]
    "sei_thickness",            # SEI_Thick — per-cell [nm]
    "capacity_fade",            # CapFade   — per-cell [%]
    "pack_voltage",             # PackVoltage [V]
    "pack_current",             # PackCurrent [A]
    # Overpotentials — useful for thesis comparison with liionpack
    "rxn_overpotential",        # Rxn [V]
    "ohmic_solid",              # OhmS [V]
    "ohmic_electrolyte",        # OhmE [V]
    "concentration_overpotential",  # Conc [V]
    "sei_voltage",              # SEI  [V]
    # External circuit losses
    "contact_resistance_drop",  # OhmRC [V]
    "busbar_resistance_drop",   # OhmRB [V]
])

# ── Experiment Loop ─────────────────────────────────────────────────────────

summary_data = []

print("==================================================")
print(f" Starting Capacity Fade Experiment ({N_CYCLES} cycles)")
print("==================================================")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join("..", "simulation_result", "Experiment1"), exist_ok=True)

for c_rate in C_RATES:
    print(f"\n---> Running Simulation for {c_rate}C Discharge <---")

    discharge_current = c_rate * 5.0
    charge_current    = -5.0  # fixed 1C charge to isolate discharge stress

    controller_config = {
        'cc_current':       charge_current,
        'cv_voltage':       4.2,
        'cutoff_current':   1.0,
        'discharge_current': discharge_current,
        'min_voltage':      2.5,
        'max_voltage':      4.2,
        'n_cycles':         N_CYCLES,
    }

    battery_solver = ImplicitBatterySolver(
        config,
        discretization,
        overrides={},
        initial_state_mode='fully_charged',
    )

    controller = build_controller('cycle_cccv', **controller_config)

    battery_solver.simulate(
        t_end=36_000_000,
        dt_init=1.0,
        controller=controller,
        dt_max=50.0,
        output_spec=experiment_spec,
    )

    # ── Load results from .npz ───────────────────────────────────────────────
    print(f"  Loading results for {c_rate}C …")

    # The .npz is saved to simulation_result/<script_name>/results/
    run_name = "Experiment1"
    npz_path = os.path.join("..", "simulation_result", run_name, "results",
                            "simulation_results.npz")
    data = np.load(npz_path)

    times_s   = data["times"]                  # [n_steps]
    fade_pct  = data["CapFade"][:, 0]          # cell 0, [%]
    sei_nm    = data["SEI_Thick"][:, 0]        # cell 0, [nm]

    # Save a clean per-C-rate npz for later analysis inside simulation_result/Experiment1/
    out_fname = os.path.join("..", "simulation_result", "Experiment1", f"results_{c_rate}C.npz")
    np.savez_compressed(
        out_fname,
        time_hrs=times_s / 3600.0,
        capacity_fade_pct=fade_pct,
        sei_thickness_nm=sei_nm,
    )
    print(f"  Saved {out_fname}")

    summary_data.append({
        'c_rate': c_rate,
        'time':   times_s / 3600.0,
        'fade':   fade_pct,
        'sei':    sei_nm,
    })

    del battery_solver, controller, data
    gc.collect()

# ── Final Comparison Plots ───────────────────────────────────────────────────

print("\nGenerating comparison plots…")
plt.show = plt_show_orig

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"C-Rate Capacity Fade Study — {N_CYCLES} Cycles", fontsize=13)

for d in summary_data:
    axs[0].plot(d['time'], d['fade'], label=f"{d['c_rate']}C")
    axs[1].plot(d['time'], d['sei'],  label=f"{d['c_rate']}C")

axs[0].set_title('Capacity Fade vs Time')
axs[0].set_ylabel('Capacity Fade [%]')
axs[0].set_xlabel('Time [hours]')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].set_title('SEI Thickness vs Time')
axs[1].set_ylabel('SEI Thickness [nm]')
axs[1].set_xlabel('Time [hours]')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plot_path = os.path.join("..", "simulation_result", "Experiment1", "Experiment1_Plots.png")
plt.savefig(plot_path, dpi=300)
print(f"Experiment completed! Plots saved to '{plot_path}'.")
plt.show()
