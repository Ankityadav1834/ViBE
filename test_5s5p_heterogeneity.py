import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers import build_controller
from main import ImplicitBatterySolver


N_SERIES = 5
N_PARALLEL = 5
NOMINAL_CELL_CAPACITY_AH = 5.0

# Chen2020/PyBaMM: area = 0.00531 m2 and nominal h = 10 W m-2 K-1.
# This repo's default hA is therefore 0.0531 W/K.
WEAK_COOLING_HA = 0.0531
GOOD_COOLING_HA = 0.531

# Measured from this solver at the fully charged initial state with the default
# parameter set. Gogoana et al. used a 20% mismatch as a significant case.
NOMINAL_INITIAL_R_OHM = 0.005710941711384999
R_CONTACT_20PCT = 0.20 * NOMINAL_INITIAL_R_OHM
R_CONTACT_30PCT = 0.30 * NOMINAL_INITIAL_R_OHM

HETEROGENEOUS_BRANCH = 2
WEAK_LINK_CELL = (2, HETEROGENEOUS_BRANCH)


def build_heterogeneity_overrides():
    """Apply a literature-backed resistance and cooling mismatch to one 5-cell branch."""
    overrides = {}

    for s_idx in range(N_SERIES):
        for p_idx in range(N_PARALLEL):
            overrides[(s_idx, p_idx)] = {
                "hA": GOOD_COOLING_HA,
                "R_contact": 0.0,
            }

    for s_idx in range(N_SERIES):
        overrides[(s_idx, HETEROGENEOUS_BRANCH)].update(
            {
                "hA": WEAK_COOLING_HA,
                "R_contact": R_CONTACT_20PCT,
            }
        )

    overrides[WEAK_LINK_CELL]["R_contact"] = R_CONTACT_30PCT
    return overrides


def build_solver(device):
    config = {
        "device": device,
        "n_series": N_SERIES,
        "n_parallel": N_PARALLEL,
        "electrolyte_spatial_method": "finite_volume",
        "solid_spatial_method": "chebyshev",
        "stress_options": {"enabled": False},
        "sei_options": {"enabled": True, "sei_model": "reaction limited"},
    }

    discretization = {
        "Nr_n": 10,
        "Nr_p": 10,
        "Nx_n": 10,
        "Nx_s": 10,
        "Nx_p": 10,
        "Nsei": 1,
    }

    thermal_options = {
        "enabled": True,
        "strategy": "ambient",
        "model": "builtin",
        "ambient_temp": 298.15,
    }

    return ImplicitBatterySolver(
        config,
        discretization,
        overrides=build_heterogeneity_overrides(),
        initial_state_mode="fully_charged",
        thermal_options=thermal_options,
    )


def run_simulation(args):
    pack_current = args.c_rate * NOMINAL_CELL_CAPACITY_AH * N_PARALLEL
    solver = build_solver(args.device)

    print("=" * 72)
    print("5S5P heterogeneous pack simulation")
    print("=" * 72)
    print(f"Device: {solver.device}")
    print(f"C-rate: {args.c_rate:.2f}C, pack current: {pack_current:.2f} A")
    print(
        "Heterogeneous branch: "
        f"p={HETEROGENEOUS_BRANCH}, hA={WEAK_COOLING_HA:.4f} W/K, "
        f"R_contact={R_CONTACT_20PCT:.6f} Ohm"
    )
    print(
        "Weak-link cell: "
        f"s={WEAK_LINK_CELL[0]}, p={WEAK_LINK_CELL[1]}, "
        f"R_contact={R_CONTACT_30PCT:.6f} Ohm"
    )

    if args.smoke:
        controller = build_controller(
            "constant_current",
            current=pack_current,
            min_voltage=2.5,
            max_voltage=4.25,
        )
        solver.simulate(
            t_end=args.smoke_seconds,
            dt_init=1.0,
            controller=controller,
            dt_max=min(args.dt_max, 30.0),
        )
    else:
        controller = build_controller(
            "cycle_cccv",
            cc_current=-pack_current,
            cv_voltage=4.2,
            cutoff_current=0.05 * pack_current,
            discharge_current=pack_current,
            min_voltage=2.5,
            max_voltage=4.25,
            n_cycles=args.cycles,
        )
        solver.simulate(
            t_end=args.t_end_days * 24.0 * 3600.0,
            dt_init=1.0,
            controller=controller,
            dt_max=args.dt_max,
        )


def result_dir():
    run_name = os.path.splitext(os.path.basename(__file__))[0]
    return os.path.join(os.getcwd(), "simulation_result", run_name, "results")


def summarize_results():
    out_dir = result_dir()
    result_path = os.path.join(out_dir, "simulation_results.npz")
    if not os.path.exists(result_path):
        print(f"No result file found at {result_path}")
        return

    data = np.load(result_path)
    times = data["times"]
    currents = data["Curr"].reshape(len(times), N_SERIES, N_PARALLEL)
    temps_c = data["Temp"].reshape(len(times), N_SERIES, N_PARALLEL) - 273.15
    lsei_nm = data["SEI_Thick"].reshape(len(times), N_SERIES, N_PARALLEL)
    term_v = data["TermV"].reshape(len(times), N_SERIES, N_PARALLEL)

    final_branch = pd.DataFrame(
        {
            "parallel_branch": np.arange(N_PARALLEL),
            "mean_final_current_A": currents[-1].mean(axis=0),
            "mean_abs_current_A": np.abs(currents).mean(axis=(0, 1)),
            "peak_abs_current_A": np.abs(currents).max(axis=(0, 1)),
            "final_mean_temperature_C": temps_c[-1].mean(axis=0),
            "peak_temperature_C": temps_c.max(axis=(0, 1)),
            "final_mean_sei_nm": lsei_nm[-1].mean(axis=0),
            "max_sei_nm": lsei_nm.max(axis=(0, 1)),
            "final_min_cell_voltage_V": term_v[-1].min(axis=0),
        }
    )

    summary_path = os.path.join(out_dir, "heterogeneity_summary.csv")
    final_branch.to_csv(summary_path, index=False)

    current_spread = (
        np.max(np.abs(currents), axis=2) - np.min(np.abs(currents), axis=2)
    )
    temp_spread = np.max(temps_c, axis=(1, 2)) - np.min(temps_c, axis=(1, 2))
    sei_spread = np.max(lsei_nm, axis=(1, 2)) - np.min(lsei_nm, axis=(1, 2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t_hr = times / 3600.0

    for p_idx in range(N_PARALLEL):
        axes[0, 0].plot(t_hr, np.abs(currents[:, :, p_idx]).mean(axis=1), label=f"p{p_idx}")
        axes[0, 1].plot(t_hr, temps_c[:, :, p_idx].mean(axis=1), label=f"p{p_idx}")
        axes[1, 0].plot(t_hr, lsei_nm[:, :, p_idx].mean(axis=1), label=f"p{p_idx}")

    axes[1, 1].plot(t_hr, current_spread.mean(axis=1), label="current spread [A]")
    axes[1, 1].plot(t_hr, temp_spread, label="temperature spread [C]")
    axes[1, 1].plot(t_hr, sei_spread, label="SEI spread [nm]")

    axes[0, 0].set_title("Mean absolute branch current")
    axes[0, 0].set_ylabel("A")
    axes[0, 1].set_title("Mean branch temperature")
    axes[0, 1].set_ylabel("deg C")
    axes[1, 0].set_title("Mean branch SEI thickness")
    axes[1, 0].set_ylabel("nm")
    axes[1, 1].set_title("Pack heterogeneity metrics")

    for ax in axes.ravel():
        ax.set_xlabel("time [h]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_path = os.path.join(out_dir, "heterogeneity_diagnostics.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("\nHeterogeneity summary:")
    print(final_branch.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nSaved {summary_path}")
    print(f"Saved {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a literature-backed 5S5P heterogeneity simulation."
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', etc.")
    parser.add_argument("--c-rate", type=float, default=2.0, help="Pack C-rate based on 5 Ah cells.")
    parser.add_argument("--cycles", type=int, default=100, help="Number of CCCV cycles.")
    parser.add_argument("--t-end-days", type=float, default=2.0, help="Maximum simulated duration.")
    parser.add_argument("--dt-max", type=float, default=80.0, help="Maximum solver time step in seconds.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short 5S5P constant-current discharge smoke test.",
    )
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        default=600.0,
        help="Duration for --smoke mode.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only summarize existing results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_run:
        run_simulation(args)
    summarize_results()
