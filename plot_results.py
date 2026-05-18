import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Default parameters to convert SEI thickness to Capacity Fade (Chen2020 defaults)
rho_sei = 1690.0
Msei = 0.162
Rs_n = 5.86e-06
cs_max = 33133.0
Lsei_0 = 5e-9

def get_capacity_fade_pct(sei_thickness_nm):
    Lsei_m = sei_thickness_nm * 1e-9
    delta_Lsei = Lsei_m - Lsei_0
    fade_frac = 2.0 * (rho_sei / Msei) * delta_Lsei * (3.0 / Rs_n) / cs_max
    return fade_frac * 100.0

def main():
    parser = argparse.ArgumentParser(description="Plot simulation results from .npz file")
    parser.add_argument("--cells", type=str, default="all", help="Comma-separated list of cells to plot (e.g. '0,2,5') or 'all'")
    parser.add_argument("--vars", type=str, default="voltage,current,temperature,sei,fade", 
                        help="Comma-separated variables to plot: voltage, current, temperature, sei, fade")
    
    # FIXED: Replaced backslashes (\) with forward slashes (/) to avoid \t and \r escape character issues.
    default_path = "simulation_result/cooling_test_exp/results/simulation_results.npz"
    parser.add_argument("--file", type=str, default=default_path, help="Path to npz file")
    
    args = parser.parse_args()

    print(f"Loading {args.file}...")
    try:
        data = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: Could not find '{args.file}'. Make sure the simulation has finished saving and the path is correct relative to where you are running the script.")
        return

    times_hours = data['times'] / 3600.0
    n_cells = data['TermV'].shape[1]
    
    if args.cells.lower() == "all":
        cells_to_plot = list(range(n_cells))
    else:
        cells_to_plot = [int(c.strip()) for c in args.cells.split(',')]

    vars_to_plot = [v.strip().lower() for v in args.vars.split(',')]
    n_plots = len(vars_to_plot)
    
    if n_plots == 0:
        print("No variables selected to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
        
    for ax, var in zip(axes, vars_to_plot):
        if var == "voltage":
            for c in cells_to_plot:
                ax.plot(times_hours, data['TermV'][:, c], label=f'Cell {c}', alpha=0.7)
            ax.set_ylabel("Voltage [V]")
            ax.set_title("Terminal Voltage")
            
        elif var == "current":
            for c in cells_to_plot:
                ax.plot(times_hours, data['Curr'][:, c], label=f'Cell {c}', alpha=0.7)
            ax.set_ylabel("Current [A]")
            ax.set_title("Cell Current")
            
        elif var == "temperature":
            for c in cells_to_plot:
                ax.plot(times_hours, data['Temp'][:, c] - 273.15, label=f'Cell {c}', alpha=0.7)
            ax.set_ylabel("Temperature [C]")
            ax.set_title("Cell Temperature")
            
        elif var == "sei":
            for c in cells_to_plot:
                ax.plot(times_hours, data['SEI_Thick'][:, c], label=f'Cell {c}', alpha=0.7)
            ax.set_ylabel("SEI Thickness [nm]")
            ax.set_title("SEI Growth")
            
        elif var == "fade":
            for c in cells_to_plot:
                fade = get_capacity_fade_pct(data['SEI_Thick'][:, c])
                ax.plot(times_hours, fade, label=f'Cell {c}', alpha=0.7)
            ax.set_ylabel("Capacity Fade [%]")
            ax.set_title("Capacity Fade from SEI")
            
        else:
            print(f"Warning: Unknown variable '{var}'")
            
        # Only show legend if plotting fewer than 15 cells to avoid clutter
        if len(cells_to_plot) <= 15:
            ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time [Hours]")
    plt.tight_layout()
    plt.savefig('custom_simulation_plots.png', dpi=300)
    print("Saved custom_simulation_plots.png!")
    plt.show()

if __name__ == "__main__":
    main()