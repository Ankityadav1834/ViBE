import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_stress(csv_file, output_file=None, cell=None, local_index=None):
    df = pd.read_csv(csv_file)

    stress_state = df[(df["kind"] == "state") & (df["name"] == "stress")]
    stress_outputs = df[(df["kind"] == "output") & (df["name"].isin(["stress_mean", "stress_min", "stress_max", "force_from_stress"]))]

    if stress_state.empty and stress_outputs.empty:
        raise ValueError(
            "No stress data found in the CSV. Set stress_options['enabled'] = True in run.py, "
            "rerun the simulation, then run this plotter again."
        )

    fig, ax = plt.subplots(figsize=(11, 6))

    if not stress_state.empty:
        data = stress_state.copy()
        if cell is not None:
            data = data[data["cell"] == cell]
        if local_index is not None:
            data = data[data["local_index"] == local_index]

        if data.empty:
            raise ValueError("No stress state rows match the selected cell/local_index filters.")

        if local_index is None:
            grouped = data.groupby(["time", "cell"], as_index=False)["value"].mean()
            for cell_id, cell_data in grouped.groupby("cell"):
                ax.plot(cell_data["time"], cell_data["value"], label=f"Cell {cell_id} stress mean")
            ax.set_ylabel("Mean stress")
        else:
            for cell_id, cell_data in data.groupby("cell"):
                ax.plot(cell_data["time"], cell_data["value"], label=f"Cell {cell_id} stress[{local_index}]")
            ax.set_ylabel("Stress")
    else:
        for name, output_data in stress_outputs.groupby("name"):
            if cell is None:
                data = output_data.groupby("time", as_index=False)["value"].mean()
                label = f"{name} pack mean"
            else:
                data = output_data[output_data["cell"] == cell]
                label = f"{name} cell {cell}"
            if not data.empty:
                ax.plot(data["time"], data["value"], label=label)
        ax.set_ylabel("Stress-derived output")

    ax.set_xlabel("Time [s]")
    ax.set_title("Stress vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=200)
        print(f"Saved {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot stress values from all_results.csv.")
    parser.add_argument("--csv", default="all_results.csv", help="Input results CSV file.")
    parser.add_argument("--out", default=None, help="Optional output image path, e.g. stress_plot.png.")
    parser.add_argument("--cell", type=int, default=None, help="Optional cell index to plot.")
    parser.add_argument("--local-index", type=int, default=None, help="Optional stress node/local index to plot.")
    args = parser.parse_args()

    plot_stress(args.csv, output_file=args.out, cell=args.cell, local_index=args.local_index)


if __name__ == "__main__":
    main()
