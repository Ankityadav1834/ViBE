import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_variable(csv_file, variable, cells=None, local_index=None):
    df = pd.read_csv(csv_file)
    data = df[df['name'] == variable]
    if cells is not None:
        data = data[data['cell'].isin(cells)]
    if local_index is not None:
        data = data[data['local_index'] == local_index]
    if data.empty:
        print(f"No data found for variable '{variable}' with the given filters.")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for cell_id, cell_data in data.groupby('cell'):
        label = f"Cell {cell_id}"
        if local_index is not None:
            label += f" local_index {local_index}"
        ax.plot(cell_data['time'], cell_data['value'], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(variable)
    ax.set_title(f"{variable} vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="General variable plotter for all_results.csv.")
    parser.add_argument('--csv', default='all_results.csv', help='Input results CSV file.')
    parser.add_argument('--variable', required=True, help='Variable name to plot (from the "name" column).')
    parser.add_argument('--cells', type=int, nargs='*', default=None, help='Optional list of cell indices to plot.')
    parser.add_argument('--local-index', type=int, default=None, help='Optional local index to plot.')
    args = parser.parse_args()
    plot_variable(args.csv, args.variable, cells=args.cells, local_index=args.local_index)

if __name__ == "__main__":
    main()
