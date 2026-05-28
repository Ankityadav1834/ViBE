import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Path to results
    results_path = os.path.join(
        '..', 'simulation_result', 'cooling_test_exp', 'results', 'simulation_results.npz'
    )
    
    if not os.path.exists(results_path):
        print(f"Error: Could not find results at {results_path}")
        return
        
    data = np.load(results_path)
    
    # Calculate SEI growth (proportional to capacity loss)
    sei_initial = data['SEI_Thick'][0]
    sei_final = data['SEI_Thick'][-1]
    sei_growth_nm = sei_final - sei_initial
    
    # Convert SEI growth (nm) to Capacity Loss (%)
    # Conversion factor derived from cell geometry (Ah_per_m / Nominal_Ah * 100)
    # Ah_per_m = (A * Ln * a_n) * F / (Vbar_sei * 3600) = ~937077.4 Ah/m
    # Factor = 1e-9 * 937077.4 / 5.0 * 100 = 0.0187415
    cap_loss_pct = sei_growth_nm * 0.0187415
    
    # Reshape to 10x10 grid
    n_series = 10
    n_parallel = 10
    grid = cap_loss_pct.reshape((n_series, n_parallel))
    
    # The over-cooled cells from cooling_test_exp.py
    overcooled_cells = [(0,1), (1,0), (2,0), (3,0), (4,0)]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    # We use a colormap where lower growth (cooler) is blue, and higher growth (hotter) is red
    im = plt.imshow(grid, cmap='coolwarm', origin='upper', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Capacity Loss [%]', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(n_series):
        for j in range(n_parallel):
            val = grid[i, j]
            text_color = 'white' if (val < np.mean(grid) - 0.002 or val > np.mean(grid) + 0.002) else 'black'
            
            # Highlight overcooled cells
            if (i, j) in overcooled_cells:
                # Add a marker and bold text for cooled cells
                plt.text(j, i, f"{val:.3f}%\n(Cooled)", ha="center", va="center", 
                         color='cyan', fontweight='bold', fontsize=9)
            else:
                plt.text(j, i, f"{val:.3f}%", ha="center", va="center", 
                         color=text_color, fontsize=9)
    
    # Aesthetics
    plt.title('Final Capacity Loss Heatmap (10x10 Pack)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Parallel Column Index', fontsize=14)
    plt.ylabel('Series Row Index', fontsize=14)
    plt.xticks(np.arange(n_parallel))
    plt.yticks(np.arange(n_series))
    
    # Highlight the specific overcooled cells with a border
    ax = plt.gca()
    for (i, j) in overcooled_cells:
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='cyan', linewidth=3)
        ax.add_patch(rect)
        
    plt.tight_layout()
    plot_path = os.path.join('..', 'simulation_result', 'cooling_test_exp', 'capacity_loss_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {plot_path}")

if __name__ == '__main__':
    main()
