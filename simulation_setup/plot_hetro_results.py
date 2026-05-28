import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load data
    npz_path = os.path.join("..", "simulation_result", "hetro", "results", "simulation_results.npz")
    data = np.load(npz_path)
    
    times = data['times'] / 60.0  # Convert to minutes
    curr = data['Curr']
    temp = data['Temp'] - 273.15  # Convert to Celsius
    
    # We want to plot cell 0 and cell 1
    # Create the Current Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, curr[:, 0], label='Cell 0 (Cooled)', color='#1f77b4', linewidth=2)
    plt.plot(times, curr[:, 1], label='Cell 1 (Standard)', color='#d62728', linewidth=2, linestyle='--')
    
    plt.title('Current Distribution in Heterogeneous Pack', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Time [minutes]', fontsize=14)
    plt.ylabel('Current [A]', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'simulation_result', 'hetro', 'heterogeneity_current_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved heterogeneity_current_comparison.png")
    
    # Create the Temperature Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, temp[:, 0], label='Cell 0 (Cooled)', color='#1f77b4', linewidth=2)
    plt.plot(times, temp[:, 1], label='Cell 1 (Standard)', color='#d62728', linewidth=2, linestyle='--')
    
    plt.title('Temperature Profile in Heterogeneous Pack', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Time [minutes]', fontsize=14)
    plt.ylabel('Temperature [°C]', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'simulation_result', 'hetro', 'heterogeneity_temperature_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved heterogeneity_temperature_comparison.png")

if __name__ == "__main__":
    main()
