import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add the parent directory to path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ImplicitBatterySolver

# Configuration
PACK_CONFIG = {
    'n_series': 1,
    'n_parallel': 1,
    'topology': 'series_first',
    'device': 'cuda',
    'stress_options': {'enabled': False},
}

DISCRETIZATION = {
    'Nr_n': 100, 'Nr_p': 100,
    'Nx_n': 50, 'Nx_s': 50, 'Nx_p': 50,
    'Nsei': 1,
}

RATES = {'1C': 5.0, '2C': 10.0, '3C': 15.0}
COLORS = {'1C': '#1f77b4', '2C': '#ff7f0e', '3C': '#2ca02c'}
CUTOFF_VOLTAGE = 2.5
DT = 1.0
DT_MAX = 10.0

def run_discharge(c_rate_name, i_discharge):
    print(f"Running {c_rate_name} discharge at {i_discharge} A...")
    battery = ImplicitBatterySolver(
        PACK_CONFIG, DISCRETIZATION, {},
        initial_state_mode='fully_charged',
    )
    
    t = 0.0
    dt = DT
    times = []
    voltages = []
    
    times.append(t)
    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(battery.y, i_discharge)
        v_term = battery.get_exact_terminal_voltages(battery.y, I_cells).mean().item()
    voltages.append(v_term)
    
    step_count = 0
    fail_count = 0
    while True:
        with torch.no_grad():
            I_cells = battery.compute_effective_cell_currents(battery.y, i_discharge)
            y_new, ok = battery.basic_solver.newton_step(battery.y, dt, I_cells)
        
        if not ok:
            dt = max(dt * 0.5, 1e-3)
            fail_count += 1
            if fail_count > 100:
                print("Failed to converge 100 times. Breaking.")
                break
            continue
            
        fail_count = 0
        battery.y = y_new
        t += dt
        dt = min(dt * 1.05, DT_MAX)
        step_count += 1
        
        with torch.no_grad():
            I_cells = battery.compute_effective_cell_currents(battery.y, i_discharge)
            v_term = battery.get_exact_terminal_voltages(battery.y, I_cells).mean().item()
        
        times.append(t)
        voltages.append(v_term)
        
        if step_count % 100 == 0:
            print(f"  Time: {t:.1f}s, dt: {dt:.2f}s, Voltage: {v_term:.4f}V")
        
        if v_term <= CUTOFF_VOLTAGE:
            if len(voltages) >= 2 and voltages[-2] > CUTOFF_VOLTAGE:
                v_prev = voltages[-2]
                t_prev = times[-2]
                fraction = (v_prev - CUTOFF_VOLTAGE) / (v_prev - v_term + 1e-12)
                t_exact = t_prev + fraction * (t - t_prev)
                times[-1] = t_exact
                voltages[-1] = CUTOFF_VOLTAGE
            print(f"  Reached cutoff voltage {voltages[-1]:.4f}V at {times[-1]:.1f}s")
            break
            
    return np.array(times), np.array(voltages)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    plt.figure(figsize=(10, 6))
    
    for rate, i_mag in RATES.items():
        # Run our model
        t_model, v_model = run_discharge(rate, i_mag)
        
        # Load PyBaMM data from data subfolder
        pybamm_file = os.path.join('data', f'terminal_voltage_{rate.lower()}.csv')
        try:
            pybamm_data = np.loadtxt(pybamm_file, delimiter=',', skiprows=1)
            t_pybamm = pybamm_data[:, 0]
            v_pybamm = pybamm_data[:, 1]
            
            # Use cubic spline to create a smooth curve for PyBaMM plotting
            from scipy.interpolate import CubicSpline
            
            # Remove any duplicate time points in PyBaMM data just in case
            unique_idx = np.unique(t_pybamm, return_index=True)[1]
            t_pybamm = t_pybamm[unique_idx]
            v_pybamm = v_pybamm[unique_idx]
            
            cs = CubicSpline(t_pybamm, v_pybamm)
            t_smooth = np.linspace(t_pybamm[0], t_pybamm[-1], 1000)
            v_smooth = cs(t_smooth)
            
            # Interpolate PyBaMM smoothly onto model time to calculate error
            max_t = min(np.max(t_model), np.max(t_pybamm))
            valid_idx = t_model <= max_t
            v_pybamm_interp = cs(t_model[valid_idx])
            rmse = np.sqrt(np.mean((v_model[valid_idx] - v_pybamm_interp)**2))
            
            # Save VIBE results to CSV inside simulation_result/Cell_Verification_results/
            output_csv = os.path.join('..', 'simulation_result', 'Cell_Verification_results', f'vibe_terminal_voltage_{rate.lower()}.csv')
            np.savetxt(output_csv, np.column_stack((t_model, v_model)), delimiter=',', header='Time [s],Terminal voltage [V]', comments='')
            
            # Plot PyBaMM (smooth)
            plt.plot(t_smooth, v_smooth, color=COLORS[rate], linestyle='-', 
                     label=f'PyBaMM {rate}')
            
            # Plot Model
            plt.plot(t_model, v_model, color=COLORS[rate], linestyle=':', linewidth=2,
                     label=f'VIBE {rate} (RMSE: {rmse:.4f}V)')
            
            print(f"{rate} RMSE: {rmse:.4f} V")
        except Exception as e:
            print(f"Failed to load or process {pybamm_file}: {e}")
            plt.plot(t_model, v_model, color=COLORS[rate], linestyle=':', linewidth=2,
                     label=f'VIBE {rate}')

    plt.title('Discharge Curves: VIBE vs PyBaMM', fontsize=14, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Terminal Voltage [V]', fontsize=12)
    plt.xlim(0, 3600)
    plt.ylim(2.4, 4.2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join('..', 'simulation_result', 'Cell_Verification_results', 'Discharge_Comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {os.path.abspath(plot_path)}")

if __name__ == '__main__':
    main()
