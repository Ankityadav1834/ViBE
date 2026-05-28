import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import CubicSpline

# Add the parent directory to path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ImplicitBatterySolver

# Configuration
BASE_CONFIG = {
    'n_series': 1,
    'n_parallel': 1,
    'topology': 'series_first',
    'device': 'cpu',
    'stress_options': {'enabled': False},   # disable stress for cleaner comparison
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n': 5, 'Nx_s': 5, 'Nx_p': 5,
    'Nsei': 1,
}

MODELS = {
    'interstitial-diffusion limited': 'interstitial_diffusion_limited',
    'reaction limited': 'reaction_limited',
    'solvent-diffusion limited': 'solvent_diffusion_limited'
}

COLORS = {
    'interstitial-diffusion limited': '#ff7f0e',
    'reaction limited': '#2ca02c',
    'solvent-diffusion limited': '#d62728'
}

I_DISCHARGE = 5.0
CUTOFF_VOLTAGE = 2.5
DT = 1.0
DT_MAX = 10.0

def run_sei_discharge(sei_model_name, i_discharge):
    print(f"Running 1C discharge for SEI model: {sei_model_name}...")
    cfg = dict(BASE_CONFIG)
    cfg['sei_options'] = {'enabled': True, 'sei_model': sei_model_name}
    
    battery = ImplicitBatterySolver(
        cfg, DISCRETIZATION, {},
        initial_state_mode='fully_charged',
    )
    
    t = 0.0
    dt = DT
    times = []
    lsei_vals = []
    
    times.append(t)
    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(battery.y, i_discharge)
        v_term = battery.get_exact_terminal_voltages(battery.y, I_cells).mean().item()
        lsei = battery.physics.state(battery.y, 'Lsei').mean().item() * 1e9  # m to nm
    lsei_vals.append(lsei)
    
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
            lsei = battery.physics.state(battery.y, 'Lsei').mean().item() * 1e9
        
        times.append(t)
        lsei_vals.append(lsei)
        
        if step_count % 100 == 0:
            print(f"  Time: {t:.1f}s, dt: {dt:.2f}s, Voltage: {v_term:.4f}V, Lsei: {lsei:.4f}nm")
        
        if v_term <= CUTOFF_VOLTAGE:
            print(f"  Reached cutoff voltage at {t:.1f}s")
            break
            
    return np.array(times), np.array(lsei_vals)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    plt.figure(figsize=(12, 7))
    
    for model_name, file_suffix in MODELS.items():
        # Run our model
        t_model, lsei_model = run_sei_discharge(model_name, I_DISCHARGE)
        
        # Load PyBaMM data from data subfolder
        pybamm_file = os.path.join('data', f'sei_thickness_{file_suffix}.csv')
        try:
            pybamm_data = np.loadtxt(pybamm_file, delimiter=',', skiprows=1)
            t_pybamm = pybamm_data[:, 0]
            lsei_pybamm = pybamm_data[:, 1] * 1e9  # Convert m to nm
            
            # Use cubic spline to create a smooth curve for PyBaMM plotting
            unique_idx = np.unique(t_pybamm, return_index=True)[1]
            t_pybamm = t_pybamm[unique_idx]
            lsei_pybamm = lsei_pybamm[unique_idx]
            
            cs = CubicSpline(t_pybamm, lsei_pybamm)
            t_smooth = np.linspace(t_pybamm[0], t_pybamm[-1], 1000)
            lsei_smooth = cs(t_smooth)
            
            # Interpolate PyBaMM smoothly onto model time to calculate error
            max_t = min(np.max(t_model), np.max(t_pybamm))
            valid_idx = t_model <= max_t
            
            lsei_pybamm_interp = cs(t_model[valid_idx])
            rmse = np.sqrt(np.mean((lsei_model[valid_idx] - lsei_pybamm_interp)**2))
            
            # Save VIBE results to CSV inside simulation_result/Cell_Verification_results/
            output_csv = os.path.join('..', 'simulation_result', 'Cell_Verification_results', f'vibe_sei_thickness_{file_suffix}.csv')
            np.savetxt(output_csv, np.column_stack((t_model, lsei_model)), delimiter=',', header='Time [s],SEI Thickness [nm]', comments='')
            
            # Plot PyBaMM (smooth)
            plt.plot(t_smooth, lsei_smooth, color=COLORS[model_name], linestyle='-', 
                     label=f'PyBaMM {model_name}')
            
            # Plot Model
            plt.plot(t_model, lsei_model, color=COLORS[model_name], linestyle=':', linewidth=2.5,
                     label=f'VIBE {model_name} (RMSE: {rmse:.4e} nm)')
            
            print(f"{model_name} RMSE: {rmse:.4e} nm")
        except Exception as e:
            print(f"Failed to load or process {pybamm_file}: {e}")
            plt.plot(t_model, lsei_model, color=COLORS[model_name], linestyle=':', linewidth=2.5,
                     label=f'VIBE {model_name}')

    plt.title('SEI Thickness Growth: VIBE vs PyBaMM (1C Discharge)', fontsize=14, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('SEI Thickness [nm]', fontsize=12)
    plt.xlim(0, 3600)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    
    plot_path = os.path.join('..', 'simulation_result', 'Cell_Verification_results', 'SEI_Comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {os.path.abspath(plot_path)}")

if __name__ == '__main__':
    main()
