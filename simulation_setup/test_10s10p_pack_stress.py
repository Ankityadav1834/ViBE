import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os

from main import ImplicitBatterySolver
from controllers import build_controller

# Configuration for the 10s10p battery pack
config = {
    'device': 'auto',  # Automatically use CUDA if available
    'n_series': 10,
    'n_parallel': 10,
    'electrolyte_spatial_method': 'finite_volume',
    'solid_spatial_method': 'chebyshev',
    'stress_options': {'enabled': True},  # Disabled for speed unless needed
    'sei_options': {'enabled': True, 'sei_model': 'reaction limited'},
}

# Standard grid discretization
discretization = {
    'Nr_n': 50, 'Nr_p': 50,
    'Nx_n': 15, 'Nx_s': 15, 'Nx_p': 15,
    'Nsei': 5
}

# The user requested that the entire 3rd parallel branch (index 2) is thermally disadvantaged
# Default hA is 0.0531. We apply the user's requested hA = 0.25
overrides = {
    (s, 2): {"hA": 0.25} for s in range(10)
}

# Optional: Add small contact resistance variations if desired
# overrides[(0, 2)]['R_contact'] = 0.002

initial_state_mode = 'fully_charged'

# 50 cycle Controller settings
# Note: CycleController compares cell limits, so voltages are cell-level (4.2V), 
# but currents are pack-level (50A = 1C for a 10P pack of 5Ah cells).
controller_config = {
    'cc_current': -50.0,         # 1C Charge
    'cv_voltage': 4.2,           # Max cell voltage
    'cutoff_current': 2.5,       # C/20 cutoff for the pack (2.5A)
    'discharge_current': 50.0,   # 1C Discharge
    'min_voltage': 2.5,          # Min cell voltage
    'max_voltage': 4.25,         # Hard safety cutoff
    'n_cycles': 100
}

# Optional thermal pack setup (using ambient strategy, relying on Gth for cell-to-cell)
thermal_options = {
    'enabled': True,
    'strategy': 'ambient',
    'model': 'builtin'
}

print("="*60)
print(f"Initializing {config['n_series']}S{config['n_parallel']}P Pack Simulation")
print("="*60)

# Build the solver
battery_solver = ImplicitBatterySolver(
    config,
    discretization,
    overrides=overrides,
    initial_state_mode=initial_state_mode,
    thermal_options=thermal_options
)

print(f"Solver running on device: {battery_solver.device}")
print(f"Thermally disadvantaged branch (p=2) applied to {config['n_series']} cells.")

# Build controller
controller = build_controller('cycle_cccv', **controller_config)

# Run the simulation
battery_solver.simulate(
    t_end=10 * 24 * 3600,   # Large maximum time (10 days) to allow 50 cycles to finish
    dt_init=1.0,
    controller=controller,
    dt_max=200.0             # Max time step
)

print("\nSimulation complete. Results saved to simulation_results.npz")
