import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

# ---------------------------------------------------------------------------
# Disable plt.show() temporarily to prevent the solver's internal cycle plots 
# from blocking the loop execution. We will restore it at the end.
# ---------------------------------------------------------------------------
plt_show_orig = plt.show
plt.show = lambda: None

from main import ImplicitBatterySolver
from controllers import build_controller

# Define the C-rates you want to compare
C_RATES = [0.5, 1.0, 2.0, 3.0]

# Number of cycles. 
# Note: 200 cycles of a full PDE model can take several hours! 
# We set it to 20 here so you can test it reasonably fast, but feel free to change it to 200.
N_CYCLES = 20

# Use 1 cell to speed up the experiment
config = {
    'n_series': 1,
    'n_parallel': 1,
    'stress_options': {'enabled': False}
}

# Standard grid
discretization = {
    'Nr_n': 10, 'Nr_p': 10, 'Nx_n': 5, 'Nx_s': 5, 'Nx_p': 5, 'Nsei': 1
}

summary_data = []

print("==================================================")
print(f" Starting Capacity Fade Experiment ({N_CYCLES} cycles)")
print("==================================================")

for c_rate in C_RATES:
    print(f"\n---> Running Simulation for {c_rate}C Discharge <---")
    
    # 1C current for a 5Ah nominal cell is roughly 5A
    discharge_current = c_rate * 5.0
    
    # Keep charge rate fixed at 1C to isolate the stress caused purely by fast discharging
    charge_current = -5.0  
    
    controller_config = {
        'cc_current': charge_current,
        'cv_voltage': 4.2,
        'cutoff_current': 1.0,
        'discharge_current': discharge_current,
        'min_voltage': 2.5,
        'max_voltage': 4.2,
        'n_cycles': N_CYCLES
    }
    
    battery_solver = ImplicitBatterySolver(
        config,
        discretization,
        overrides={},
        initial_state_mode='fully_charged'
    )
    
    controller = build_controller('cycle_cccv', **controller_config)
    
    # Run simulation
    battery_solver.simulate(
        t_end=36000000,
        dt_init=1.0,
        controller=controller,
        dt_max=50.0
    )
    
    print(f"Extracting data for {c_rate}C...")
    df = pd.read_csv('all_results.csv')
    df_out = df[df['kind'] == 'output']
    
    # Extract the custom outputs we added to the output manager
    fade = df_out[df_out['name'] == 'capacity_fade_pct']['value'].values
    stress = df_out[df_out['name'] == 'dis_stress_vm_peak']['value'].values
    sei = df_out[df_out['name'] == 'sei_thickness_nm']['value'].values
    time_arr = df_out[df_out['name'] == 'capacity_fade_pct']['time'].values
    
    # Save a clean CSV just for this C-rate
    c_rate_df = pd.DataFrame({
        'time_hrs': time_arr / 3600.0,
        'capacity_fade_pct': fade,
        'dis_stress_vm_peak_MPa': stress / 1e6,
        'sei_thickness_nm': sei
    })
    c_rate_df.to_csv(f'results_{c_rate}C.csv', index=False)
    
    summary_data.append({
        'c_rate': c_rate,
        'time': time_arr / 3600.0,
        'fade': fade,
        'stress': stress / 1e6, # Convert Pa to MPa
        'sei': sei
    })
    
    # Clean up memory before the next cycle
    del battery_solver, controller, df, df_out, c_rate_df
    gc.collect()

print("\nGenerating comparison plots...")

# Restore plt.show
plt.show = plt_show_orig

# Create the final comparison plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for data in summary_data:
    axs[0].plot(data['time'], data['fade'], label=f"{data['c_rate']}C")
    axs[1].plot(data['time'], data['stress'], label=f"{data['c_rate']}C", alpha=0.7)
    axs[2].plot(data['time'], data['sei'], label=f"{data['c_rate']}C")

axs[0].set_title('Capacity Fade vs Time')
axs[0].set_ylabel('Capacity Fade [%]')
axs[0].set_xlabel('Time [hours]')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].set_title('DIS Peak Stress vs Time')
axs[1].set_ylabel('Von Mises Stress [MPa]')
axs[1].set_xlabel('Time [hours]')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

axs[2].set_title('SEI Thickness vs Time')
axs[2].set_ylabel('SEI Thickness [nm]')
axs[2].set_xlabel('Time [hours]')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Experiment1_Plots.png', dpi=300)
print("Experiment completed! Plots saved to 'Experiment1_Plots.png'.")

# Show the plot window at the very end
plt.show()
