import torch
from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ==============================================================================
# 1. Physics & Simulation Settings
# ==============================================================================
config = {
    'device': 'auto',   # 'auto' = use GPU if available, or set 'cpu' / 'cuda' explicitly
    'n_series': 2,      
    'n_parallel': 1,   
    'electrolyte_spatial_method': 'finite_volume',  
    'solid_spatial_method': 'chebyshev',            
    'stress_options': {
        'enabled': True,      
        'initial': 0.0,        
        'scale': 1e6,          
        'diffusivity': 1e-12,  
        'relaxation': 1e-4,    
        'coupling': 1.0,       
        'force_area': 0.1027   
    },
    'sei_options': {
        'enabled': True
    }
}

# Discretization parameters
discretization = {
    'Nr_n': 10,  'Nr_p': 10,  
    'Nx_n': 10,  'Nx_s': 10,  'Nx_p': 10,  
    'Nsei': 1    
}

# ==============================================================================
# 2. Operating Scenario
# ==============================================================================
controller_config = {
    'cycle_cccv': {
        'cc_current': -10.0,         # Charging current (A)
        'cv_voltage': 4.2,          # CV voltage (V)
        'cutoff_current': 1.0,      # Cutoff current for CV (A)
        'discharge_current': 10.0,  # Discharge current (A)
        'min_voltage': 2.4,         # Discharge cutoff voltage (V)
        'max_voltage': 4.4,         # Charge cutoff voltage (V)
        'n_cycles': 1          
    }
}

# Optional Initial State Options
initial_state_mode = 'fully_charged' 
initial_state_options = {
    'cutoff_voltage': 2.5,
    'discharge_current': 10.0,
    'dt': 1.0,
    'max_time': 5000.0,
    'coarse_dt': 10.0,
    'refine_margin': 0.08
}

# ==============================================================================
# 3. Pack Options (Thermal / Balancing)
# ==============================================================================
balancing_options = {
    'enabled': False,
    'strategy': 'passive',
    'r_bleed': 5.0,
    'v_threshold': 4.0,
}

thermal_options = {
    'enabled': True,
    'strategy': 'ambient', 
    'ambient_temp': 298.15,
}

# ==============================================================================
# 4. Output Specification (liionpack-style selective saving)
# ==============================================================================
# Choose which variables to compute and save to simulation_results.npz.
# - "default"  → voltage, current, SOC, temperature, SEI thickness, pack V/I,
#                 all overpotentials (rxn, ohmic solid/electrolyte, conc, SEI,
#                 contact resistance drop, busbar resistance drop)
# - "all"      → every available output
# - list       → only the named outputs
#
# Resistance values in standard parameters now match liionpack defaults:
#   R_contact = 1e-2 Ω  (Tranter et al. 2022, Rc = 10 mΩ)
#   R_bus     = 1e-4 Ω  (Tranter et al. 2022, Rb = 0.1 mΩ)
output_spec = OutputSpec("default")

# Alternative examples:
# output_spec = OutputSpec(["terminal_voltage", "temperature", "soc"])
# output_spec = OutputSpec(["default", "contact_resistance_drop"])
# output_spec = OutputSpec("all")

# ==============================================================================
# 5. Run Simulation
# ==============================================================================
if __name__ == "__main__":
    solver = ImplicitBatterySolver(
        config,
        discretization,
        overrides={},
        initial_state_mode=initial_state_mode,
        initial_state_options=initial_state_options,
        balancing_options=balancing_options,
        thermal_options=thermal_options
    )

    controller = build_controller('cycle_cccv', **controller_config['cycle_cccv'])
    
    print("Starting simulation with fully modular equations...")
    solver.simulate(
        t_end=360000, 
        dt_init=1.0,
        controller=controller,
        dt_max=50.0,
        output_spec=output_spec,
    )
