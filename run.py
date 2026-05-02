from main import ImplicitBatterySolver
from controllers import build_controller

# Configuration for the battery pack
config = {
    'n_series': 4,      # Number of cells in series
    'n_parallel': 2,   # Number of cells in parallel
    'stress_options': {
        'enabled': True,      # Set True to add the example stress PDE state.
        'initial': 0.0,        # Initial stress-like field value [Pa]
        'scale': 1e6,          # Newton scaling for stress [Pa]
        'diffusivity': 1e-12,  # Stress smoothing coefficient
        'relaxation': 1e-4,    # Stress relaxation rate [1/s]
        'coupling': 1.0,       # Source strength from electrolyte concentration deviation
        'force_area': 0.1027   # Area used for derived force_from_stress output
    }
}

# Controller settings
controller_config = {
    'constant_current': {
        'current': 15.0
    },
    'current_profile': {
        'time_points': [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0, 4800.0, 5400.0, 6000.0, 6600.0, 7000.0],
        'current_points': [-30.0, 50.0, 100.0, 0.0, -100.0, 50.0, -30.0, 80.0, -80.0, 20.0, -20.0, 0.0, -50.0]
    },
    'cc_cv': {
        'cc_current': -25.0,
        'cv_voltage': 4.2,
        'cutoff_current': 1.0
    },
    'mpc': {
        'n_parallel': config['n_parallel'],
        't_limit': 313.15,
        'target_c_rate': 2.0
    },
    'cycle_cccv': {
        'cc_current': -15.0,         # Charging current (A)
        'cv_voltage': 4.2,          # CV voltage (V)
        'cutoff_current': 1.0,      # Cutoff current for CV (A)
        'discharge_current': 10.0,  # Discharge current (A)
        'min_voltage': 2.4,         # Discharge cutoff voltage (V)
        'max_voltage': 4.2,         # Charge cutoff voltage (V)
        'n_cycles': 200               # Number of cycles
    }
}

solver_method = 'basic'   # 'basic' or 'advanced'
pack_current = 20.0

# Discretization parameters
discretization = {
    'Nr_n': 10,  # Radial nodes in anode
    'Nr_p': 10,  # Radial nodes in cathode
    'Nx_n': 10,  # Axial nodes in anode
    'Nx_s': 10,  # Axial nodes in separator
    'Nx_p': 10,  # Axial nodes in cathode
    'Nsei': 1    # SEI nodes
}

# Parameter overrides (if any)
overrides = {(2,2): {'hA_ambient': 0.0931}}  # e.g., {(0,0): {'T_amb': 308.15}} for cell 0,0

# Initial state selection
initial_state_mode = 'fully_charged'   # 'fully_charged' or 'fully_discharged'
initial_state_options = {
    'cutoff_voltage': 2.5,
    'discharge_current': 10.0,
    'dt': 1.0,
    'max_time': 5000.0,
    'coarse_dt': 10.0,
    'refine_margin': 0.08
}

# Optional pack subsystems
balancing_options = {
    'enabled': True,
    'strategy': 'passive',   # 'none', 'passive', 'active_capacitor', 'active_inductor'
    'r_bleed': 5.0,
    'v_threshold': 4.0,
    'r_eq': 0.1,
    'transfer_gain': 1.5
}

thermal_options = {
    'enabled': True,
    'strategy': 'ambient',   # 'ambient', 'liquid', 'pcm'
    'hA_scale': 1.0,
    'ambient_temp': 298.15,
    'hA_contact': 2.5,
    'm_dot_cp': 5.0,
    'inlet_temp': 298.15,
    'hA_ambient': 0.0531,
    'melt_temp': 308.15,
    'latent_heat': 50000.0,
    'smoothing_width': 1.5
}

# Create the battery solver
battery_solver = ImplicitBatterySolver(
    config,
    discretization,
    overrides,
    initial_state_mode=initial_state_mode,
    initial_state_options=initial_state_options,
    balancing_options=balancing_options,
    thermal_options=thermal_options
)

"""
Simulation mode selection
Set use_controller = True and controller_strategy = 'cycle_cccv' to enable cycling mode.
"""
use_controller = True
controller_strategy = 'cycle_cccv'   # 'constant_current', 'current_profile', 'cc_cv', 'mpc', 'cycle_cccv'
controller_dt_max = 50.0

if use_controller:
    controller = build_controller(controller_strategy, **controller_config[controller_strategy])
    battery_solver.simulate(
        t_end=3600000,  # Large enough to allow all cycles
        dt_init=1.0,
        controller=controller,
        dt_max=controller_dt_max
    )
else:
    battery_solver.simulate(
        t_end=500,
        dt_init=1.0,
        I_pack=pack_current,
        method=solver_method
    )
