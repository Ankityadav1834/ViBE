<div align="center">

# ⚡ VIBE — Virtual Battery Environment 

**A physics-based, GPU-accelerated battery pack simulation framework built with PyTorch.**  
Designed for researchers studying electrochemical degradation, pack heterogeneity, thermal management, and optimal charging control.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

1. [What is VIBE?](#1-what-is-vibe)
2. [Key Features](#2-key-features)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [User Guides](#6-user-guides)
   - [6.1 Running a Basic Discharge Simulation](#61-running-a-basic-discharge-simulation)
   - [6.2 Using a Different Chemistry (Na-ion, LFP, etc.)](#62-using-a-different-chemistry-na-ion-lfp-etc)
   - [6.3 Changing the Charging Control Strategy](#63-changing-the-charging-control-strategy)
   - [6.4 Implementing a Custom MPC Controller](#64-implementing-a-custom-mpc-controller)
   - [6.5 Changing the Cooling Strategy](#65-changing-the-cooling-strategy)
   - [6.6 Setting Up a Heterogeneity Analysis](#66-setting-up-a-heterogeneity-analysis)
   - [6.7 Adding a New Equation to the Physics Model](#67-adding-a-new-equation-to-the-physics-model)
   - [6.8 Adding a New SEI Degradation Model](#68-adding-a-new-sei-degradation-model)
   - [6.9 Configuring Pack Topology and Balancing](#69-configuring-pack-topology-and-balancing)
   - [6.10 Selecting Outputs and Reading Results](#610-selecting-outputs-and-reading-results)
7. [Physics Model Reference](#7-physics-model-reference)
8. [Configuration Reference](#8-configuration-reference)
9. [API Reference](#9-api-reference)
10. [Extending VIBE](#10-extending-vibe)

---

## 1. What is ViBE?

**ViBE** (**Vi**rtual **B**attery **E**nvironment) is a modular, GPU-accelerated Python framework for high-fidelity, coupled electrochemical–thermal–degradation simulation of large-scale heterogeneous battery packs.

While traditional battery simulators rely on explicit solvers that struggle to scale, ViBE solves the **Doyle–Fuller–Newman (DFN)** equations alongside thermal and degradation models using a fully implicit Backward-Euler Newton–Raphson solver with adaptive time-stepping. 

Built entirely on PyTorch, ViBE leverages batched tensor operations (`torch.vmap`) to parallelise multi-cell pack simulations across CPUs and GPUs. Furthermore, its fully differentiable architecture enables **exact local differential sensitivity analysis** (e.g., computing the exact gradient of temperature with respect to initial state-of-charge) directly through reverse-mode automatic differentiation (`torch.func.jacrev`), completely eliminating the truncation errors and massive computational overhead of traditional finite-difference methods.

### Core Capabilities & Physics

- **Modular PDE Framework**: Define conservation-law PDEs via an abstract syntax tree (AST) with interchangeable discretisations (Finite Volume or Chebyshev spectral collocation).
- **GPU-Accelerated Pack Scalability**: A generalised Kirchhoff-network solver tightly coupled to per-cell DFN physics.
- **Coupled Degradation**: A plug-and-play registry of 7 mechanistically distinct SEI growth models (solvent-diffusion limited, reaction limited, tunneling, etc.).
- **Thermal Heterogeneity**: Local Joule and entropic heat generation coupled with ambient convective, active liquid, or phase-change material (PCM) cooling.
- **Electrochemical**: Solid diffusion in particles, through-cell electrolyte transport, and Butler–Volmer kinetics.
- **Pack-level current distribution** for multi-cell S×P configurations

### What makes it different?

| Feature | VIBE | Typical ODE solvers |
|---------|------|---------------------|
| Implicit solver | ✅ (Newton–Raphson + adaptive dt) | ❌ (explicit, small dt required) |
| GPU acceleration | ✅ (via torch.vmap) | ❌ |
| Automatic differentiation | ✅ (torch.func.jacrev) | ❌ |
| Modular physics | ✅ (registry-based) | ❌ |
| CSV-based parameter loading | ✅ | ❌ |
| Multi-cell heterogeneity | ✅ | Rarely |

---

## 2. Key Features

- **5 charging strategies**: Constant Current, Current Profile, CC-CV, MPC (via `do_mpc`), CC-CV Cycling
- **3 thermal models**: Lumped convective (`builtin`), Isothermal, Entropic (Bernardi 1985)
- **3 cooling strategies**: Ambient convection, Active liquid cooling, Phase-change material (PCM)
- **7 SEI growth models**: PyBaMM-compatible (solvent-diffusion, reaction, EC, tunneling, ...)
- **4 balancing strategies**: None, Passive resistive, Active capacitor, Active inductor
- **2 pack topologies**: Series-first (`S-P`) and Parallel-first (`P-S`)
- **Dynamic parameter loading**: Drop in CSV files to swap chemistries without changing any code
- **Modular state equations**: Add new PDEs (stress, lithium plating, ...) via a registry
- **Structured output**: `.npz` results files + CSV summary for every simulation

---

## 3. Repository Structure

```
Vibe_Battery/
│
├── main.py                     ← Core solver: BatteryPhysics + ImplicitBatterySolver
├── electrochemistry.py         ← Butler-Volmer, OCP, circuit parameters
├── model_registry.py           ← STATE_EQUATIONS and DERIVED_OUTPUTS registry
├── controllers.py              ← All charging control strategies
├── solver.py                   ← BasicSolver, AdvancedSolver, ControlledSolver
├── sei_models.py               ← 7 PyBaMM-compatible SEI growth models
├── temperature_models.py       ← Thermal model library (builtin / isothermal / entropic)
├── pack_modules.py             ← Balancing & cooling strategies
├── parameter_loader.py         ← CSV-based chemistry loader (TorchInterpolant)
├── pde_framework.py            ← Finite Volume / Chebyshev PDE operators
├── states/                     ← Enabled state equation specs
├── pde_sim/                    ← Output management (OutputSpec, ResultsProcessor)
├── sim_config.py               ← Ready-to-run annotated config template
│
├── parameters/                 ← Chemistry parameter folders (add yours here)
│   ├── li_ion_chen2020/
│   │   └── params.json         ← LG M50 Li-ion scalar parameters
│   └── na_ion_chayambuka2022/
│       ├── params.json         ← Na-ion scalar parameters
│       ├── U_n.csv  U_p.csv    ← OCP curves
│       ├── D_n.csv  D_p.csv    ← Solid diffusivity (concentration-dependent)
│       ├── k_n.csv  k_p.csv    ← Reaction rate constants
│       ├── D_e.csv             ← Electrolyte diffusivity
│       └── sigma_e.csv         ← Electrolyte conductivity
│
├── simulation_setup/           ← All experiment scripts (run from here)
│   ├── run.py                  ← General-purpose simulation launcher
│   ├── Experiment1.py          ← Basic discharge/charge cycles
│   ├── Experiment_Heterogeneity.py   ← Pack heterogeneity study
│   ├── Experiment_Sensitivity.py     ← Parameter sensitivity analysis
│   ├── Experiment_SOC_Sensitivity.py ← SOC-dependent sensitivity
│   ├── Experiment_AutogradSensitivity.py ← Autograd-based gradient study
│   ├── Discharge_Cycle.py      ← Verification vs PyBaMM
│   ├── SEI_Comparison.py       ← SEI model comparison
│   ├── Example_ParameterLoading.py ← Chemistry loading example
│   └── data/                   ← Reference CSVs (PyBaMM verification data)
│
└── simulation_result/          ← All outputs go here (auto-created)
    ├── <run_name>/
    │   └── results/
    │       ├── simulation_results.npz
    │       └── all_results.csv
    └── heterogeneity_results/
```

---

## 4. Installation

### Prerequisites

- Python 3.10 or newer
- PyTorch 2.0+ ([install guide](https://pytorch.org/get-started/locally/))

### Step 1: Clone the repository

```bash
git clone https://github.com/your-org/Vibe_Battery.git
cd Vibe_Battery
```

### Step 2: Install dependencies

```bash
pip install torch numpy pandas matplotlib scipy
```

### Step 3 (optional): GPU support

VIBE automatically uses CUDA if available. No code changes needed — just set `'device': 'auto'` in the config.

### Step 4 (optional): MPC controller support

The MPC charging strategy requires `do_mpc`:

```bash
pip install do-mpc
```

### Step 5: Verify the installation

```bash
python simulation_setup/Example_ParameterLoading.py
```

Expected output:
```
Available chemistries in 'C:\Vibe_Battery\parameters':
  li_ion_chen2020
  na_ion_chayambuka2022

=== 1. Default Li-ion ===
Li-ion OC terminal voltage: 4.0483 V
[OK] All three loading methods work correctly.
```

---

## 5. Quick Start

The fastest way to run a simulation is through `sim_config.py`:

```bash
python sim_config.py
```

This runs a 1-cycle CC-CV charge/discharge on a 2S×1P Li-ion pack and saves results to `simulation_result/sim_config/results/`.

Or use the general launcher:

```bash
python simulation_setup/run.py
```

Results are always saved as:
- `simulation_result/<run_name>/results/simulation_results.npz` — all time-series data
- `simulation_result/<run_name>/results/all_results.csv` — same data in tabular form

---

## 6. User Guides

### 6.1 Running a Basic Discharge Simulation

The minimal setup requires three objects: a **config dict**, a **discretization dict**, and a solver instance.

```python
from main import ImplicitBatterySolver

config = {
    'device': 'auto',        # 'cpu' or 'cuda' to force
    'n_series': 1,
    'n_parallel': 1,
    'electrolyte_spatial_method': 'finite_volume',   # or 'chebyshev'
    'solid_spatial_method': 'chebyshev',
    'sei_options': {'enabled': False},    # disable SEI for speed
    'stress_options': {'enabled': False}, # disable mechanical stress
}

discretization = {
    'Nr_n': 10,   # radial nodes in anode particle
    'Nr_p': 10,   # radial nodes in cathode particle
    'Nx_n': 10,   # axial nodes in anode electrode
    'Nx_s': 10,   # axial nodes in separator
    'Nx_p': 10,   # axial nodes in cathode electrode
    'Nsei': 1,    # SEI state nodes (keep at 1)
}

solver = ImplicitBatterySolver(config, discretization, overrides={})

# Run a 1-hour discharge at 5 A
solver.simulate(t_end=3600.0, dt_init=1.0, I_pack=5.0, method='basic')
```

**Choosing solver method:**

| Method | Description | When to use |
|--------|-------------|-------------|
| `'basic'` | Adaptive Newton (no LTE control) | Quick runs, debugging |
| `'advanced'` | PID-controlled LTE step-size | Accurate long simulations |

**Choosing discretization fidelity:**

| Resolution | `Nr_n/p` | `Nx_n/s/p` | Speed | Accuracy |
|------------|---------|-----------|-------|----------|
| Coarse | 5 | 5 | Very fast | Low |
| Medium | 10 | 10 | Fast | Good |
| Fine | 20 | 20 | Slow | High |
| Research | 50 | 30 | Very slow | Excellent |

---

### 6.2 Using a Different Chemistry (Na-ion, LFP, etc.)

VIBE uses a **parameter folder** system. Each folder contains a `params.json` (scalar constants) and optional CSV files for concentration/stoichiometry-dependent functions.

#### Option A — Load by folder path (recommended)

```python
solver = ImplicitBatterySolver(
    config, discretization, {},
    chemistry_folder='parameters/na_ion_chayambuka2022'
)
```

#### Option B — Pre-load and inspect

```python
from parameter_loader import load_chemistry_from_folder

chem = load_chemistry_from_folder('parameters/na_ion_chayambuka2022')
print(chem['name'])           # "na_ion_chayambuka2022"
print(chem['params']['Ln'])   # electrode thickness

solver = ImplicitBatterySolver(config, discretization, {}, chemistry=chem)
```

#### Option C — List all available chemistries

```python
from parameter_loader import list_available_chemistries
list_available_chemistries()   # prints all valid folders under parameters/
```

#### Adding a new chemistry from scratch

1. Create a folder: `parameters/my_lfp_cell/`

2. Create `params.json` (copy from `li_ion_chen2020/params.json` and edit):

```json
{
  "_description": "LFP cell – your reference",
  "Ln": 7.0e-05,
  "Lp": 6.0e-05,
  "Ls": 2.5e-05,
  "A":  0.1,
  "cs_max_n": 29583.0,
  "cs_max_p": 22806.0,
  "cs_n_init": 26624.0,
  "cs_p_init": 1141.0,
  ...
}
```

3. (Optional) Add CSV files for concentration-dependent properties:

| File | X-axis | Y-axis | Units |
|------|--------|--------|-------|
| `U_n.csv` | Stoichiometry (0–1) | OCP | V |
| `U_p.csv` | Stoichiometry (0–1) | OCP | V |
| `D_n.csv` | Concentration | Solid diffusivity | m²/s |
| `D_p.csv` | Concentration | Solid diffusivity | m²/s |
| `k_n.csv` | Concentration | Exchange current rate | m/s |
| `k_p.csv` | Concentration | Exchange current rate | m/s |
| `D_e.csv` | Concentration | Electrolyte diffusivity | m²/s |
| `sigma_e.csv` | Concentration | Electrolyte conductivity | S/m |

Each CSV must have a **header row** and **two data columns** (x, y). Example (`U_n.csv`):
```
Negative particle stoichiometry,OCP [V]
0.001, 1.319
0.060, 0.845
0.128, 0.577
...
```

4. Use it:
```python
solver = ImplicitBatterySolver(config, disc, {},
    chemistry_folder='parameters/my_lfp_cell')
```

> **Note:** Scalar fallbacks apply — if you omit `D_n.csv`, `Ds_n` from `params.json` is used as a constant. You only need to provide CSVs for parameters that vary with concentration.

---

### 6.3 Changing the Charging Control Strategy

All control strategies are in `controllers.py` and selected by name via `build_controller()`.

#### Available strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| Constant Current | `'constant_current'` | Fixed current throughout |
| Current Profile | `'current_profile'` | Piecewise linear current vs time |
| CC-CV | `'cc_cv'` | Standard charge profile |
| MPC | `'mpc'` | Model Predictive Control (requires `do_mpc`) |
| CC-CV Cycling | `'cycle_cccv'` | Automated multi-cycle testing |

#### Example: Standard CC-CV Charge

```python
from controllers import build_controller

controller = build_controller(
    'cc_cv',
    cc_current=-10.0,      # CC phase current [A] (negative = charging)
    cv_voltage=4.2,        # CV target voltage [V]
    cutoff_current=1.0,    # Stop when current drops below this [A]
)

solver.simulate(
    t_end=10000,
    dt_init=1.0,
    controller=controller,
)
```

#### Example: Multi-cycle CC-CV test

```python
controller = build_controller(
    'cycle_cccv',
    cc_current=-5.0,           # Charge current [A]
    cv_voltage=4.2,            # Charge cutoff voltage [V]
    cutoff_current=0.5,        # CV phase end current [A]
    discharge_current=5.0,     # Discharge current [A]
    min_voltage=2.5,           # Discharge cutoff [V]
    n_cycles=10,               # Number of full cycles
)

solver.simulate(t_end=36_000_000, dt_init=1.0, controller=controller, dt_max=50.0)
```

#### Example: Arbitrary current profile

```python
controller = build_controller(
    'current_profile',
    time_points=[0, 600, 1200, 1800, 2400],   # seconds
    current_points=[5.0, 10.0, -5.0, 0.0, 8.0],  # amperes
)
```

---

### 6.4 Implementing a Custom MPC Controller

VIBE ships with a `do_mpc`-based MPC that maximises SOC while limiting temperature and SEI growth. Here is how to write your own controller from scratch.

#### Step 1: Subclass `BaseController`

```python
# In controllers.py or your own file
from controllers import BaseController
import torch

class MyCustomMPC(BaseController):
    def __init__(self, max_current=10.0, temp_limit=318.15, **kwargs):
        super().__init__(initial_current=0.0, stop_on_voltage_limits=True, **kwargs)
        self.max_current = max_current
        self.temp_limit  = temp_limit
        self.current_stage = "MPC"

    def compute_current(self, t, y_state, solver, dt_sim):
        """
        Called at every time step.

        Parameters
        ----------
        t        : float   — current simulation time [s]
        y_state  : Tensor  — current state [n_cells, state_size]
        solver   : ImplicitBatterySolver
        dt_sim   : float   — current time step [s]

        Returns
        -------
        float — the desired pack-level current [A]
              (positive = discharge, negative = charge)
        """
        # Read current state
        cell_voltages = solver.get_exact_terminal_voltages(y_state, self.last_i if hasattr(self, 'last_i') else 0.0)
        max_temp = torch.max(y_state[:, -1]).item()  # last state = temperature

        # Simple rule: if temperature is high, reduce current
        if max_temp > self.temp_limit:
            I = self.max_current * 0.5
        else:
            I = self.max_current

        self.last_i = I
        return I

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        """Return (True, "reason") to stop the simulation, (False, "") otherwise."""
        min_v = torch.min(cell_voltages).item()
        if min_v <= self.min_voltage:
            return True, f"Min voltage reached at t={t:.1f}s"
        return False, ""
```

#### Step 2: Use it in a simulation

```python
controller = MyCustomMPC(max_current=10.0, temp_limit=318.15, min_voltage=2.5)

solver.simulate(
    t_end=7200,
    dt_init=1.0,
    controller=controller,
    dt_max=10.0,
)
```

#### Accessing physics state inside the controller

```python
# y_state shape: [n_cells, state_size]
# Useful slices (use physics.state_layout for exact indices):

# Temperature of each cell (always the last state variable)
T_cells = y_state[:, -1]          # [n_cells]  Kelvin

# Get voltage
voltages = solver.get_exact_terminal_voltages(y_state, current)

# Get SOC (approximate)
cs_n = solver.physics.state(y_state, 'cs_n')   # [n_cells, Nr_n]
soc  = cs_n[:, -1] / solver.physics.params['cs_max_n']
```

#### Registering with `build_controller` (optional)

Add to the `build_controller` function in `controllers.py`:

```python
if strategy == "my_mpc":
    return MyCustomMPC(**kwargs)
```

---

### 6.5 Changing the Cooling Strategy

Cooling strategies are configured via `thermal_options` passed to `ImplicitBatterySolver`. They are implemented in `pack_modules.py`.

#### Available strategies

| Strategy | `strategy` key | Description |
|----------|---------------|-------------|
| Ambient convection | `'ambient'` | Convective cooling to a fixed ambient temperature |
| Active liquid cooling | `'liquid'` | Coolant flow through a cold-plate |
| Phase-change material | `'pcm'` | Latent-heat buffer at a melt temperature |

#### Example: Ambient convection (default)

```python
thermal_options = {
    'enabled': True,
    'strategy': 'ambient',
    'hA_scale': 1.0,         # multiplier on the per-cell hA from params
    'ambient_temp': 298.15,  # override ambient temperature [K]
}
```

#### Example: Active liquid cooling

```python
thermal_options = {
    'enabled': True,
    'strategy': 'liquid',
    'hA_contact': 5.0,       # [W/K] thermal contact conductance to fluid
    'm_dot_cp':  10.0,        # [W/K] coolant flow rate × specific heat
    'inlet_temp': 293.15,    # [K] coolant inlet temperature
}
```

The model passes coolant sequentially through series-connected cells (cell 0 → cell N-1), so inlet-end cells run cooler.

#### Example: Phase-change material (PCM)

```python
thermal_options = {
    'enabled': True,
    'strategy': 'pcm',
    'melt_temp': 308.15,      # [K] PCM melting point (e.g. 35 °C paraffin)
    'latent_heat': 180000.0,  # [J/kg] specific latent heat × density
    'smoothing_width': 1.5,   # [K] Gaussian width of the delta approximation
    'hA_ambient': 0.05,       # [W/K] outer ambient convection (background)
    'ambient_temp': 298.15,
}
```

#### Adding a custom cooling strategy

1. Open `pack_modules.py`
2. Subclass `ThermalStrategy`:

```python
class MyCustomCooling(ThermalStrategy):
    name = "custom"

    def __init__(self, my_param=1.0, **kwargs):
        self.my_param = my_param

    def compute_temperature_rhs(self, battery, temperatures, heat_generation):
        """
        Parameters
        ----------
        battery      : ImplicitBatterySolver — access .Cth, .Gth, .h_amb, etc.
        temperatures : Tensor [n_cells, 1]
        heat_generation : Tensor [n_cells, 1]

        Returns
        -------
        Tensor [n_cells, 1]  — dT/dt [K/s]
        """
        q_cool = self.my_param * (temperatures - 298.15)
        q_cond = battery.Gth @ temperatures - battery.Gth.sum(1, True) * temperatures
        return (q_cond - q_cool) / (battery.Cth + 1e-12)
```

3. Register in `build_thermal_module`:
```python
if strategy == "custom":
    return MyCustomCooling(**kwargs)
```

4. Use:
```python
thermal_options = {'enabled': True, 'strategy': 'custom', 'my_param': 2.0}
```

---

### 6.6 Setting Up a Heterogeneity Analysis

Heterogeneity is introduced via **cell-level overrides** — you change one or more parameters of specific cells while leaving others at default.

#### Basic override syntax

```python
# Format: {(series_idx, parallel_idx): {param: new_value}}
overrides = {
    (0, 1): {'hA': 0.02},            # cell at row 0, col 1 has worse cooling
    (1, 0): {'R_contact': 0.05},     # cell at row 1, col 0 has bad connection
}

solver = ImplicitBatterySolver(
    config, discretization, overrides,
    initial_state_mode='fully_charged',
)
```

#### Parameters commonly varied for heterogeneity studies

| Parameter | Physical meaning | Typical defect |
|-----------|-----------------|----------------|
| `hA` | Cooling coefficient [W/K] | Blocked cooling channel: ×0.5 |
| `R_contact` | Contact resistance [Ω] | Weld defect: ×3 |
| `Ds_n` | Anode solid diffusivity [m²/s] | Particle cracking: ×0.7 |
| `m_ref_n` | Exchange current density | Surface film: ×0.8 |
| `T_amb` | Ambient temperature [K] | Hot-spot environment |
| `eps_e_n` | Anode porosity | Electrolyte starvation |
| `Lsei_0` | Initial SEI thickness [m] | Aged cell in fresh pack |

#### Full multi-scenario study (see `Experiment_Heterogeneity.py`)

```python
SCENARIOS = {
    'R_contact_high': {(0, 1): {'R_contact': 0.03}},   # +200% contact R
    'hA_low':         {(0, 1): {'hA': 0.025}},          # -50% cooling
    'Ds_n_low':       {(0, 1): {'Ds_n': 2.6e-14}},      # -20% anode Ds
}

for name, overrides in SCENARIOS.items():
    solver = ImplicitBatterySolver(config, disc, overrides)
    solver.simulate(t_end=..., controller=..., run_name=f'hetero_{name}')
```

Results will appear in `simulation_result/hetero_<name>/results/`.

#### Useful derived metrics (computed automatically)

- `temperature` per cell
- `sei_thickness_nm` per cell
- `capacity_fade_pct` per cell
- `cell_current` — current distribution across cells
- `soc` per cell

---

### 6.7 Adding a New Equation to the Physics Model

The state equation system is fully modular. All state variables (solid concentration, electrolyte, SEI thickness, temperature, mechanical stress) are registered in `model_registry.py` under `STATE_EQUATIONS`. 

ViBE allows you to declare complex, spatially distributed conservation-law PDEs using its operator framework, which automatically handles boundary conditions and Finite Volume / Chebyshev discretization.

Here is a complex example demonstrating how the **mechanical stress PDE** is implemented across the through-cell domain:

#### Step 1: Define the Flux and Source Terms

First, define how the field diffuses and what generates it. The framework provides the context of the entire cell (concentrations, temperature, SEI growth).

```python
# Define how stress propagates (flux coefficient)
def stress_flux_coefficient(physics, y_flat, i_app, p, context):
    options = physics.state_equation_options.get("stress", {})
    stress = physics.state(y_flat, "stress")
    diffusivity = torch.ones_like(stress) * float(options.get("diffusivity", 1e-12))
    
    # physics.through_cell_flux_coefficient handles the grid mapping automatically
    return physics.through_cell_flux_coefficient(diffusivity)

# Define what generates stress (source term)
def stress_source(physics, y_flat, i_app, p, context):
    options = physics.state_equation_options.get("stress", {})
    stress = physics.state(y_flat, "stress")
    
    # Example: stress is generated by SEI growth and solid concentration gradients
    dlsei_dt = context.get('dLsei_dt', torch.zeros(physics.n_cells, 1, device=physics.device))
    coupling = float(options.get("coupling", 1.0))
    relaxation = float(options.get("relaxation", 1e-4))
    
    # The framework automatically broadcasts scalar SEI growth to the spatial grid
    source = coupling * dlsei_dt.expand_as(stress) - relaxation * stress
    return source
```

#### Step 2: Register the PDE with Boundary Conditions

Use the `through_cell_diffusion_operator` to automatically generate the Jacobian-compatible RHS, including complex boundary conditions (like enforcing a specific force at the current collectors).

```python
from pde_framework import through_cell_diffusion_operator, BoundaryCondition

STATE_EQUATIONS = {
    # ... existing entries ...

    "stress": StateEquationSpec(
        order=450,                             # Defines position in the state vector
        size=physics_attr("Nel"),              # Same spatial resolution as the electrolyte
        initial=lambda physics, config: config.get("stress_options", {}).get("initial", 0.0),
        scale=lambda physics, config: config.get("stress_options", {}).get("scale", 1e6),
        nonnegative=False,
        
        # The operator automatically builds the PDE: dS/dt = ∇·(D ∇S) + Source
        operator=through_cell_diffusion_operator(
            "stress",
            parameters={
                "flux_coefficient": stress_flux_coefficient,
                "source": stress_source,
            },
            # Enforce Neumann (flux) BC at the negative current collector
            left_bc=BoundaryCondition(
                type="neumann", 
                value=lambda p, y, i, par, ctx: par["stress_options"].get("force_area", 0.0)
            ),
            # Enforce Dirichlet (fixed value) BC at the positive current collector
            right_bc=BoundaryCondition(type="dirichlet", value=0.0)
        ),
        
        # Toggleable via sim_config.py
        enabled=lambda config: config.get("stress_options", {}).get("enabled", False),
        options_key="stress_options",
        equation="∂σ/∂t = ∇·(D ∇σ) + k(dLsei/dt) - λσ",
        notes="Mechanical stress propagation model across the cell."
    ),
}
```

#### Step 3: Enable in Configuration

You can now toggle and tune this complex PDE directly from your configuration dictionary without touching the solver loop:

```python
config = {
    ...
    'stress_options': {
        'enabled': True,
        'diffusivity': 1e-12,
        'relaxation': 1e-4,
        'coupling': 1.0,
        'force_area': 0.1027,
        'scale': 1e6,           # Newton scaling hint
    }
}
```

---

### 6.8 Adding a New SEI Degradation Model

SEI models live in `sei_models.py`. All models follow the PyBaMM interface.

#### Step 1: Write the model function

```python
# In sei_models.py

def sei_my_model(Lsei, i_n, Un, T_cell, p, device):
    """
    My custom SEI model.

    Parameters
    ----------
    Lsei   : Tensor  — SEI thickness [m]
    i_n    : Tensor  — intercalation current density [A/m²]
    Un     : Tensor  — anode OCP [V]
    T_cell : Tensor  — temperature [K]
    p      : dict    — cell parameters
    device : torch.device

    Returns
    -------
    j_sei : Tensor [A/m²], negative (reduction reaction)
    """
    F   = p['F']
    R_g = p['R_g']
    # ... your physics ...
    j_sei = -1e-8 * torch.exp(-F * Un / (R_g * T_cell))
    return j_sei
```

#### Step 2: Register it

```python
# In sei_models.py
_REGISTRY = {
    ...
    "my_model": sei_my_model,
}
```

#### Step 3: Select in config

```python
config = {
    'sei_options': {
        'enabled': True,
        'sei_model': 'my_model',   # ← your new key
    }
}
```

---

### 6.9 Configuring Pack Topology and Balancing

#### Pack topology

```python
config = {
    'n_series':   4,       # cells in series
    'n_parallel': 2,       # cells in parallel
    'topology': 'series_first',  # or 'parallel_first'
}
```

- **`parallel_first`** (P-S): parallel groups form rows, then stacked in series
- **`series_first`** (S-P): series strings run top-to-bottom, parallel strings are columns

#### Cell overrides in a pack

Index `(s, p)` where `s` is the series index (0-based) and `p` is the parallel index:

```python
overrides = {
    (0, 0): {'T_amb': 308.15},   # cell at position row-0, col-0 is in a hotter environment
}
```

#### Balancing strategies

```python
balancing_options = {
    'enabled': True,
    'strategy': 'passive',    # 'none', 'passive', 'active_capacitor', 'active_inductor'
    'r_bleed': 5.0,           # bleed resistor [Ω] (passive only)
    'v_threshold': 4.1,       # balance above this voltage [V] (passive only)
}
```

| Strategy | Mechanism | Parameters |
|----------|-----------|------------|
| `none` | No balancing | — |
| `passive` | Resistive bleed on over-voltage cells | `r_bleed`, `v_threshold` |
| `active_capacitor` | Capacitor-based charge transfer | `r_eq` |
| `active_inductor` | Inductor-based equalisation | `transfer_gain` |

---

### 6.10 Selecting Outputs and Reading Results

#### OutputSpec

Control which quantities are computed and saved using `OutputSpec`:

```python
from pde_sim.output import OutputSpec

# Preset: common variables
output_spec = OutputSpec("default")

# Preset: everything
output_spec = OutputSpec("all")

# Custom selection
output_spec = OutputSpec([
    "terminal_voltage",
    "cell_current",
    "temperature",
    "soc",
    "sei_thickness_nm",
    "capacity_fade_pct",
    "pack_voltage",
])
```

Available output names:

| Name | Description |
|------|-------------|
| `terminal_voltage` | Per-cell terminal voltage [V] |
| `cell_current` | Per-cell current [A] |
| `pack_voltage` | Total pack voltage [V] |
| `temperature` | Per-cell temperature [K] |
| `soc` | Per-cell state of charge (0–1) |
| `sei_thickness_nm` | Per-cell SEI thickness [nm] |
| `capacity_fade_pct` | Capacity fade from SEI [%] |
| `dis_stress_vm_peak` | Peak von Mises stress in anode [Pa] |
| `dis_stress_th_surf` | Hoop stress at anode surface [Pa] |
| `stress_mean/min/max` | Stress field statistics [Pa] |

#### Reading results

```python
import numpy as np

data = np.load('simulation_result/my_run/results/simulation_results.npz')

times = data['times']            # [n_steps] seconds
V     = data['TermV']            # [n_steps, n_cells] terminal voltage
I     = data['Curr']             # [n_steps, n_cells] current
T     = data['Temp']             # [n_steps, n_cells] temperature
SOC   = data['SOC']              # [n_steps, n_cells] SOC
SEI   = data['SEI_Thick']        # [n_steps, n_cells] nm
fade  = data['CapFade']          # [n_steps, n_cells] %
```

---

## 7. Physics Model Reference

### Electrochemistry

The DFN model equations solved at each time step:

```
Solid diffusion (spherical):
  ∂cs/∂t = Ds/(Rs/N)² · A_sph · cs + flux_BC · B_sph

Electrolyte transport (through-cell):
  ∂(ε·ce)/∂t = -∇·(-Deff · ∇ce) + source

Butler-Volmer kinetics:
  j = 2·j0·sinh(F·η / (2RT))
  η  = V_cell - OCV - I·R_ohm - I·R_sei - V_conc

SEI growth:
  dLsei/dt = i_side · Msei / (2F·ρsei)

Thermal (lumped):
  dT/dt = [I·(OCV - V_cell) - hA·(T - T_amb)] / (ρCp·Vol)
```

### State vector layout

```
y = [ cs_n (Nr_n)  |  cs_p (Nr_p)  |  ce (Nx_n+Nx_s+Nx_p)  |  Lsei (1)  |  T (1) ]
```

If stress is enabled:
```
y = [ cs_n | cs_p | ce | stress (Nel) | Lsei | T ]
```

---

## 8. Configuration Reference

### Full config dict

```python
config = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    'device': 'auto',         # 'auto', 'cpu', 'cuda', 'cuda:0'

    # ── Pack geometry ─────────────────────────────────────────────────────────
    'n_series':   1,
    'n_parallel': 1,
    'topology':   'parallel_first',  # 'parallel_first' or 'series_first'

    # ── Spatial discretization ────────────────────────────────────────────────
    'electrolyte_spatial_method': 'finite_volume',   # 'finite_volume', 'chebyshev'
    'solid_spatial_method':       'chebyshev',        # currently chebyshev only

    # ── SEI degradation ───────────────────────────────────────────────────────
    'sei_options': {
        'enabled':   True,
        'sei_model': 'builtin',    # see sei_models.py for all options
    },

    # ── Mechanical stress PDE ─────────────────────────────────────────────────
    'stress_options': {
        'enabled':     False,
        'diffusivity': 1e-12,
        'relaxation':  1e-4,
        'coupling':    1.0,
        'scale':       1e6,
        'initial':     0.0,
    },

    # ── Thermal model ─────────────────────────────────────────────────────────
    'thermal_options': {
        'enabled':  True,
        'model':    'builtin',    # 'builtin', 'isothermal', 'entropic'
        'strategy': 'ambient',    # 'ambient', 'liquid', 'pcm'
        ...
    },

    # ── Extra state fields (advanced) ─────────────────────────────────────────
    'extra_state_fields': [],
}
```

### Full discretization dict

```python
discretization = {
    'Nr_n': 10,   # radial nodes in negative particle (≥ 3)
    'Nr_p': 10,   # radial nodes in positive particle (≥ 3)
    'Nx_n': 10,   # x-nodes in negative electrode (≥ 3)
    'Nx_s': 10,   # x-nodes in separator           (≥ 3)
    'Nx_p': 10,   # x-nodes in positive electrode  (≥ 3)
    'Nsei': 1,    # SEI state nodes (keep at 1)
}
```

### `simulate()` signature

```python
solver.simulate(
    t_end        = 3600.0,        # simulation end time [s]
    dt_init      = 1.0,           # initial time step [s]
    I_pack       = 5.0,           # constant current [A] (ignored if controller given)
    method       = 'basic',       # 'basic' or 'advanced' (ignored with controller)
    controller   = None,          # BaseController instance
    dt_max       = None,          # maximum time step [s]
    output_spec  = None,          # OutputSpec instance
    run_name     = None,          # results folder name (auto-detected from filename)
)
```

---

## 9. API Reference

### `ImplicitBatterySolver`

```python
solver = ImplicitBatterySolver(
    config,                          # dict — physics and pack settings
    discretization,                  # dict — spatial resolution
    overrides,                       # dict — per-cell parameter overrides
    initial_state_mode='fully_charged',  # 'fully_charged' or 'fully_discharged'
    initial_state_options={},        # options for discharge-to-empty reference run
    balancing_options={'enabled': False, 'strategy': 'none'},
    thermal_options={'enabled': False, 'strategy': 'ambient'},
    chemistry=None,                  # pre-loaded chemistry dict
    chemistry_folder=None,           # path string to chemistry folder
)
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solver.simulate(...)` | — | Run the simulation |
| `solver.get_exact_terminal_voltages(y, I)` | `Tensor [n_cells]` | Terminal voltages |
| `solver.get_pack_voltage(y, I)` | `float` | Pack-level voltage |
| `solver.solve_current_distribution(y, I)` | `Tensor [n_cells, 1]` | Per-cell currents |
| `solver.check_voltage_limits(y, I)` | `(bool, idx, V, kind)` | Voltage limit check |
| `ImplicitBatterySolver.get_standard_parameters(chemistry)` | `dict` | Default param dict |

### `parameter_loader`

```python
from parameter_loader import load_chemistry_from_folder, list_available_chemistries

chem = load_chemistry_from_folder('parameters/na_ion_chayambuka2022')
# Returns:
# {
#   'name':   'na_ion_chayambuka2022',
#   'params': { ... scalar and callable params ... },
#   'ocp_n':  callable(sto) -> V,
#   'ocp_p':  callable(sto) -> V,
#   'cond_e': callable(ce, T) -> S/m,
#   'diff_e': callable(ce, T) -> m²/s,
# }

list_available_chemistries()   # scans parameters/ directory
```

---

## 10. Extending VIBE

### Adding a new output variable

In `model_registry.py`, add to `DERIVED_OUTPUTS`:

```python
def _my_output(battery, y, i_pack):
    """Must return a flat tensor [n_cells]."""
    cs_n = battery.physics.state(y, 'cs_n')
    return torch.mean(cs_n, dim=1).flatten()

DERIVED_OUTPUTS = {
    ...
    "my_output": DerivedOutputSpec(_my_output, requires_state=("cs_n",)),
}
```

Then select it: `output_spec = OutputSpec(["my_output"])`.

### Adding a new thermal model

In `temperature_models.py`, add a function and register it:

```python
def thermal_my_model(I_app, OCV, V_cell, T_cell, p, device):
    Q_gen = I_app * (OCV - V_cell)
    Q_cool = p['hA'] * (T_cell - p['T_amb']) * my_correction
    return (Q_gen - Q_cool) / (p['rho_Cp'] * p['Vol_cell'])

_REGISTRY["my_model"] = thermal_my_model
```

Use: `config['thermal_options']['model'] = 'my_model'`.

### Adding a new sensitivity analysis

VIBE supports automatic differentiation via `torch.func.jacrev`. See `Experiment_AutogradSensitivity.py` for a complete example that computes `∂V/∂p` for every parameter simultaneously.

---

## Contributing

Pull requests and chemistry contributions (`parameters/` folders with `params.json` + CSVs) are welcome.

---

## Citation

If you use VIBE in your research, please cite:

```bibtex
@misc{vibe2025,
  title  = {VIBE: Variable-fidelity Implicit Battery Emulator},
  author = {Ankit Yadav},
  year   = {2025},
  url    = {https://github.com/Ankityadav1834/Vibe_Battery}
}
```

---

<div align="center">
Built with PyTorch · Physics by Doyle–Fuller–Newman · Made for battery researchers
</div>
