import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from initial_states import get_hardcoded_discharged_state
from states import enabled_state_equations, evaluate_registered_rhs
from output_manager import SimulationOutputManager
from pack_modules import build_balancing_module, build_thermal_module
from pde_framework import (
    ChebyshevRegionOperators,
    CompositeMesh,
    ConservativeFluxPDEModel,
    DiffusionSourcePDEModel,
    ElectrolytePDEModel,
    FiniteVolumeOperators,
    GeneralOperatorPDEModel,
    OperatorPDEPipeline,
    SphericalParticlePDEModel,
    StateLayout,
)
from solver import BasicSolver, AdvancedSolver, ControlledSolver

import os
from parameter_loader import load_chemistry_from_folder, list_available_chemistries

# Use Double Precision for Battery Physics Stability
torch.set_default_dtype(torch.float64)

# Default parameters/ directory (siblings to main.py)
_PARAMETERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters')

class BatteryPhysics(torch.nn.Module):
    def __init__(self, config, disc, params_list, device, chemistry=None):
        super().__init__()
        self.device = device
        self.n_cells = len(params_list)
        # Store the chemistry dict so electrochemistry.py can read OCP/diff_e etc.
        self.chemistry = chemistry
        
        # Discretization
        self.Nr_n, self.Nr_p = disc['Nr_n'], disc['Nr_p']
        self.Nx_n, self.Nx_s, self.Nx_p = disc['Nx_n'], disc['Nx_s'], disc['Nx_p']
        self.Nel = self.Nx_n + self.Nx_s + self.Nx_p
        self.Nsei = disc['Nsei']
        
        p0 = params_list[0]
        
        # 1. Generate Chebyshev Nodes and Differentiation Matrices [0, 1] mapped
        self.r_n_ref, self.Dr_n_ref, self.D2r_n_ref = self._build_cheb_operators(self.Nr_n)
        self.r_p_ref, self.Dr_p_ref, self.D2r_p_ref = self._build_cheb_operators(self.Nr_p)
        
        self.x_n_ref, self.Dx_n_ref, self.D2x_n_ref = self._build_cheb_operators(self.Nx_n)
        self.x_s_ref, self.Dx_s_ref, self.D2x_s_ref = self._build_cheb_operators(self.Nx_s)
        self.x_p_ref, self.Dx_p_ref, self.D2x_p_ref = self._build_cheb_operators(self.Nx_p)

        # 2. Clenshaw-Curtis Quadrature Weights (for accurate integration over domains)
        self.W_n = self._clenshaw_curtis_weights(self.Nx_n)
        self.W_s = self._clenshaw_curtis_weights(self.Nx_s)
        self.W_p = self._clenshaw_curtis_weights(self.Nx_p)
        
        # Porosity vector 
        self.eps_vec = torch.cat([
            torch.full((self.Nx_n,), p0['eps_e_n']),
            torch.full((self.Nx_s,), p0['eps_e_s']),
            torch.full((self.Nx_p,), p0['eps_e_p'])
        ]).to(device)

        self.discretization_methods = {
            'electrolyte': config.get('electrolyte_spatial_method', 'finite_volume'),
            'solid': config.get('solid_spatial_method', 'chebyshev')
        }
        self.electrolyte_mesh = self._build_electrolyte_mesh(p0)
        self.electrolyte_operators = self._build_electrolyte_operators()

        self.state_equation_specs = enabled_state_equations(self, config)
        self.state_equation_options = {
            name: spec.options(config)
            for name, spec in self.state_equation_specs.items()
        }
        self.operator_pipeline = self._build_operator_pipeline()
        self.electrolyte_model = (
            self.operator_pipeline.model('electrolyte')
            if 'electrolyte' in self.operator_pipeline
            else ElectrolytePDEModel(self.electrolyte_mesh, self.electrolyte_operators)
        )
        self.stress_options = self.state_equation_options.get('stress', {})
        self.stress_enabled = 'stress' in self.state_equation_specs
        self.sei_enabled = 'Lsei' in self.state_equation_specs
        # Physics model selection — read from config, fall back to 'builtin'
        self.sei_model_name     = config.get('sei_options', {}).get('sei_model', 'builtin')
        self.stress_model_name  = config.get('stress_options', {}).get('model', 'builtin')
        self.thermal_model_name = config.get('thermal_options', {}).get('model', 'builtin')
        self.plating_model_name = config.get('lithium_plating_options', {}).get('model', 'builtin')
        self.state_layout = self._build_state_layout(config)
        self.state_size = self.state_layout.total_size
        
        # --- Separate callable params from scalar/tensor params ---
        # physics.params must be a pure dict-of-tensors so torch.vmap and
        # torch.func.jacrev can batch over it.  CSV-loaded function-valued
        # parameters (Ds_n, Ds_p, m_ref_n, m_ref_p) are stored separately
        # in physics.callable_params as a list-of-dicts (one dict per cell).
        # electrochemistry.py reads callable_params[cell_idx][key] directly.
        CALLABLE_KEYS = {'Ds_n', 'Ds_p', 'm_ref_n', 'm_ref_p'}
        
        # Keys that go into the tensor params dict
        keys = [
            'Ln','Ls','Lp','A','Rs_n','Rs_p','F','R_g','T','Ds_n','Ds_p',
            'cs_max_n','cs_max_p','as_n','as_p','t_plus','b',
            'Dsolv','kappa_sei','Msei','rho_sei','Us','kf','beta',
            'm_ref_n', 'm_ref_p', 'sigma_n', 'sigma_p', 
            'E_r_n', 'E_r_p', # Activation Energies
            'Vol_cell', 'rho_Cp', 'hA', 'T_amb' , 'eps_e_n' , 'eps_e_s' , 'eps_e_p',
            'ce_0', 'Lsei_0', 'R_contact', 'R_bus'
        ]
        
        # Build callable_params: list of per-cell dicts with any callables
        self.callable_params = []
        for p in params_list:
            cell_fns = {}
            for k in CALLABLE_KEYS:
                v = p.get(k, None)
                if callable(v):
                    cell_fns[k] = v
            self.callable_params.append(cell_fns)

        # Build tensor params dict: use scalar fallback for callable keys
        # (the scalar is only used as a placeholder; actual computation in
        #  electrochemistry.py routes through callable_params when present)
        self.params = {}
        for k in keys:
            vals = []
            for p in params_list:
                v = p.get(k, 0.0)
                if callable(v):
                    vals.append(0.0)   # placeholder; callable_params is the source
                else:
                    vals.append(float(v))
            self.params[k] = torch.tensor(vals, device=device).reshape(-1, 1)

        # Convenience physics-level callables (shared across cells).
        # These are set from cell-0's callable_params; in practice all cells
        # share the same chemistry functions.  Per-cell overrides can be added
        # by modifying callable_params[i] entries directly.
        for k in ('Ds_n', 'Ds_p', 'm_ref_n', 'm_ref_p'):
            fn = self.callable_params[0].get(k, None) if self.callable_params else None
            setattr(self, f'fn_{k}', fn)   # e.g. physics.fn_Ds_n = <callable | None>


    def _build_state_layout(self, config):
        layout = StateLayout()
        temperature_spec = None
        for name, spec in self.state_equation_specs.items():
            if name == 'temperature':
                temperature_spec = spec
                continue
            layout.register(
                name,
                spec.resolved_size(self, config),
                initial=spec.resolved_initial(self, config),
                scale=spec.resolved_scale(self, config),
                nonnegative=spec.nonnegative,
            )

        for field in config.get('extra_state_fields', []):
            layout.register(
                field['name'],
                field['size'],
                initial=field.get('initial', 0.0),
                scale=field.get('scale', 1.0),
                nonnegative=field.get('nonnegative', False),
            )

        if temperature_spec is not None:
            layout.register(
                'temperature',
                temperature_spec.resolved_size(self, config),
                initial=temperature_spec.resolved_initial(self, config),
                scale=temperature_spec.resolved_scale(self, config),
                nonnegative=temperature_spec.nonnegative,
            )
        return layout

    def _build_operator_pipeline(self):
        pipeline = OperatorPDEPipeline()
        for name, state_spec in self.state_equation_specs.items():
            operator_spec = state_spec.operator
            if operator_spec is None:
                continue

            if operator_spec.evaluator in ('general', 'operator'):
                if operator_spec.domain != 'through_cell':
                    raise ValueError(
                        f"General operator PDE {name!r} uses unsupported domain "
                        f"{operator_spec.domain!r}. Supported domain: 'through_cell'."
                    )
                model = GeneralOperatorPDEModel(
                    self.electrolyte_mesh,
                    self.electrolyte_operators,
                    spec=operator_spec,
                )
            elif operator_spec.evaluator in ('conservative', 'flux_source'):
                if operator_spec.domain != 'through_cell':
                    raise ValueError(
                        f"Conservative PDE {name!r} uses unsupported domain "
                        f"{operator_spec.domain!r}. Supported domain: 'through_cell'."
                    )
                model = ConservativeFluxPDEModel(
                    self.electrolyte_mesh,
                    self.electrolyte_operators,
                    spec=operator_spec,
                )
            elif operator_spec.evaluator == 'diffusion_source':
                if operator_spec.domain != 'through_cell':
                    raise ValueError(
                        f"Diffusion-source PDE {name!r} uses unsupported domain "
                        f"{operator_spec.domain!r}. Supported domain: 'through_cell'."
                    )
                model = DiffusionSourcePDEModel(
                    self.electrolyte_mesh,
                    self.electrolyte_operators,
                    spec=operator_spec,
                )
            elif operator_spec.evaluator == 'spherical_particle':
                if operator_spec.domain == 'negative_particle':
                    model = SphericalParticlePDEModel(
                        self.r_n_ref,
                        self.Dr_n_ref,
                        self.D2r_n_ref,
                        spec=operator_spec,
                    )
                elif operator_spec.domain == 'positive_particle':
                    model = SphericalParticlePDEModel(
                        self.r_p_ref,
                        self.Dr_p_ref,
                        self.D2r_p_ref,
                        spec=operator_spec,
                    )
                else:
                    raise ValueError(
                        f"Spherical particle PDE {name!r} uses unsupported domain "
                        f"{operator_spec.domain!r}. Supported domains: "
                        "'negative_particle', 'positive_particle'."
                    )
            else:
                raise ValueError(
                    f"Operator PDE {name!r} has unsupported evaluator "
                    f"{operator_spec.evaluator!r}."
                )

            pipeline.register(name, model)
        return pipeline

    def state(self, y, name):
        return self.state_layout.get(y, name)

    def through_cell_flux_coefficient(self, diffusivity):
        coefficient = -diffusivity
        if self.discretization_methods['electrolyte'] == 'finite_volume':
            coefficient = lambda ctx, coeff=diffusivity: -ctx.operators.face_coefficients(coeff)
        return coefficient

    def evaluate_diffusion_source_rhs(self, state_values, diffusivity, source):
        return self.electrolyte_model.evaluate(
            values=state_values,
            parameters={
                'flux_coefficient': self.through_cell_flux_coefficient(diffusivity),
                'source': source,
            },
        )

    def _resolve_operator_item(self, resolver, y_flat, i_app, p, context, default=None):
        if resolver is None:
            return default
        if callable(resolver):
            return resolver(self, y_flat, i_app, p, context)
        if isinstance(resolver, str):
            if resolver in context:
                return context[resolver]
            if resolver in p:
                return p[resolver]
            if resolver in self.state_layout:
                return self.state(y_flat, resolver)
        return resolver

    def _resolve_operator_map(self, resolvers, y_flat, i_app, p, context):
        return {
            key: self._resolve_operator_item(value, y_flat, i_app, p, context)
            for key, value in resolvers.items()
        }

    def _resolve_operator_boundary_conditions(self, boundary_conditions, y_flat, i_app, p, context):
        resolved = {}
        for location, condition in boundary_conditions.items():
            if isinstance(condition, dict):
                kind = condition.get('kind', condition.get('type', 'neumann'))
                value = condition.get('value', 0.0)
            else:
                kind = getattr(condition, 'kind', 'neumann')
                value = getattr(condition, 'value', condition)
            value = self._resolve_operator_item(value, y_flat, i_app, p, context, default=value)
            resolved[location] = {'type': kind, 'value': value}
        return resolved

    def evaluate_operator_state(self, name, y_flat, i_app, p, context, derivatives=None, return_details=False):
        state_spec = self.state_equation_specs[name]
        operator_spec = state_spec.operator
        if operator_spec is None:
            raise ValueError(f"State {name!r} does not define an operator equation.")
        if name not in self.operator_pipeline:
            raise ValueError(f"State {name!r} is not registered in the operator pipeline.")

        default_values = self.state(y_flat, name)
        values = self._resolve_operator_item(
            operator_spec.values,
            y_flat,
            i_app,
            p,
            context,
            default=default_values,
        )
        time_values = self._resolve_operator_item(
            operator_spec.time_values,
            y_flat,
            i_app,
            p,
            context,
            default=default_values,
        )
        parameters = self._resolve_operator_map(operator_spec.parameters, y_flat, i_app, p, context)
        variables = self._resolve_operator_map(operator_spec.variables, y_flat, i_app, p, context)
        boundary_conditions = self._resolve_operator_boundary_conditions(
            operator_spec.boundary_conditions,
            y_flat,
            i_app,
            p,
            context,
        )

        y_old = context.get('y_old')
        old_values = self.state(y_old, name) if y_old is not None else None
        evaluation = self.operator_pipeline.evaluate_state(
            name,
            values=values,
            time_values=time_values,
            parameters=parameters,
            variables=variables,
            boundary_conditions=boundary_conditions,
            old_values=old_values,
            dt=context.get('dt'),
            return_details=return_details,
        )
        if derivatives is not None:
            derivatives[name] = evaluation.rhs if return_details else evaluation
        return evaluation

    def evaluate_registered_operator_rhs(self, y_flat, i_app, p, context, derivatives):
        for name, state_spec in self.state_equation_specs.items():
            operator_spec = state_spec.operator
            if operator_spec is None or name in derivatives:
                continue
            if name not in self.operator_pipeline:
                continue

            self.evaluate_operator_state(name, y_flat, i_app, p, context, derivatives=derivatives)
        return derivatives

    def _build_electrolyte_mesh(self, p0):
        region_spec = {
            'negative': {'length': p0['Ln'], 'N': self.Nx_n},
            'separator': {'length': p0['Ls'], 'N': self.Nx_s},
            'positive': {'length': p0['Lp'], 'N': self.Nx_p},
        }
        return CompositeMesh(
            region_spec,
            method=self.discretization_methods['electrolyte'],
            device=self.device,
            dtype=torch.float64
        )

    def _build_electrolyte_operators(self):
        method = self.discretization_methods['electrolyte']
        if method == 'chebyshev':
            return ChebyshevRegionOperators(self.electrolyte_mesh)
        if method == 'finite_volume':
            return FiniteVolumeOperators(self.electrolyte_mesh)
        if method == 'finite_difference':
            from pde_framework import FiniteDifferenceOperators
            return FiniteDifferenceOperators(self.electrolyte_mesh)
        raise ValueError("electrolyte_spatial_method must be 'chebyshev', 'finite_volume', or 'finite_difference'")

    def _build_cheb_operators(self, N):
        """Generates Chebyshev nodes on [0, 1] and differentiation matrices D and D2"""
        # Gauss-Lobatto nodes mapped to [0, 1]
        x = 0.5 * (1.0 - torch.cos(torch.pi * torch.arange(N, device=self.device, dtype=torch.float64) / (N - 1)))
        
        # Barycentric Differentiation Matrix
        D = torch.zeros((N, N), device=self.device, dtype=torch.float64)
        w = torch.ones(N, device=self.device, dtype=torch.float64)
        for i in range(N):
            diff = x[i] - x
            diff[i] = 1.0  # Avoid div by zero
            w[i] = 1.0 / torch.prod(diff)
            
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
                    
        # Diagonal (negative sum of row to preserve constant functions)
        for i in range(N):
            D[i, i] = -torch.sum(D[i, :]) + D[i, i]
            
        D2 = D @ D
        return x, D, D2

    def _clenshaw_curtis_weights(self, N):
        """Calculates Quadrature Weights for integration over Chebyshev nodes on [0,1]"""
        w = torch.zeros(N, device=self.device, dtype=torch.float64)
        for i in range(N):
            # Convert theta to a tensor to satisfy torch.cos()
            theta = torch.tensor(torch.pi * i / (N - 1), device=self.device, dtype=torch.float64)
            sum_val = 0.0
            for j in range(1, (N - 1) // 2 + 1):
                b = 2.0 / (1.0 - 4.0 * j**2)
                sum_val += b * torch.cos(2.0 * j * theta)
            w[i] = 1.0 + sum_val
            if (N - 1) % 2 == 0:
                w[i] += (1.0 / (1.0 - (N - 1)**2)) * torch.cos((N - 1) * theta)
        w[0] /= 2.0; w[-1] /= 2.0
        w *= 2.0 / (N - 1)
        w *= 0.5 # Scale weight from [-1, 1] to [0, 1] domain
        return w
    
    def calculate_ocp(self, stoich, electrode):
        s = torch.clamp(stoich, 0.001, 0.999)
        chem = getattr(self, 'chemistry', None)
        if chem is not None:
            if electrode == 'anode':
                return chem['ocp_n'](s)
            else:
                return chem['ocp_p'](s)
        # Fallback: Li-ion Chen2020 analytic OCP
        if electrode == 'anode':
            return (1.9793 * torch.exp(-39.3631*s) + 0.2482 - 
                    0.0909 * torch.tanh(29.8538*(s-0.1234)) - 
                    0.04478 * torch.tanh(14.9159*(s-0.2769)) - 
                    0.0205 * torch.tanh(30.4444*(s-0.6103)))
        else:
            return (-0.8090*s + 4.4875 - 
                    0.0428 * torch.tanh(18.5138*(s-0.5542)) - 
                    17.7326 * torch.tanh(15.7890*(s-0.3117)) + 
                    17.5842 * torch.tanh(15.9308*(s-0.3120)))

    def calculate_electrolyte_physics(self, ce_eps, p):
        ce_real = ce_eps / (self.eps_vec + 1e-12)
        c_k = ce_real / 1000.0
        local_kappa = (0.1297 * c_k**3 - 2.51 * c_k**1.5 + 3.329 * c_k)
        
        # Exact integral of (1/kappa) over domains using C-C Quadrature
        int_inv_kappa = (
            p['Ln'] * torch.sum(self.W_n / (local_kappa[:self.Nx_n] + 1e-12)) +
            p['Ls'] * torch.sum(self.W_s / (local_kappa[self.Nx_n:self.Nx_n+self.Nx_s] + 1e-12)) +
            p['Lp'] * torch.sum(self.W_p / (local_kappa[-self.Nx_p:] + 1e-12))
        )
        
        total_L = p['Ln'] + p['Ls'] + p['Lp']
        eff_kappa = total_L / (int_inv_kappa + 1e-12)
        
        R_eff_term = (
            p['Ln'] / (3 * p['eps_e_n'] ** p['b']) +
            p['Ls'] / (p['eps_e_s'] ** p['b']) +
            p['Lp'] / (3 * p['eps_e_p'] ** p['b'])
        )
        return (1.0 / (p['A'] * (eff_kappa + 1e-9))) * R_eff_term

    def calculate_conc_overpotential(self, ce_eps, T_cell, p):
        ce_real = ce_eps / (self.eps_vec + 1e-12)
        ln_ce = torch.log(torch.clamp(ce_real, min=1e-6))
        
        ln_ce_n = ln_ce[:self.Nx_n]
        ln_ce_p = ln_ce[-self.Nx_p:]
        
        # Chebyshev numerical integration
        avg_log_n = torch.sum(ln_ce_n * self.W_n) 
        avg_log_p = torch.sum(ln_ce_p * self.W_p)
        
        term_RTF = 2 * p['R_g'] * T_cell / p['F']
        V_conc = term_RTF * (1.0 - p['t_plus']) * (avg_log_n - avg_log_p)
        return V_conc

    def get_circuit_parameters(self, y_batch):
        import electrochemistry
        return electrochemistry.get_circuit_parameters(self, y_batch)
    
    def compute_derivatives_functional(self, y_flat, I_app, p, y_old=None, dt=None):
        import electrochemistry
        return electrochemistry.compute_derivatives_functional(self, y_flat, I_app, p, y_old, dt)

    def batched_derivatives(self, y_batch, I_batch, y_old_batch=None, dt=None):
        if y_old_batch is not None:
            return torch.vmap(self.compute_derivatives_functional, in_dims=(0, 0, 0, 0, None))(y_batch, I_batch.squeeze(-1), self.params, y_old_batch, dt)
        else:
            return torch.vmap(self.compute_derivatives_functional, in_dims=(0, 0, 0, None, None))(y_batch, I_batch.squeeze(-1), self.params, None, None)

class ImplicitBatterySolver:
    min_cell_voltage = 2.5
    max_cell_voltage = 4.2
    _discharged_state_cache = {}

    def __init__(
        self,
        config,
        discretization,
        overrides,
        initial_state_mode='fully_charged',
        initial_state_options=None,
        balancing_options=None,
        thermal_options=None,
        chemistry=None,
        chemistry_folder=None,
    ):
        """
        Parameters
        ----------
        chemistry : dict, optional
            A chemistry dict as returned by ``load_chemistry_from_folder()``.
            If provided, it overrides the default Li-ion parameters.
        chemistry_folder : str, optional
            Path to a chemistry parameter folder (must contain ``params.json``).
            The loader reads scalars from ``params.json`` and builds
            interpolating functions from any CSV files present.
            If both *chemistry* and *chemistry_folder* are given, *chemistry*
            takes precedence.
        """
        _dev = config.get('device', 'auto')
        if _dev == 'auto' or _dev is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(_dev)   # 'cuda', 'cpu', 'cuda:0', etc.
        self.config, self.disc = config, discretization
        self.n_cells = config['n_series'] * config['n_parallel']
        self.n_series, self.n_parallel = config['n_series'], config['n_parallel']
        self.initial_state_mode = initial_state_mode
        self.initial_state_options = initial_state_options or {}
        self.balancing_options = balancing_options or {'enabled': False, 'strategy': 'none'}
        self.thermal_options = thermal_options or {'enabled': False, 'strategy': 'ambient'}
        
        # ---- Resolve chemistry -----------------------------------------
        if chemistry is None and chemistry_folder is not None:
            chemistry = load_chemistry_from_folder(chemistry_folder)

        self._chemistry = chemistry  # may be None (legacy / default Li-ion)

        # ---- Build per-cell parameter list ---------------------------------
        self.raw_params = [self.get_standard_parameters(chemistry) for _ in range(self.n_cells)]
        if overrides:
            for k, v in overrides.items():
                self.raw_params[k[0]*config['n_parallel'] + k[1]].update(v)
                
        self.physics = BatteryPhysics(config, discretization, self.raw_params, self.device,
                                      chemistry=self._chemistry)
        self.sei_enabled = self.physics.sei_enabled
        self.Gth = self._build_thermal_connection_matrix().to(self.device)
        self.Cth = torch.tensor([p['rho_Cp']*p['Vol_cell'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.h_amb = torch.tensor([p['hA'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.T_amb_vec = torch.tensor([p['T_amb'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.balancing_module = build_balancing_module(**self.balancing_options)
        self.thermal_module = build_thermal_module(**self.thermal_options)

        self.scale = self.physics.state_layout.scale_vector(self.device, torch.float64)
        self.nonnegative_mask = self.physics.state_layout.nonnegative_mask(self.device)

        self.basic_solver = BasicSolver(self)
        self.advanced_solver = AdvancedSolver(self)
        self.controlled_solver = ControlledSolver(self)
        self.output_manager = SimulationOutputManager(self)
        self.y = self._build_initial_state(discretization, self.raw_params).to(self.device)

    def _create_fully_charged_state(self, disc, params):
        return self.physics.state_layout.initial_state(params, self.device, torch.float64)

    def _build_initial_state(self, disc, params):
        mode = self.initial_state_mode.lower()
        if mode in ('fully_charged', 'charged', 'full'):
            return self._create_fully_charged_state(disc, params)
        if mode in ('fully_discharged', 'discharged', 'empty'):
            return self._create_fully_discharged_state(disc, params)
        raise ValueError("initial_state_mode must be 'fully_charged' or 'fully_discharged'")

    def _create_fully_discharged_state(self, disc, params):
        hardcoded_state = get_hardcoded_discharged_state(disc)
        if hardcoded_state is not None:
            print("Using saved fully discharged initial state preset.")
            if hardcoded_state.numel() == self.physics.state_size:
                return hardcoded_state.unsqueeze(0).repeat(self.n_cells, 1)

            return self._merge_base_state_with_current_layout(hardcoded_state, disc, params)

        reference_state = self._get_discharged_reference_state(
            disc,
            cutoff_voltage=self.initial_state_options.get('cutoff_voltage', self.min_cell_voltage),
            discharge_current=self.initial_state_options.get('discharge_current', 10.0),
            dt=self.initial_state_options.get('dt', 1.0),
            max_time=self.initial_state_options.get('max_time', 5000.0),
            coarse_dt=self.initial_state_options.get('coarse_dt', 10.0),
            refine_margin=self.initial_state_options.get('refine_margin', 0.08)
        )
        if reference_state.numel() == self.physics.state_size:
            return reference_state.unsqueeze(0).repeat(self.n_cells, 1)
        return self._merge_base_state_with_current_layout(reference_state, disc, params)

    def _merge_base_state_with_current_layout(self, base_state, disc, params):
        y = self._create_fully_charged_state(disc, params)
        old_slices = {}
        cursor = 0
        for name, size in [
            ('cs_n', disc['Nr_n']),
            ('cs_p', disc['Nr_p']),
            ('electrolyte', self.physics.Nel),
        ]:
            old_slices[name] = slice(cursor, cursor + size)
            cursor += size

        if self.sei_enabled:
            for name, size in [('sei', disc['Nsei']), ('Lsei', 1)]:
                old_slices[name] = slice(cursor, cursor + size)
                cursor += size

        old_slices['temperature'] = slice(cursor, cursor + 1)
        cursor += 1

        for name, old_slice in old_slices.items():
            if name in self.physics.state_layout:
                y[:, self.physics.state_layout.slice(name)] = base_state[old_slice].unsqueeze(0)
        return y

    @classmethod
    def _get_discharged_reference_state(
        cls,
        disc,
        cutoff_voltage=2.5,
        discharge_current=10.0,
        dt=1.0,
        max_time=5000.0,
        coarse_dt=10.0,
        refine_margin=0.08
    ):
        cache_key = (
            disc['Nr_n'],
            disc['Nr_p'],
            disc['Nx_n'],
            disc['Nx_s'],
            disc['Nx_p'],
            disc['Nsei'],
            float(cutoff_voltage),
            float(discharge_current),
            float(dt),
            float(max_time),
            float(coarse_dt),
            float(refine_margin)
        )
        if cache_key in cls._discharged_state_cache:
            return cls._discharged_state_cache[cache_key].clone()

        print(
            "Generating fully discharged initial state from a 1-cell reference discharge "
            f"to {cutoff_voltage:.2f} V at {discharge_current:.2f} A with dt={dt:.2f}s."
        )

        reference_solver = cls(
            config={'n_series': 1, 'n_parallel': 1},
            discretization=disc,
            overrides={},
            initial_state_mode='fully_charged',
            balancing_options={'enabled': False, 'strategy': 'none'},
            thermal_options={'enabled': False, 'strategy': 'ambient'}
        )

        t = 0.0
        coarse_dt = max(float(dt), float(coarse_dt))
        refine_threshold = cutoff_voltage + float(refine_margin)
        reference_state_before_refine = reference_solver.y.clone()
        time_before_refine = 0.0
        while t < max_time:
            step_dt = coarse_dt if coarse_dt > dt else dt
            min_voltage_before = torch.min(
                reference_solver.get_exact_terminal_voltages(reference_solver.y, discharge_current)
            ).item()

            if coarse_dt > dt and min_voltage_before <= refine_threshold:
                reference_solver.y = reference_state_before_refine.clone()
                t = time_before_refine
                step_dt = dt
                coarse_dt = dt

            i_cells = reference_solver.solve_current_distribution(reference_solver.y, discharge_current)
            y_new, success = reference_solver.basic_solver.newton_step(reference_solver.y, step_dt, i_cells)
            if not success:
                raise RuntimeError(
                    "Failed to generate the fully discharged initial state because the "
                    f"reference Newton solve did not converge at t={t:.1f}s."
                )

            reference_state_before_refine = reference_solver.y.clone()
            time_before_refine = t
            reference_solver.y = y_new
            t += step_dt
            min_voltage = torch.min(
                reference_solver.get_exact_terminal_voltages(reference_solver.y, discharge_current)
            ).item()
            if min_voltage <= cutoff_voltage:
                state = reference_solver.y[0].detach().cpu().clone()
                cls._discharged_state_cache[cache_key] = state
                print(
                    f"Captured discharged reference state at t={t:.1f}s with "
                    f"terminal voltage {min_voltage:.4f} V."
                )
                return state.clone()

        raise RuntimeError(
            "Failed to generate the fully discharged initial state because the "
            f"reference cell did not reach {cutoff_voltage:.2f} V within {max_time:.1f}s."
        )

    def solve_current_distribution(self, y, I_pack_val):
        OCV, R, I0n, I0p, T, Vc = [
            x.view(self.n_series, self.n_parallel)
            for x in self.physics.get_circuit_parameters(y)
        ]
        
        topology = self.config.get('topology', 'parallel_first')
        RTF = 8.314 * T / 96485.33
        
        if topology == 'parallel_first':
            I_grid = torch.full_like(OCV, I_pack_val / self.n_parallel)
            for _ in range(5):
                Rd_n = (2 * RTF) / torch.sqrt(I_grid**2 + (2 * I0n)**2 + 1e-12)
                Rd_p = (2 * RTF) / torch.sqrt(I_grid**2 + (2 * I0p)**2 + 1e-12)
                R_total = R + Rd_n + Rd_p
                V_curr = (
                    OCV
                    - I_grid * R
                    - 2 * RTF * torch.asinh(I_grid / (2 * I0n + 1e-12))
                    - 2 * RTF * torch.asinh(I_grid / (2 * I0p + 1e-12))
                    - Vc
                )
                V_virtual = V_curr + I_grid * R_total
                G = 1.0 / R_total
                V_term = (torch.sum(V_virtual * G, 1, True) - I_pack_val) / torch.sum(G, 1, True)
                I_grid = (V_virtual - V_term) * G
            return I_grid.view(self.n_cells, 1)
            
        elif topology == 'series_first':
            I_string = torch.full((1, self.n_parallel), I_pack_val / self.n_parallel, device=OCV.device, dtype=OCV.dtype)
            for _ in range(5):
                I_grid = I_string.expand(self.n_series, self.n_parallel)
                Rd_n = (2 * RTF) / torch.sqrt(I_grid**2 + (2 * I0n)**2 + 1e-12)
                Rd_p = (2 * RTF) / torch.sqrt(I_grid**2 + (2 * I0p)**2 + 1e-12)
                R_total = R + Rd_n + Rd_p
                V_curr = (
                    OCV
                    - I_grid * R
                    - 2 * RTF * torch.asinh(I_grid / (2 * I0n + 1e-12))
                    - 2 * RTF * torch.asinh(I_grid / (2 * I0p + 1e-12))
                    - Vc
                )
                V_virtual = V_curr + I_grid * R_total
                
                R_string = torch.sum(R_total, 0, True)
                V_string_virt = torch.sum(V_virtual, 0, True)
                
                G_string = 1.0 / R_string
                V_pack = (torch.sum(V_string_virt * G_string, 1, True) - I_pack_val) / torch.sum(G_string, 1, True)
                
                I_string = (V_string_virt - V_pack) * G_string
            
            I_grid = I_string.expand(self.n_series, self.n_parallel)
            return I_grid.reshape(self.n_cells, 1)
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def compute_balancing_currents(self, y, I_pack_val):
        cell_voltages = self.get_exact_terminal_voltages(y, I_pack_val).view(self.n_series, self.n_parallel)
        string_currents = torch.full_like(cell_voltages, I_pack_val / self.n_parallel)
        balancing_grid = self.balancing_module.compute_balancing_currents(cell_voltages, string_currents)
        return balancing_grid.view(self.n_cells, 1)

    def compute_effective_cell_currents(self, y, I_pack_val):
        base_currents = self.solve_current_distribution(y, I_pack_val)
        balancing_currents = self.compute_balancing_currents(y, I_pack_val)
        return base_currents + balancing_currents

    def compute_heat_generation(self, y_batch, cell_currents):
        with torch.no_grad():
            OCV, R_series, I0_n, I0_p, T_cell, V_conc = self.physics.get_circuit_parameters(y_batch)
            term_RTF = 2 * self.physics.params['R_g'] * T_cell / self.physics.params['F']
            denom_n = 2 * I0_n * self.physics.params['A'] * self.physics.params['Ln'] * self.physics.params['as_n']
            denom_p = 2 * I0_p * self.physics.params['A'] * self.physics.params['Lp'] * self.physics.params['as_p']
            V_rxn = term_RTF * (torch.asinh(cell_currents / denom_n) - torch.asinh(-cell_currents / denom_p))
            V_cell = OCV - (cell_currents * R_series) - V_rxn - V_conc
            return cell_currents * (OCV - V_cell)

    def compute_thermal_rhs(self, y, cell_currents):
        temperatures = y[:, -1:]
        heat_generation = self.compute_heat_generation(y, cell_currents)
        return self.thermal_module.compute_temperature_rhs(self, temperatures, heat_generation)

    def compute_cell_to_cell_conduction(self, y):
        """
        Returns only the Gth inter-cell conduction term [K/s], without
        heat generation or cooling strategy. Used when temperature_models.py
        already handles per-cell Q_gen and Q_cool, so we only need to add
        the pack-level spatial coupling between neighboring cells.

            dT_cond/dt = (Gth @ T - diag(Gth)*T) / Cth
        """
        temperatures = y[:, -1:]
        q_cond = self.Gth @ temperatures - torch.sum(self.Gth, dim=1, keepdim=True) * temperatures
        return q_cond / (self.Cth + 1e-12)


    def get_exact_terminal_voltages(self, y_batch, I_pack_val):
        with torch.no_grad():
            if torch.is_tensor(I_pack_val):
                I_cells = I_pack_val.view(self.n_cells, 1).to(self.device)
            else:
                I_cells = self.solve_current_distribution(y_batch, I_pack_val)
            OCV, R_series, I0_n, I0_p, T_cell, V_conc = self.physics.get_circuit_parameters(y_batch)

            term_RTF = 2 * self.physics.params['R_g'] * T_cell / self.physics.params['F']
            denom_n = 2 * I0_n * self.physics.params['A'] * self.physics.params['Ln'] * self.physics.params['as_n']
            denom_p = 2 * I0_p * self.physics.params['A'] * self.physics.params['Lp'] * self.physics.params['as_p']
            V_rxn = term_RTF * (torch.asinh(I_cells / denom_n) - torch.asinh(-I_cells / denom_p))
            V_terminal = OCV - (I_cells * R_series) - V_rxn - V_conc
            return V_terminal.flatten()

    def get_pack_voltage(self, y, I_pack_val):
        if torch.is_tensor(I_pack_val):
            current_input = I_pack_val
        else:
            current_input = self.compute_effective_cell_currents(y, I_pack_val)
        cell_voltages = self.get_exact_terminal_voltages(y, current_input)
        voltage_grid = cell_voltages.view(self.n_series, self.n_parallel)
        
        topology = self.config.get('topology', 'parallel_first')
        if topology == 'parallel_first':
            return torch.sum(torch.mean(voltage_grid, dim=1)).item()
        elif topology == 'series_first':
            return torch.mean(torch.sum(voltage_grid, dim=0)).item()
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def check_voltage_limits(self, y, I_pack_val, min_voltage=None, max_voltage=None):
        min_voltage = self.min_cell_voltage if min_voltage is None else min_voltage
        max_voltage = self.max_cell_voltage if max_voltage is None else max_voltage
        if torch.is_tensor(I_pack_val):
            current_input = I_pack_val
        else:
            current_input = self.compute_effective_cell_currents(y, I_pack_val)
        cell_voltages = self.get_exact_terminal_voltages(y, current_input)

        max_val, max_idx = torch.max(cell_voltages, dim=0)
        min_val, min_idx = torch.min(cell_voltages, dim=0)

        if max_val.item() >= max_voltage:
            return True, int(max_idx.item()), float(max_val.item()), "max"
        if min_val.item() <= min_voltage:
            return True, int(min_idx.item()), float(min_val.item()), "min"
        return False, None, None, None

    def simulate(self, t_end, dt_init, I_pack=None, method='basic', controller=None,
                 dt_max=None, output_spec=None, run_name=None):
        import sys
        import os
        from pde_sim.output import OutputSpec

        if not run_name:
            run_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        if not run_name or run_name == '-c':
            run_name = 'default_run'

        project_root = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(project_root, 'simulation_result', run_name, 'results')
        os.makedirs(out_dir, exist_ok=True)

        # Use caller's spec or fall back to the default
        spec = output_spec or OutputSpec("default")

        old_cwd = os.getcwd()
        # Pin the project root so local modules (electrochemistry, etc.)
        # remain importable after os.chdir moves into the output directory.
        import sys as _sys
        _project_root = project_root
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        os.chdir(out_dir)
        try:
            if controller is not None:
                self.controlled_solver.simulate(
                    t_end, dt_init, controller,
                    dt_max=dt_max, out_dir=out_dir, output_spec=spec,
                )
                return

            if method == 'basic':
                self.basic_solver.simulate(t_end, dt_init, I_pack,
                                           out_dir=out_dir, output_spec=spec)
            elif method == 'advanced':
                self.advanced_solver.simulate(t_end, dt_init, I_pack,
                                              out_dir=out_dir, output_spec=spec)
            else:
                raise ValueError("Method must be 'basic' or 'advanced'")
        finally:
            os.chdir(old_cwd)

    def _build_thermal_connection_matrix(self, k_contact=0.5, area=0.005, dist=0.02):
        G = torch.zeros((self.n_cells, self.n_cells))
        G_val = k_contact * area / dist
        for s in range(self.n_series):
            for p in range(self.n_parallel):
                idx = s * self.n_parallel + p
                if p < self.n_parallel-1: G[idx, idx+1] = G[idx+1, idx] = G_val
                if s < self.n_series-1: G[idx, idx+self.n_parallel] = G[idx+self.n_parallel, idx] = G_val
        return G

    @staticmethod
    def get_standard_parameters(chemistry=None):
        """
        Return the default parameter dict for a single cell.

        Parameters
        ----------
        chemistry : dict, optional
            A chemistry dict loaded via ``load_chemistry_from_folder()``.  If
            supplied, its ``params`` sub-dict is used as the starting point,
            which means CSV-based function parameters (Ds_n, Ds_p, m_ref_n,
            m_ref_p) are automatically included.
            If *None*, the original hardcoded Li-ion Chen2020 values are used
            so existing scripts need no changes.
        """
        if chemistry is not None:
            # Start from the loaded chemistry's scalar / callable parameters
            base = dict(chemistry['params'])
            # Ensure every key expected by BatteryPhysics is present
            defaults = {
                'Dsolv': 0.0, 'kf': 0.0, 'beta': 0.0,
                'R_contact': 1e-2, 'R_bus': 1e-4,
            }
            for k, v in defaults.items():
                base.setdefault(k, v)
            return base

        # --- Legacy Li-ion Chen2020 hardcoded default ---
        return {
            'Ln': 8.52e-05, 'Ls': 1.20e-05, 'Lp': 7.56e-05, 'A': 0.1027, 'Rs_n': 5.86e-06, 'Rs_p': 5.22e-06,
            'F': 96485.33, 'R_g': 8.314, 'Ds_n': 3.3e-14, 'Ds_p': 4e-15, 'cs_max_n': 33133.0, 'cs_max_p': 63104.0,
            'as_n': 3.83e5, 'as_p': 3.82e5, 'sigma_n': 215.0, 'sigma_p': 0.18, 'eps_e_n': 0.25, 'eps_e_s': 0.47, 'eps_e_p': 0.335,
            't_plus': 0.2594, 'ce_0': 1000.0, 'b': 1.5, 'm_ref_n': 6.48e-7, 'E_r_n': 35000, 'm_ref_p': 3.42e-6, 'E_r_p': 17800,
            'kappa_sei': 1/200000.0, 'Msei': 0.162, 'rho_sei': 1690.0, 'Us': 0.4, 'Lsei_0': 5e-9,
            'Vol_cell': 2.42e-5, 'rho_Cp': 1.7676e6, 'hA': 0.0531, 'T_amb': 298.15,
            # Resistance values matched to liionpack defaults (Tranter et al., 2022)
            # Rc = 1e-2 Ω  (tab-to-busbar weld / interconnection)
            # Rb = 1e-4 Ω  (busbar segment per cell)
            # Rt = 1e-5 Ω  (terminal lug) — folded into R_bus for lumped model
            'R_contact': 1e-2,   # Ω  — weld/contact resistance per cell
            'R_bus':     1e-4,   # Ω  — busbar + terminal lumped per cell
        }
