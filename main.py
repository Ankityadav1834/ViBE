import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from initial_states import get_hardcoded_discharged_state
from pack_modules import build_balancing_module, build_thermal_module
from pde_framework import CompositeMesh, ChebyshevRegionOperators, ElectrolytePDEModel, FiniteVolumeOperators, StateLayout
from solver import BasicSolver, AdvancedSolver, ControlledSolver

# Use Double Precision for Battery Physics Stability
torch.set_default_dtype(torch.float64)

class BatteryPhysics(torch.nn.Module):
    def __init__(self, config, disc, params_list, device):
        super().__init__()
        self.device = device
        self.n_cells = len(params_list)
        
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
        self.electrolyte_model = ElectrolytePDEModel(self.electrolyte_mesh, self.electrolyte_operators)
        self.state_layout = self._build_state_layout(config)
        self.state_size = self.state_layout.total_size
        
        # Pack parameters into a dictionary of tensors [N_cells, 1]
        keys = [
            'Ln','Ls','Lp','A','Rs_n','Rs_p','F','R_g','T','Ds_n','Ds_p',
            'cs_max_n','cs_max_p','as_n','as_p','t_plus','b',
            'Dsolv','kappa_sei','Msei','rho_sei','Us','kf','beta',
            'm_ref_n', 'm_ref_p', 'sigma_n', 'sigma_p', 
            'E_r_n', 'E_r_p', # Activation Energies
            'Vol_cell', 'rho_Cp', 'hA', 'T_amb' , 'eps_e_n' , 'eps_e_s' , 'eps_e_p' 
        ]
        
        self.params = {}
        for k in keys:
            vals = [p.get(k, 0.0) for p in params_list]
            self.params[k] = torch.tensor(vals, device=device).reshape(-1, 1)

    def _build_state_layout(self, config):
        layout = StateLayout()
        layout.register('cs_n', self.Nr_n, initial=29866.0, scale=30000.0)
        layout.register('cs_p', self.Nr_p, initial=17038.0, scale=30000.0)
        layout.register(
            'electrolyte',
            self.Nel,
            initial=lambda p, device, dtype: self.eps_vec.to(device=device, dtype=dtype) * p['ce_0'],
            scale=1000.0,
        )

        for field in config.get('extra_state_fields', []):
            layout.register(
                field['name'],
                field['size'],
                initial=field.get('initial', 0.0),
                scale=field.get('scale', 1.0),
            )

        layout.register('sei', self.Nsei, initial=0.0, scale=1.0)
        layout.register('Lsei', 1, initial='Lsei_0', scale=1e-8)
        layout.register('temperature', 1, initial='T_amb', scale=300.0)
        return layout

    def state(self, y, name):
        return self.state_layout.get(y, name)

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
        raise ValueError("electrolyte_spatial_method must be 'chebyshev' or 'finite_volume'")

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
        p = self.params
        cs_n = self.state(y_batch, 'cs_n')
        cs_p = self.state(y_batch, 'cs_p')
        ce_eps = self.state(y_batch, 'electrolyte')
        Lsei = self.state(y_batch, 'Lsei')
        T_cell = self.state(y_batch, 'temperature')
        
        theta_n = cs_n[:, -1:] / p['cs_max_n']
        theta_p = cs_p[:, -1:] / p['cs_max_p']
        Un = self.calculate_ocp(theta_n, 'anode')
        Up = self.calculate_ocp(theta_p, 'cathode')
        OCV = Up - Un
        
        V_conc_list = []
        R_elec_list = []
        for i in range(y_batch.shape[0]):
             p_i = {k: v[i] for k, v in p.items()}
             V_conc_list.append(self.calculate_conc_overpotential(ce_eps[i], T_cell[i], p_i))
             R_elec_list.append(self.calculate_electrolyte_physics(ce_eps[i], p_i))
             
        V_conc = torch.stack(V_conc_list).reshape(-1, 1)
        R_electrolyte = torch.stack(R_elec_list).reshape(-1, 1)

        inv_T = 1.0 / T_cell
        inv_T_ref = 1.0 / 298.15
        arr_n = torch.exp(p['E_r_n'] / p['R_g'] * (inv_T_ref - inv_T))
        arr_p = torch.exp(p['E_r_p'] / p['R_g'] * (inv_T_ref - inv_T))
        
        ce_real_batch = ce_eps / (self.eps_vec.unsqueeze(0) + 1e-12)
        
        W_n_batch = self.W_n.unsqueeze(0)
        W_p_batch = self.W_p.unsqueeze(0)
        
        sqrt_ce_n = torch.sum(torch.sqrt(torch.clamp(ce_real_batch[:, :self.Nx_n], min=1e-12)) * W_n_batch, dim=1, keepdim=True)
        sqrt_ce_p = torch.sum(torch.sqrt(torch.clamp(ce_real_batch[:, -self.Nx_p:], min=1e-12)) * W_p_batch, dim=1, keepdim=True)

        j0_n = p['m_ref_n'] * arr_n * sqrt_ce_n * torch.sqrt(torch.clamp(cs_n[:, -1:] * (p['cs_max_n'] - cs_n[:, -1:]), min=1e-12))
        j0_p = p['m_ref_p'] * arr_p * sqrt_ce_p * torch.sqrt(torch.clamp(cs_p[:, -1:] * (p['cs_max_p'] - cs_p[:, -1:]), min=1e-12))
        
        I0_n = j0_n * p['A'] * p['Ln'] * p['as_n']
        I0_p = j0_p * p['A'] * p['Lp'] * p['as_p']

        R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3.0 * p['A'])
        R_sei = Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
        R_series = R_solid_ohm + R_electrolyte + R_sei 
        
        return OCV, R_series, I0_n, I0_p, T_cell, V_conc
    
    def compute_derivatives_functional(self, y_flat, I_app, p, y_old=None, dt=None):
        cs_n = self.state(y_flat, 'cs_n')
        cs_p = self.state(y_flat, 'cs_p')
        ce_eps = self.state(y_flat, 'electrolyte')
        Lsei = self.state(y_flat, 'Lsei')
        T_cell = self.state(y_flat, 'temperature')
        
        inv_T = 1.0 / T_cell
        inv_T_ref = 1.0 / 298.15
        arr_n = torch.exp(p['E_r_n'] / p['R_g'] * (inv_T_ref - inv_T))
        arr_p = torch.exp(p['E_r_p'] / p['R_g'] * (inv_T_ref - inv_T))
        
        theta_n = cs_n[-1:] / p['cs_max_n']
        theta_p = cs_p[-1:] / p['cs_max_p']
        Un = self.calculate_ocp(theta_n, 'anode')
        Up = self.calculate_ocp(theta_p, 'cathode')
        OCV = Up - Un
        
        R_electrolyte = self.calculate_electrolyte_physics(ce_eps, p)
        V_conc = self.calculate_conc_overpotential(ce_eps, T_cell, p)
        
        R_sei = Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
        R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3.0 * p['A'])
        V_ohm_total = I_app * (R_solid_ohm + R_electrolyte + R_sei)
        
        ce_real = ce_eps / (self.eps_vec + 1e-12)
        sqrt_ce_n_avg = torch.sum(torch.sqrt(torch.clamp(ce_real[:self.Nx_n], min=1e-12)) * self.W_n)
        sqrt_ce_p_avg = torch.sum(torch.sqrt(torch.clamp(ce_real[-self.Nx_p:], min=1e-12)) * self.W_p)
        
        j0_n = p['m_ref_n'] * arr_n * sqrt_ce_n_avg * torch.sqrt(torch.clamp(cs_n[-1] * (p['cs_max_n'] - cs_n[-1]), min=1e-12))
        j0_p = p['m_ref_p'] * arr_p * sqrt_ce_p_avg * torch.sqrt(torch.clamp(cs_p[-1] * (p['cs_max_p'] - cs_p[-1]), min=1e-12))
        
        term_RTF = 2 * p['R_g'] * T_cell / p['F']
        phi_s_n = Un - I_app * R_sei
        i_side = 165.96e-6*torch.exp(-6.3e9*Lsei)*torch.exp(-0.55*p['F']*(phi_s_n-p['Us'])/(p['R_g']*T_cell))  + p['F']*(3.7398e-15/Lsei)*0.015*torch.exp(-Un*p['F']/(p['R_g']*T_cell))
        g_side = p['as_n']*p['Ln']*p['A'] * i_side
        dLsei_dt = i_side * p['Msei'] / (2*p['F']*p['rho_sei'])
        
        i_n = (I_app) / (p['as_n'] * p['A'] * p['Ln'])
        i_p = -I_app / (p['as_p'] * p['A'] * p['Lp'])
        eta_n = term_RTF * torch.asinh(i_n / (2*j0_n + 1e-12))
        eta_p = term_RTF * torch.asinh(i_p / (2*j0_p + 1e-12))
        V_rxn = eta_n - eta_p
        
        V_cell = OCV - V_rxn - V_ohm_total - V_conc
        Q_gen = I_app * (OCV - V_cell)
        dT_dt = Q_gen / (p['rho_Cp'] * p['Vol_cell'])

        # ==========================================
        # 1. Solid Concentration PDE (Chebyshev)
        # ==========================================
        Dr_n = self.Dr_n_ref / p['Rs_n']
        D2r_n = self.D2r_n_ref / (p['Rs_n']**2)
        r_n = self.r_n_ref * p['Rs_n']
        
        r_safe_n = r_n.clone(); r_safe_n[0] = 1.0 # Protect div zero, index 0 is L'hopital
        t1_n = torch.mv(D2r_n, cs_n)
        t2_n = (2.0 / r_safe_n) * torch.mv(Dr_n, cs_n)
        dcs_n = p['Ds_n'] * (t1_n + t2_n)
        dcs_n[0] = 3.0 * p['Ds_n'] * t1_n[0] # L'hopital exactly applies here
        
        Dr_p = self.Dr_p_ref / p['Rs_p']
        D2r_p = self.D2r_p_ref / (p['Rs_p']**2)
        r_p = self.r_p_ref * p['Rs_p']
        
        r_safe_p = r_p.clone(); r_safe_p[0] = 1.0
        t1_p = torch.mv(D2r_p, cs_p)
        t2_p = (2.0 / r_safe_p) * torch.mv(Dr_p, cs_p)
        dcs_p = p['Ds_p'] * (t1_p + t2_p)
        dcs_p[0] = 3.0 * p['Ds_p'] * t1_p[0]

        # Flux Constraints
        flux_n = -(I_app - g_side) / (p['A'] * p['Ln'] * p['F'] * p['as_n'])
        BC_rn_surf = torch.dot(Dr_n[-1, :], cs_n) - flux_n / p['Ds_n']
        BC_rn_cent = torch.dot(Dr_n[0, :], cs_n) 
        
        flux_p = I_app / (p['A'] * p['Lp'] * p['F'] * p['as_p'])
        BC_rp_surf = torch.dot(Dr_p[-1, :], cs_p) - flux_p / p['Ds_p']
        BC_rp_cent = torch.dot(Dr_p[0, :], cs_p)

        # ==========================================
        # 2. Electrolyte Concentration PDE (Chebyshev)
        # ==========================================
        ce_n = ce_real[:self.Nx_n]
        ce_s = ce_real[self.Nx_n:self.Nx_n+self.Nx_s]
        ce_p = ce_real[-self.Nx_p:]
        
        def get_Deff(c, eps):
            c_m = c / 1000.0
            return (8.79e-11*c_m**2 - 3.97e-10*c_m + 4.86e-10) * (eps**p['b'])
            
        Deff_n = get_Deff(ce_n, p['eps_e_n'])
        Deff_s = get_Deff(ce_s, p['eps_e_s'])
        Deff_p = get_Deff(ce_p, p['eps_e_p'])
        
        Dx_n = self.Dx_n_ref / p['Ln']
        Dx_s = self.Dx_s_ref / p['Ls']
        Dx_p = self.Dx_p_ref / p['Lp']

        s_coeff = (1.0 - p['t_plus']) / (p['A'] * p['F'])
        src_n = s_coeff * I_app / p['Ln']
        src_p = -s_coeff * I_app / p['Lp']

        ce_state = torch.cat([ce_n, ce_s, ce_p])
        deff_state = torch.cat([-Deff_n, -Deff_s, -Deff_p])
        src_state = torch.cat([
            torch.ones_like(ce_n) * src_n,
            torch.zeros_like(ce_s),
            torch.ones_like(ce_p) * src_p
        ])
        if self.discretization_methods['electrolyte'] == 'finite_volume':
            face_coefficients = self.electrolyte_operators.face_coefficients(deff_state)
            flux_coefficient = lambda ctx, coeff=deff_state: ctx.operators.face_coefficients(coeff)
            flux_state = face_coefficients * self.electrolyte_operators.gradient(ce_state)
        else:
            flux_coefficient = deff_state
            flux_state = deff_state * self.electrolyte_operators.gradient(ce_state)
        dce = self.electrolyte_model.evaluate(
            ce_state,
            flux_coefficient=flux_coefficient,
            source=src_state
        )

        # ==========================================
        # 3. DAE Constraint Embedding 
        # ==========================================
        dcs_n_out = dcs_n.clone()
        dcs_p_out = dcs_p.clone()
        dce_out = dce.clone()

        if y_old is not None and dt is not None:
            # Overwrite Boundary Time Derivatives with Algebraic Constraints
            # Mathematically: R = y - y_o - dt * dy
            # If dy = (y - y_o)/dt - BC/dt --> R = BC
            y_o_cs_n = self.state(y_old, 'cs_n')
            y_o_cs_p = self.state(y_old, 'cs_p')
            ce_o = self.state(y_old, 'electrolyte')
            
            dcs_n_out[0]  = (cs_n[0] - y_o_cs_n[0])/dt - BC_rn_cent/dt
            dcs_n_out[-1] = (cs_n[-1] - y_o_cs_n[-1])/dt - BC_rn_surf/dt
            
            dcs_p_out[0]  = (cs_p[0] - y_o_cs_p[0])/dt - BC_rp_cent/dt
            dcs_p_out[-1] = (cs_p[-1] - y_o_cs_p[-1])/dt - BC_rp_surf/dt

            dce_out = self.electrolyte_model.apply_chebyshev_boundary_constraints(
                ce=ce_state,
                dce=dce_out,
                flux=flux_state,
                ce_old=ce_o,
                dt=dt,
                boundary_conditions={
                    'left': {'type': 'neumann', 'value': torch.tensor(0.0, device=self.device, dtype=torch.float64)},
                    'right': {'type': 'neumann', 'value': torch.tensor(0.0, device=self.device, dtype=torch.float64)}
                },
            )

        derivatives = {
            'cs_n': dcs_n_out,
            'cs_p': dcs_p_out,
            'electrolyte': dce_out,
            'sei': torch.zeros(self.Nsei, device=self.device, dtype=y_flat.dtype),
            'Lsei': dLsei_dt,
            'temperature': dT_dt,
        }
        return self.state_layout.pack(derivatives, device=self.device, dtype=y_flat.dtype)

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
        thermal_options=None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config, self.disc = config, discretization
        self.n_cells = config['n_series'] * config['n_parallel']
        self.n_series, self.n_parallel = config['n_series'], config['n_parallel']
        self.initial_state_mode = initial_state_mode
        self.initial_state_options = initial_state_options or {}
        self.balancing_options = balancing_options or {'enabled': False, 'strategy': 'none'}
        self.thermal_options = thermal_options or {'enabled': False, 'strategy': 'ambient'}
        
        self.raw_params = [self.get_standard_parameters() for _ in range(self.n_cells)]
        if overrides:
            for k, v in overrides.items():
                self.raw_params[k[0]*config['n_parallel'] + k[1]].update(v)
                
        self.physics = BatteryPhysics(config, discretization, self.raw_params, self.device)
        self.Gth = self._build_thermal_connection_matrix().to(self.device)
        self.Cth = torch.tensor([p['rho_Cp']*p['Vol_cell'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.h_amb = torch.tensor([p['hA'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.T_amb_vec = torch.tensor([p['T_amb'] for p in self.raw_params], device=self.device).reshape(-1,1)
        self.balancing_module = build_balancing_module(**self.balancing_options)
        self.thermal_module = build_thermal_module(**self.thermal_options)

        self.scale = self.physics.state_layout.scale_vector(self.device, torch.float64)

        self.basic_solver = BasicSolver(self)
        self.advanced_solver = AdvancedSolver(self)
        self.controlled_solver = ControlledSolver(self)
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
            ('sei', disc['Nsei']),
            ('Lsei', 1),
            ('temperature', 1),
        ]:
            old_slices[name] = slice(cursor, cursor + size)
            cursor += size

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
        I_grid = torch.full_like(OCV, I_pack_val / self.n_parallel)
        RTF = 8.314 * T / 96485.33
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
        return torch.sum(torch.mean(voltage_grid, dim=1)).item()

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

    def simulate(self, t_end, dt_init, I_pack=None, method='basic', controller=None, dt_max=None):
        if controller is not None:
            self.controlled_solver.simulate(t_end, dt_init, controller, dt_max=dt_max)
            return

        if method == 'basic':
            self.basic_solver.simulate(t_end, dt_init, I_pack)
        elif method == 'advanced':
            self.advanced_solver.simulate(t_end, dt_init, I_pack)
        else:
            raise ValueError("Method must be 'basic' or 'advanced'")

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
    def get_standard_parameters():
        return {
            'Ln': 8.52e-05, 'Ls': 1.20e-05, 'Lp': 7.56e-05, 'A': 0.1027, 'Rs_n': 5.86e-06, 'Rs_p': 5.22e-06,
            'F': 96485.33, 'R_g': 8.314, 'Ds_n': 3.3e-14, 'Ds_p': 4e-15, 'cs_max_n': 33133.0, 'cs_max_p': 63104.0,
            'as_n': 3.83e5, 'as_p': 3.82e5, 'sigma_n': 215.0, 'sigma_p': 0.18, 'eps_e_n': 0.25, 'eps_e_s': 0.47, 'eps_e_p': 0.335,
            't_plus': 0.2594, 'ce_0': 1000.0, 'b': 1.5, 'm_ref_n': 6.48e-7, 'E_r_n': 35000, 'm_ref_p': 3.42e-6, 'E_r_p': 17800,
            'kappa_sei': 1/200000.0, 'Msei': 0.162, 'rho_sei': 1690.0, 'Us': 0.4, 'Lsei_0': 5e-9,
            'Vol_cell': 2.42e-5, 'rho_Cp': 1.7676e6, 'hA': 0.0531, 'T_amb': 298.15
        }
