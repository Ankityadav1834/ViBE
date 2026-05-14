import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

class BasicSolver:
    def __init__(self, battery_solver):
        self.battery = battery_solver

    def solve_current_distribution(self, y, I_pack_val):
        return self.battery.compute_effective_cell_currents(y, I_pack_val)

    def get_pack_voltage(self, y, I_pack_val):
        return self.battery.get_pack_voltage(y, I_pack_val)

    def thermal_predictor(self, y_curr, dt, I_app):
        y_pred = y_curr.clone()
        T_old = y_curr[:, -1:]
        dY = self.battery.physics.batched_derivatives(y_curr, I_app)
        thermal_rhs = dY[:, -1:] + self.battery.compute_thermal_rhs(y_curr, I_app)
        y_pred[:, -1:] = T_old + dt * thermal_rhs
        return y_pred

    def newton_step(self, y_curr, dt, I_app, tol=1e-6, max_iter=15):
        y_next = self.thermal_predictor(y_curr, dt, I_app)
        I_vec = I_app.squeeze(-1)
        S = self.battery.scale
        invS = 1.0 / S
        
        for k in range(max_iter):
            # Pass y_curr and dt to enforce DAE constraints precisely
            dY = self.battery.physics.batched_derivatives(y_next, I_app, y_curr, dt)
            dY = self.apply_thermal_coupling(dY, y_next, I_app)
            R = y_next - y_curr - dt * dY
            
            res_norm = torch.norm(R * invS, p=float('inf'))
            if res_norm < tol: return y_next, True
            
            # The residual natively includes the flux/continuity BCs thanks to DAE formulation
            def res_fn(y_s, i_s, p_s, y_o, dt_val): 
                dy = self.battery.physics.compute_derivatives_functional(y_s, i_s, p_s, y_o, dt_val)
                return y_s - y_o - dt_val * dy

            try:
                # Retains massive vmap GPU speedup instead of loop-based block diagonal stack
                J = torch.vmap(torch.func.jacrev(res_fn, argnums=0), in_dims=(0,0,0,0,None))(y_next, I_vec, self.battery.physics.params, y_curr, dt)
                # Damping to prevent singular matrix issues during stiff nonlinear phases
                J += torch.eye(J.shape[-1], device=self.battery.device) * 1e-12
                delta_y = torch.linalg.solve(J, -R)
            except torch.linalg.LinAlgError:
                return y_next, False
            
            dT_max = 2.0
            delta_y[:, -1] = torch.clamp(delta_y[:, -1], -dT_max, dT_max)
            
            alpha = 1.0
            for _ in range(5):
                y_test = y_next + alpha * delta_y
                if torch.any(y_test[:, self.battery.nonnegative_mask] < 0):
                    alpha *= 0.5
                    continue
                dY_test = self.battery.physics.batched_derivatives(y_test, I_app, y_curr, dt)
                dY_test = self.apply_thermal_coupling(dY_test, y_test, I_app)
                R_test = y_test - y_curr - dt * dY_test
                if torch.norm(R_test * invS) < torch.norm(R * invS):
                    break
                alpha *= 0.5
            
            y_next = y_next + alpha * delta_y
            
        return y_next, False

    def apply_thermal_coupling(self, dY, Y, I_app):
        dY[:, -1:] += self.battery.compute_thermal_rhs(Y, I_app)
        return dY

    def simulate(self, t_end, dt_init, I_pack):
        t, dt = 0.0, dt_init
        times, y_hist = [0.0], [self.battery.y.clone().cpu()]
        print(f"Starting Basic GPU Simulation on {self.battery.device} (Adaptive Time Step - No LTE)")
        start_time = time.time()
        
        # Adaptive time stepping parameters (without LTE)
        dt_max = 50.0
        dt_min = 1e-5
        
        while t < t_end:
            I_cells = self.solve_current_distribution(self.battery.y, I_pack)
            y_new, success = self.newton_step(self.battery.y, dt, I_cells)
            
            if success:
                # == STEP ACCEPTED ==
                self.battery.y = y_new
                t += dt
                times.append(t)
                y_hist.append(self.battery.y.clone().cpu())

                limit_hit, cell_idx, cell_voltage, limit_kind = self.battery.check_voltage_limits(self.battery.y, I_pack)
                if limit_hit:
                    limit_value = self.battery.max_cell_voltage if limit_kind == "max" else self.battery.min_cell_voltage
                    print(
                        f"Stopping simulation: cell {cell_idx} reached the "
                        f"{limit_kind} voltage limit ({cell_voltage:.4f} V, limit {limit_value:.2f} V)."
                    )
                    break
                
                # Simple adaptive: increase dt slightly on success
                dt = min(dt * 1.1, dt_max)
                
                if t + dt > t_end: 
                    dt = t_end - t
                
                if len(times) % 10 == 0:
                    v_pack = self.get_pack_voltage(self.battery.y, I_pack)
                    t_max = torch.max(self.battery.y[:,-1]).item() - 273.15
                    print(f"t={t:.1f}s | dt={dt:.2f}s | Avg V={v_pack:.3f}V | Max T={t_max:.1f}C")
            else:
                # == STEP REJECTED - Decrease dt ==
                dt *= 0.5
                if dt <= dt_min:
                    print("ERROR: Minimum time step reached. Physics are too stiff to converge.")
                    break

                    
        print(f"Complete in {time.time()-start_time:.2f}s")
        self.process_results(times, y_hist, I_pack)

    def _get_voltage_breakdown_numpy(self, y_cell, I_cell, p):
        # Use PyTorch pipeline for exact results
        with torch.no_grad():
            y_t = torch.from_numpy(y_cell).unsqueeze(0).to(self.battery.device)
            # Override physics parameters temporarily if multiple cells differ, 
            # but usually get_circuit_parameters takes the whole batch.
            # To be safe and exact, we just query it for a single cell:
            OCV_t, R_series_t, I0_n_t, I0_p_t, T_cell_t, V_conc_t = self.battery.physics.get_circuit_parameters(y_t)
            
            # Since get_circuit_parameters returns for all cells, we just take the first one 
            # (assuming homogeneous params for this debug plot, or we reshape)
            # Actually y_t has shape (1, len), get_circuit_parameters returns (1, n_cells) or (1,).
            # To avoid shape mismatch, let's just compute terms manually from the returned tensors:
            term_RTF = 2 * p['R_g'] * T_cell_t[0].item() / p['F']
            denom_n = 2 * I0_n_t[0].item() * p['A'] * p['Ln'] * p['as_n']
            denom_p = 2 * I0_p_t[0].item() * p['A'] * p['Lp'] * p['as_p']
            
            V_rxn = term_RTF * (np.arcsinh(I_cell / (denom_n + 1e-12)) - np.arcsinh(-I_cell / (denom_p + 1e-12)))
            V_conc = V_conc_t[0].item()
            OCV = OCV_t[0].item()
            V_ohm_elec_and_solid = I_cell * R_series_t[0].item()
            
            # Approximations for breakdown plot
            R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3*p['A'])
            V_ohm_solid = I_cell * R_solid_ohm
            V_ohm_elec = V_ohm_elec_and_solid - V_ohm_solid
            
            if 'Lsei' in self.battery.physics.state_layout.slices:
                Lsei = self.battery.physics.state(y_t.squeeze(0), 'Lsei')[0].item()
            else:
                Lsei = p['Lsei_0']
            V_sei = I_cell * (Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei']))
            
            V_term = OCV - V_rxn - V_ohm_elec_and_solid - V_conc - V_sei
            
        return {
            "TermV": V_term, "OCV": OCV, "Rxn": V_rxn, 
            "OhmSolid": V_ohm_solid, "OhmElec": V_ohm_elec, 
            "Conc": V_conc, "SEI": V_sei
        }

    def process_results(self, times, y_list, I_pack):
        self.battery.output_manager.save(times, y_list, I_pack, filename='all_results.csv')
        print("Saved all_results.csv")

        times = np.array(times)
        y = torch.stack(y_list).numpy()
        n_steps = len(times)
        res = {k: np.zeros((n_steps, self.battery.n_cells)) for k in ['TermV', 'OCV', 'Rxn', 'OhmS', 'OhmE', 'Conc', 'SEI', 'Temp', 'Curr', 'SOC', 'SEI_Thick']}
        res['PackVoltage'] = np.zeros(n_steps)
        
        with torch.no_grad():
            for i in range(n_steps):
                y_t = torch.from_numpy(y[i]).to(self.battery.device)
                I_c_t = self.solve_current_distribution(y_t, I_pack)
                I_c = I_c_t.cpu().numpy().flatten()
                res['Curr'][i, :] = I_c
                volts = []
                for k in range(self.battery.n_cells):
                    p = self.battery.raw_params[k]
                    y_c = y[i, k]
                    bd = self._get_voltage_breakdown_numpy(y_c, I_c[k], p)
                    res['TermV'][i,k] = bd['TermV']
                    res['OCV'][i,k] = bd['OCV']
                    res['Rxn'][i,k] = bd['Rxn']
                    res['OhmS'][i,k] = bd['OhmSolid']
                    res['OhmE'][i,k] = bd['OhmElec']
                    res['Conc'][i,k] = bd['Conc']
                    res['SEI'][i,k] = bd['SEI']
                    res['Temp'][i,k] = y_c[-1] # K
                    res['SEI_Thick'][i,k] = y_c[-2] * 1e9 # nm
                    res['SOC'][i,k] = (y_c[self.battery.physics.Nr_n-1]/p['cs_max_n'] - 0.01)/0.94
                    volts.append(bd['TermV'])
                res['PackVoltage'][i] = np.sum(np.mean(np.array(volts).reshape(self.battery.n_series, self.battery.n_parallel), axis=1))

        fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
        t_h = times
        axes[0,0].plot(t_h, res['PackVoltage'], 'k'); axes[0,0].set_title('Pack Voltage')
        ax2 = axes[0,0].twinx(); ax2.plot(t_h, np.full_like(t_h, I_pack), 'r--'); ax2.set_ylabel('Pack Current [A]')
        for k in range(self.battery.n_cells): axes[0,1].plot(t_h, res['TermV'][:,k], label=f'C{k}')
        axes[0,1].legend(); axes[0,1].set_title('Cell Voltages')
        for k in range(self.battery.n_cells): axes[1,0].plot(t_h, res['Curr'][:,k])
        axes[1,0].set_title('Cell Currents [A]')
        for k in range(self.battery.n_cells): axes[1,1].plot(t_h, res['Temp'][:,k] - 273.15)
        axes[1,1].set_title('Temperature [C]')
        for k in range(self.battery.n_cells): axes[2,0].plot(t_h, res['SEI_Thick'][:,k])
        axes[2,0].set_title('SEI Thickness [nm]')
        for k in range(self.battery.n_cells): axes[2,1].plot(t_h, res['SOC'][:,k])
        axes[2,1].set_title('SOC')
        
        idx_b = 1 if self.battery.n_cells > 1 else 0
        stk1 = [res['TermV'][:,idx_b], res['SEI'][:,idx_b], res['OhmS'][:,idx_b], res['OhmE'][:,idx_b], res['Conc'][:,idx_b], res['Rxn'][:,idx_b]]
        axes[3,0].stackplot(t_h, stk1, labels=['V','SEI','OhmS','OhmE','Conc','Rxn'], alpha=0.6)
        axes[3,0].plot(t_h, res['OCV'][:,idx_b], 'k--'); axes[3,0].set_title(f'Breakdown C{idx_b}'); axes[3,0].legend(loc='lower left')
        
        idx_a = 0
        stk0 = [res['TermV'][:,idx_a], res['SEI'][:,idx_a], res['OhmS'][:,idx_a], res['OhmE'][:,idx_a], res['Conc'][:,idx_a], res['Rxn'][:,idx_a]]
        axes[3,1].stackplot(t_h, stk0, labels=['V','SEI','OhmS','OhmE','Conc','Rxn'], alpha=0.6)
        axes[3,1].plot(t_h, res['OCV'][:,idx_a], 'k--'); axes[3,1].set_title(f'Breakdown C{idx_a}'); axes[3,1].legend(loc='lower left')
        
        plt.show()
        pd.DataFrame({'time': times, 'pack_voltage': res['PackVoltage']}).to_csv('voltage_vs_time.csv', index=False)
        print("Saved voltage_vs_time.csv")


class AdvancedSolver:
    def __init__(self, battery_solver):
        self.battery = battery_solver

    def solve_current_distribution(self, y, I_pack_val):
        return self.battery.compute_effective_cell_currents(y, I_pack_val)

    def get_pack_voltage(self, y, I_pack_val):
        return self.battery.get_pack_voltage(y, I_pack_val)

    def thermal_predictor(self, y_curr, dt, I_app):
        y_pred = y_curr.clone()
        T_old = y_curr[:, -1:]
        dY = self.battery.physics.batched_derivatives(y_curr, I_app)
        thermal_rhs = dY[:, -1:] + self.battery.compute_thermal_rhs(y_curr, I_app)
        y_pred[:, -1:] = T_old + dt * thermal_rhs
        return y_pred

    def newton_step(self, y_curr, dt, I_app, tol=1e-6, max_iter=15):
        y_next = self.thermal_predictor(y_curr, dt, I_app)
        I_vec = I_app.squeeze(-1)
        S = self.battery.scale
        invS = 1.0 / S
        
        for k in range(max_iter):
            # Pass y_curr and dt to enforce DAE constraints precisely
            dY = self.battery.physics.batched_derivatives(y_next, I_app, y_curr, dt)
            dY = self.apply_thermal_coupling(dY, y_next, I_app)
            R = y_next - y_curr - dt * dY
            
            res_norm = torch.norm(R * invS, p=float('inf'))
            if res_norm < tol: return y_next, True
            
            # The residual natively includes the flux/continuity BCs thanks to DAE formulation
            def res_fn(y_s, i_s, p_s, y_o, dt_val): 
                dy = self.battery.physics.compute_derivatives_functional(y_s, i_s, p_s, y_o, dt_val)
                return y_s - y_o - dt_val * dy

            try:
                # Retains massive vmap GPU speedup instead of loop-based block diagonal stack
                J = torch.vmap(torch.func.jacrev(res_fn, argnums=0), in_dims=(0,0,0,0,None))(y_next, I_vec, self.battery.physics.params, y_curr, dt)
                # Damping to prevent singular matrix issues during stiff nonlinear phases
                J += torch.eye(J.shape[-1], device=self.battery.device) * 1e-12
                delta_y = torch.linalg.solve(J, -R)
            except torch.linalg.LinAlgError:
                return y_next, False
            
            dT_max = 2.0
            delta_y[:, -1] = torch.clamp(delta_y[:, -1], -dT_max, dT_max)
            
            alpha = 1.0
            for _ in range(5):
                y_test = y_next + alpha * delta_y
                if torch.any(y_test[:, self.battery.nonnegative_mask] < 0):
                    alpha *= 0.5
                    continue
                dY_test = self.battery.physics.batched_derivatives(y_test, I_app, y_curr, dt)
                dY_test = self.apply_thermal_coupling(dY_test, y_test, I_app)
                R_test = y_test - y_curr - dt * dY_test
                if torch.norm(R_test * invS) < torch.norm(R * invS):
                    break
                alpha *= 0.5
            
            y_next = y_next + alpha * delta_y
            
        return y_next, False

    def apply_thermal_coupling(self, dY, Y, I_app):
        dY[:, -1:] += self.battery.compute_thermal_rhs(Y, I_app)
        return dY

    def compute_step_lte(self, y_c, dt, I_pack_val, abstol=1e-5, reltol=1e-3):
        """
        Calculates Local Truncation Error using Step Doubling (Richardson Extrapolation).
        Returns the accepted state, the normalized error, and a success flag.
        """
        # 1. Take one FULL step (dt)
        I_cells_full = self.solve_current_distribution(y_c, I_pack_val)
        y_full, success_full = self.newton_step(y_c, dt, I_cells_full)
        if not success_full:
            return None, float('inf'), False
            
        # 2. Take two HALF steps (dt/2)
        dt_half = dt / 2.0
        I_cells_h1 = self.solve_current_distribution(y_c, I_pack_val)
        y_half_1, success_h1 = self.newton_step(y_c, dt_half, I_cells_h1)
        if not success_h1:
            return None, float('inf'), False
            
        I_cells_h2 = self.solve_current_distribution(y_half_1, I_pack_val)
        y_half_2, success_h2 = self.newton_step(y_half_1, dt_half, I_cells_h2)
        if not success_h2:
            return None, float('inf'), False
            
        # 3. Calculate the Local Truncation Error (LTE)
        lte_raw = torch.abs(y_half_2 - y_full)
        
        # 4. Normalize the error using scaled tolerances
        weight = abstol * self.battery.scale + reltol * torch.abs(y_half_2)
        error_norm = torch.max(lte_raw / weight).item()
        
        # Return the HALF step result mathematically
        return y_half_2, error_norm, True

    def simulate(self, t_end, dt_init, I_pack):
        t, dt = 0.0, dt_init
        times, y_hist = [0.0], [self.battery.y.clone().cpu()]
        print(f"Starting GPU Simulation on {self.battery.device} (Chebyshev Collocation w/ LTE PID Control)")
        start_time = time.time()
        
        # PID Controller Parameters
        safety_factor = 0.9    # S: Keeps us slightly below the absolute maximum step size
        dt_max = 150.0          # Maximum allowed time step
        dt_min = 1e-5          # Minimum allowed time step to prevent infinite loops
        
        while t < t_end:
            # Attempt the step with Error Estimation
            y_new, err, success = self.compute_step_lte(self.battery.y, dt, I_pack)
            
            if success:
                if err <= 1.0:
                    # == STEP ACCEPTED ==
                    self.battery.y = y_new
                    t += dt
                    
                    times.append(t)
                    y_hist.append(self.battery.y.clone().cpu())

                    limit_hit, cell_idx, cell_voltage, limit_kind = self.battery.check_voltage_limits(self.battery.y, I_pack)
                    if limit_hit:
                        limit_value = self.battery.max_cell_voltage if limit_kind == "max" else self.battery.min_cell_voltage
                        print(
                            f"Stopping simulation: cell {cell_idx} reached the "
                            f"{limit_kind} voltage limit ({cell_voltage:.4f} V, limit {limit_value:.2f} V)."
                        )
                        break
                    
                    # Calculate new step size (I-Controller)
                    dt_factor = safety_factor * (1.0 / (err + 1e-10))**0.5
                    
                    # Clamp the change to prevent wild swings
                    dt_factor = max(0.2, min(2.0, dt_factor))
                    dt = min(dt * dt_factor, dt_max)
                    
                    if t + dt > t_end: dt = t_end - t
                    
                    if len(times) % 10 == 0:
                        v_pack = self.get_pack_voltage(self.battery.y, I_pack)
                        t_max = torch.max(self.battery.y[:,-1]).item() - 273.15
                        print(f"t={t:.1f}s | dt={dt:.2f}s | Err={err:.3f} | Avg V={v_pack:.3f}V | Max T={t_max:.1f}C")
                else:
                    # == STEP REJECTED (Physics too fast/erroneous) ==
                    dt_factor = safety_factor * (1.0 / err)**0.5
                    dt_factor = max(0.2, dt_factor)
                    dt = max(dt * dt_factor, dt_min)
            else:
                # == NEWTON SOLVER FAILED ==
                dt *= 0.5
                if dt <= dt_min:
                    print("ERROR: Minimum time step reached. Physics are too stiff to converge.")
                    break
                    
        print(f"Complete in {time.time()-start_time:.2f}s")
        self.process_results(times, y_hist, I_pack)

    def _get_voltage_breakdown_numpy(self, y_cell, I_cell, p):
        Nr_n, Nr_p, Nel = self.battery.physics.Nr_n, self.battery.physics.Nr_p, self.battery.physics.Nel
        cs_n, cs_p, ce_eps = y_cell[:Nr_n], y_cell[Nr_n:Nr_n+Nr_p], y_cell[Nr_n+Nr_p:Nr_n+Nr_p+Nel]
        Lsei, T_cell = y_cell[-2], y_cell[-1]
        
        sto_n, sto_p = np.clip(cs_n[-1]/p['cs_max_n'], 1e-4, 0.999), np.clip(cs_p[-1]/p['cs_max_p'], 1e-4, 0.999)
        def ocp_n(s): return 1.9793*np.exp(-39.3631*s)+0.2482-0.0909*np.tanh(29.8538*(s-0.1234))-0.04478*np.tanh(14.9159*(s-0.2769))-0.0205*np.tanh(30.4444*(s-0.6103))
        def ocp_p(s): return -0.8090*s+4.4875-0.0428*np.tanh(18.5138*(s-0.5542))-17.7326*np.tanh(15.7890*(s-0.3117))+17.5842*np.tanh(15.9308*(s-0.3120))
        OCV = ocp_p(sto_p) - ocp_n(sto_n)
        
        inv_T, inv_Ref = 1.0/T_cell, 1.0/298.15
        arr_n = np.exp(p['E_r_n']/p['R_g'] * (inv_Ref - inv_T))
        arr_p = np.exp(p['E_r_p']/p['R_g'] * (inv_Ref - inv_T))
        
        ce_real = ce_eps / (self.battery.physics.eps_vec.cpu().numpy() + 1e-12)
        c_k = ce_real / 1000.0
        local_kappa = (0.1297 * c_k**3 - 2.51 * c_k**1.5 + 3.329 * c_k)
        
        W_n, W_s, W_p = self.battery.physics.W_n.cpu().numpy(), self.battery.physics.W_s.cpu().numpy(), self.battery.physics.W_p.cpu().numpy()
        
        int_inv_kappa = (p['Ln']*np.sum(W_n / (local_kappa[:self.battery.physics.Nx_n] + 1e-12)) +
                         p['Ls']*np.sum(W_s / (local_kappa[self.battery.physics.Nx_n:self.battery.physics.Nx_n+self.battery.physics.Nx_s] + 1e-12)) +
                         p['Lp']*np.sum(W_p / (local_kappa[-self.battery.physics.Nx_p:] + 1e-12)))
        
        eff_kappa = (p['Ln'] + p['Ls'] + p['Lp']) / (int_inv_kappa + 1e-12)
        R_eff_term = (p['Ln'] / (3 * p['eps_e_n'] ** p['b']) + p['Ls'] / (p['eps_e_s'] ** p['b']) + p['Lp'] / (3 * p['eps_e_p'] ** p['b']))
        V_ohm_elec = (I_cell / p['A']) / (eff_kappa + 1e-9) * R_eff_term
        
        R_sei = Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
        V_sei = I_cell * R_sei
        R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3*p['A'])
        V_ohm_solid = I_cell * R_solid_ohm
        
        j0_n = p['m_ref_n']*arr_n * np.sum(np.sqrt(np.maximum(1e-12, ce_real[:self.battery.physics.Nx_n]))*W_n) * np.sqrt(np.maximum(1e-12, cs_n[-1]*(p['cs_max_n']-cs_n[-1])))
        j0_p = p['m_ref_p']*arr_p * np.sum(np.sqrt(np.maximum(1e-12, ce_real[-self.battery.physics.Nx_p:]))*W_p) * np.sqrt(np.maximum(1e-12, cs_p[-1]*(p['cs_max_p']-cs_p[-1])))
        term = 2*p['R_g']*T_cell/p['F']
        V_rxn = term*np.arcsinh(I_cell/(2*p['A']*p['Ln']*p['as_n']*j0_n+1e-12)) - term*np.arcsinh(-I_cell/(2*p['A']*p['Lp']*p['as_p']*j0_p+1e-12))
        
        ln_ce = np.log(np.maximum(ce_real, 1e-6))
        V_conc = term * (1.0-p['t_plus']) * (np.sum(ln_ce[:self.battery.physics.Nx_n]*W_n) - np.sum(ln_ce[-self.battery.physics.Nx_p:]*W_p))
        V_term = OCV - V_rxn - V_ohm_solid - V_ohm_elec - V_conc - V_sei
        return {"TermV": V_term, "OCV": OCV, "Rxn": V_rxn, "OhmSolid": V_ohm_solid, "OhmElec": V_ohm_elec, "Conc": V_conc, "SEI": V_sei}

    def process_results(self, times, y_list, I_pack):
        self.battery.output_manager.save(times, y_list, I_pack, filename='all_results.csv')
        print("Saved all_results.csv")

        times = np.array(times)
        y = torch.stack(y_list).numpy()
        n_steps = len(times)
        res = {k: np.zeros((n_steps, self.battery.n_cells)) for k in ['TermV', 'OCV', 'Rxn', 'OhmS', 'OhmE', 'Conc', 'SEI', 'Temp', 'Curr', 'SOC', 'SEI_Thick']}
        res['PackVoltage'] = np.zeros(n_steps)
        
        with torch.no_grad():
            for i in range(n_steps):
                y_t = torch.from_numpy(y[i]).to(self.battery.device)
                I_c_t = self.solve_current_distribution(y_t, I_pack)
                I_c = I_c_t.cpu().numpy().flatten()
                res['Curr'][i, :] = I_c
                volts = []
                for k in range(self.battery.n_cells):
                    p = self.battery.raw_params[k]
                    y_c = y[i, k]
                    bd = self._get_voltage_breakdown_numpy(y_c, I_c[k], p)
                    res['TermV'][i,k] = bd['TermV']
                    res['OCV'][i,k] = bd['OCV']
                    res['Rxn'][i,k] = bd['Rxn']
                    res['OhmS'][i,k] = bd['OhmSolid']
                    res['OhmE'][i,k] = bd['OhmElec']
                    res['Conc'][i,k] = bd['Conc']
                    res['SEI'][i,k] = bd['SEI']
                    res['Temp'][i,k] = y_c[-1] # K
                    res['SEI_Thick'][i,k] = y_c[-2] * 1e9 # nm
                    res['SOC'][i,k] = (y_c[self.battery.physics.Nr_n-1]/p['cs_max_n'] - 0.01)/0.94
                    volts.append(bd['TermV'])
                res['PackVoltage'][i] = np.sum(np.mean(np.array(volts).reshape(self.battery.n_series, self.battery.n_parallel), axis=1))

        fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
        t_h = times
        axes[0,0].plot(t_h, res['PackVoltage'], 'k'); axes[0,0].set_title('Pack Voltage')
        ax2 = axes[0,0].twinx(); ax2.plot(t_h, np.full_like(t_h, I_pack), 'r--'); ax2.set_ylabel('Pack Current [A]')
        for k in range(self.battery.n_cells): axes[0,1].plot(t_h, res['TermV'][:,k], label=f'C{k}')
        axes[0,1].legend(); axes[0,1].set_title('Cell Voltages')
        for k in range(self.battery.n_cells): axes[1,0].plot(t_h, res['Curr'][:,k])
        axes[1,0].set_title('Cell Currents [A]')
        for k in range(self.battery.n_cells): axes[1,1].plot(t_h, res['Temp'][:,k] - 273.15)
        axes[1,1].set_title('Temperature [C]')
        for k in range(self.battery.n_cells): axes[2,0].plot(t_h, res['SEI_Thick'][:,k])
        axes[2,0].set_title('SEI Thickness [nm]')
        for k in range(self.battery.n_cells): axes[2,1].plot(t_h, res['SOC'][:,k])
        axes[2,1].set_title('SOC')
        
        idx_b = 1 if self.battery.n_cells > 1 else 0
        stk1 = [res['TermV'][:,idx_b], res['SEI'][:,idx_b], res['OhmS'][:,idx_b], res['OhmE'][:,idx_b], res['Conc'][:,idx_b], res['Rxn'][:,idx_b]]
        axes[3,0].stackplot(t_h, stk1, labels=['V','SEI','OhmS','OhmE','Conc','Rxn'], alpha=0.6)
        axes[3,0].plot(t_h, res['OCV'][:,idx_b], 'k--'); axes[3,0].set_title(f'Breakdown C{idx_b}'); axes[3,0].legend(loc='lower left')
        
        idx_a = 0
        stk0 = [res['TermV'][:,idx_a], res['SEI'][:,idx_a], res['OhmS'][:,idx_a], res['OhmE'][:,idx_a], res['Conc'][:,idx_a], res['Rxn'][:,idx_a]]
        axes[3,1].stackplot(t_h, stk0, labels=['V','SEI','OhmS','OhmE','Conc','Rxn'], alpha=0.6)
        axes[3,1].plot(t_h, res['OCV'][:,idx_a], 'k--'); axes[3,1].set_title(f'Breakdown C{idx_a}'); axes[3,1].legend(loc='lower left')
        
        plt.show()
        pd.DataFrame({'time': times, 'pack_voltage': res['PackVoltage']}).to_csv('voltage_vs_time.csv', index=False)
        print("Saved voltage_vs_time.csv")


class ControlledSolver:
    def __init__(self, battery_solver):
        self.battery = battery_solver
        self.core_solver = BasicSolver(battery_solver)

    def simulate(self, t_end, dt_init, controller, dt_max=None):
        t, dt = 0.0, dt_init
        dt_max = 5.0 if dt_max is None else dt_max
        times = [0.0]
        y_hist = [self.battery.y.clone().cpu()]
        I_pack_hist = [float(getattr(controller, 'initial_current', 0.0))]

        print(f"Starting Controlled Simulation on {self.battery.device}")
        start_time = time.time()

        while t < t_end:
            I_now = float(controller.compute_current(t, self.battery.y, self.battery, dt))
            I_cells = self.battery.compute_effective_cell_currents(self.battery.y, I_now)
            y_new, success = self.core_solver.newton_step(self.battery.y, dt, I_cells)

            if success:
                self.battery.y = y_new
                t += dt
                dt = min(dt * 1.2, dt_max)

                if t + dt > t_end:
                    dt = t_end - t

                times.append(t)
                y_hist.append(self.battery.y.clone().cpu())
                I_pack_hist.append(I_now)

                cell_voltages = self.battery.get_exact_terminal_voltages(self.battery.y, I_cells)
                should_stop, stop_message = controller.should_stop(t, self.battery.y, cell_voltages, I_now)
                if should_stop:
                    print(stop_message)
                    break

                if len(times) % 20 == 0:
                    stage = getattr(controller, 'current_stage', controller.__class__.__name__)
                    print(
                        f"[{stage}] Progress: {t / t_end * 100:.1f}% | "
                        f"Max T: {torch.max(self.battery.y[:, -1]).item() - 273.15:.2f}C"
                    )
            else:
                dt *= 0.5
                if dt < 1e-5:
                    print("ERROR: Minimum time step reached. Physics are too stiff to converge.")
                    break

        print(f"Complete in {time.time()-start_time:.2f}s")
        self.process_results(times, y_hist, I_pack_hist)

    def process_results(self, times, y_list, I_pack_hist):
        self.battery.output_manager.save(times, y_list, I_pack_hist, filename='all_results.csv')
        print("Saved all_results.csv")

        times = np.array(times)
        y = torch.stack(y_list).numpy()
        I_pack_hist = np.array(I_pack_hist)
        n_steps = len(times)
        res = {
            k: np.zeros((n_steps, self.battery.n_cells))
            for k in ['TermV', 'OCV', 'Rxn', 'OhmS', 'OhmE', 'Conc', 'SEI', 'Temp', 'Curr', 'SOC', 'SEI_Thick']
        }
        res['PackVoltage'] = np.zeros(n_steps)
        res['PackCurrent'] = I_pack_hist

        with torch.no_grad():
            for i in range(n_steps):
                y_t = torch.from_numpy(y[i]).to(self.battery.device)
                I_now = float(I_pack_hist[i])
                I_c_t = self.battery.compute_effective_cell_currents(y_t, I_now)
                I_c = I_c_t.cpu().numpy().flatten()
                res['Curr'][i, :] = I_c
                volts = []
                for k in range(self.battery.n_cells):
                    p = self.battery.raw_params[k]
                    y_c = y[i, k]
                    bd = self.core_solver._get_voltage_breakdown_numpy(y_c, I_c[k], p)
                    res['TermV'][i, k] = bd['TermV']
                    res['OCV'][i, k] = bd['OCV']
                    res['Rxn'][i, k] = bd['Rxn']
                    res['OhmS'][i, k] = bd['OhmSolid']
                    res['OhmE'][i, k] = bd['OhmElec']
                    res['Conc'][i, k] = bd['Conc']
                    res['SEI'][i, k] = bd['SEI']
                    y_t_single = torch.from_numpy(y_c).unsqueeze(0).to(self.battery.device)
                    res['Temp'][i, k] = self.battery.physics.state(y_t_single, 'temperature')[0].item()
                    if 'Lsei' in self.battery.physics.state_layout.slices:
                        res['SEI_Thick'][i, k] = self.battery.physics.state(y_t_single, 'Lsei')[0].item() * 1e9
                    else:
                        res['SEI_Thick'][i, k] = p['Lsei_0'] * 1e9
                    
                    cs_n_state = self.battery.physics.state(y_t_single, 'cs_n')[0].cpu().numpy()
                    # Average over x for surface SOC
                    surf_cs_n = np.mean(cs_n_state[self.battery.physics.Nr_n - 1::self.battery.physics.Nr_n])
                    res['SOC'][i, k] = (surf_cs_n / p['cs_max_n'] - 0.01) / 0.94
                    volts.append(bd['TermV'])
                res['PackVoltage'][i] = np.sum(
                    np.mean(np.array(volts).reshape(self.battery.n_series, self.battery.n_parallel), axis=1)
                )

        fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
        t_h = times
        axes[0, 0].plot(t_h, res['PackVoltage'], 'k')
        axes[0, 0].set_title('Pack Voltage')
        ax2 = axes[0, 0].twinx()
        ax2.plot(t_h, res['PackCurrent'], 'r--', label='Controller Output')
        ax2.set_ylabel('Pack Current [A]')
        axes[0, 0].legend()
        ax2.legend(loc='upper right')

        for k in range(self.battery.n_cells):
            axes[0, 1].plot(t_h, res['TermV'][:, k], label=f'C{k}')
        axes[0, 1].legend()
        axes[0, 1].set_title('Cell Voltages')
        for k in range(self.battery.n_cells):
            axes[1, 0].plot(t_h, res['Curr'][:, k])
        axes[1, 0].set_title('Cell Currents [A]')
        for k in range(self.battery.n_cells):
            axes[1, 1].plot(t_h, res['Temp'][:, k] - 273.15)
        axes[1, 1].set_title('Temperature [C]')
        for k in range(self.battery.n_cells):
            axes[2, 0].plot(t_h, res['SEI_Thick'][:, k])
        axes[2, 0].set_title('SEI Thickness [nm]')
        for k in range(self.battery.n_cells):
            axes[2, 1].plot(t_h, res['SOC'][:, k])
        axes[2, 1].set_title('SOC')

        idx_b = 1 if self.battery.n_cells > 1 else 0
        stk1 = [
            res['TermV'][:, idx_b],
            res['SEI'][:, idx_b],
            res['OhmS'][:, idx_b],
            res['OhmE'][:, idx_b],
            res['Conc'][:, idx_b],
            res['Rxn'][:, idx_b],
        ]
        axes[3, 0].stackplot(t_h, stk1, labels=['V', 'SEI', 'OhmS', 'OhmE', 'Conc', 'Rxn'], alpha=0.6)
        axes[3, 0].plot(t_h, res['OCV'][:, idx_b], 'k--')
        axes[3, 0].set_title(f'Breakdown C{idx_b}')
        axes[3, 0].legend(loc='lower left')

        idx_a = 0
        stk0 = [
            res['TermV'][:, idx_a],
            res['SEI'][:, idx_a],
            res['OhmS'][:, idx_a],
            res['OhmE'][:, idx_a],
            res['Conc'][:, idx_a],
            res['Rxn'][:, idx_a],
        ]
        axes[3, 1].stackplot(t_h, stk0, labels=['V', 'SEI', 'OhmS', 'OhmE', 'Conc', 'Rxn'], alpha=0.6)
        axes[3, 1].plot(t_h, res['OCV'][:, idx_a], 'k--')
        axes[3, 1].set_title(f'Breakdown C{idx_a}')
        axes[3, 1].legend(loc='lower left')

        plt.show()
        pd.DataFrame(
            {'time': times, 'pack_voltage': res['PackVoltage'], 'pack_current': res['PackCurrent']}
        ).to_csv('sim_results.csv', index=False)
        print("Saved sim_results.csv")
