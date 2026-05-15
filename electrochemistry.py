import torch

def calculate_ocp(physics, stoich, electrode):
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

def get_circuit_parameters(physics, y_batch):
    """
    Current Distribution Algorithm
    Calculates the equivalent circuit parameters (OCV, Resistance, etc) for the 
    pack-level current distribution solver.
    """
    p = physics.params
    cs_n = physics.state(y_batch, 'cs_n')
    cs_p = physics.state(y_batch, 'cs_p')
    ce_eps = physics.state(y_batch, 'electrolyte')
    T_cell = physics.state(y_batch, 'temperature')
    
    Lsei = physics.state(y_batch, 'Lsei') if physics.sei_enabled else p['Lsei_0'] * torch.ones_like(T_cell)
    
    theta_n = cs_n[:, -1:] / p['cs_max_n']
    theta_p = cs_p[:, -1:] / p['cs_max_p']
    Un = calculate_ocp(physics, theta_n, 'anode')
    Up = calculate_ocp(physics, theta_p, 'cathode')
    OCV = Up - Un
    
    V_conc_list = []
    R_elec_list = []
    for i in range(y_batch.shape[0]):
         p_i = {k: v[i] for k, v in p.items()}
         V_conc_list.append(physics.calculate_conc_overpotential(ce_eps[i], T_cell[i], p_i))
         R_elec_list.append(physics.calculate_electrolyte_physics(ce_eps[i], p_i))
         
    V_conc = torch.stack(V_conc_list).reshape(-1, 1)
    R_electrolyte = torch.stack(R_elec_list).reshape(-1, 1)

    inv_T = 1.0 / T_cell
    inv_T_ref = 1.0 / 298.15
    arr_n = torch.exp(p['E_r_n'] / p['R_g'] * (inv_T_ref - inv_T))
    arr_p = torch.exp(p['E_r_p'] / p['R_g'] * (inv_T_ref - inv_T))
    
    ce_real_batch = ce_eps / (physics.eps_vec.unsqueeze(0) + 1e-12)
    
    W_n_batch = physics.W_n.unsqueeze(0)
    W_p_batch = physics.W_p.unsqueeze(0)
    
    sqrt_ce_n = torch.sum(torch.sqrt(torch.clamp(ce_real_batch[:, :physics.Nx_n], min=1e-12)) * W_n_batch, dim=1, keepdim=True)
    sqrt_ce_p = torch.sum(torch.sqrt(torch.clamp(ce_real_batch[:, -physics.Nx_p:], min=1e-12)) * W_p_batch, dim=1, keepdim=True)

    j0_n = p['m_ref_n'] * arr_n * sqrt_ce_n * torch.sqrt(torch.clamp(cs_n[:, -1:] * (p['cs_max_n'] - cs_n[:, -1:]), min=1e-12))
    j0_p = p['m_ref_p'] * arr_p * sqrt_ce_p * torch.sqrt(torch.clamp(cs_p[:, -1:] * (p['cs_max_p'] - cs_p[:, -1:]), min=1e-12))
    
    I0_n = j0_n * p['A'] * p['Ln'] * p['as_n']
    I0_p = j0_p * p['A'] * p['Lp'] * p['as_p']

    R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3.0 * p['A'])
    R_sei = Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
    R_series = R_solid_ohm + R_electrolyte + R_sei 
    
    return OCV, R_series, I0_n, I0_p, T_cell, V_conc

def compute_derivatives_functional(physics, y_flat, I_app, p, y_old=None, dt=None):
    """
    Coupled Physics Calculations (Electrochemistry & Heat)
    
    Why isn't this logic inside `equations.py`? 
    Because these variables (OCV, Butler-Volmer kinetics, heat generation) affect 
    MULTIPLE equations simultaneously. If we put Butler-Volmer directly into the 
    lithium diffusion equation AND the electrolyte equation AND the heat equation, 
    the solver would compute it 3 times per step, drastically slowing down the simulation.
    Instead, we compute all shared physics here ONCE per step, and pass them to the 
    equations via the `context` dictionary.
    """
    from states import evaluate_registered_rhs
    
    cs_n = physics.state(y_flat, 'cs_n')
    cs_p = physics.state(y_flat, 'cs_p')
    ce_eps = physics.state(y_flat, 'electrolyte')
    T_cell = physics.state(y_flat, 'temperature')
    
    Lsei = physics.state(y_flat, 'Lsei') if physics.sei_enabled else p['Lsei_0'] * torch.ones_like(T_cell)
    
    inv_T = 1.0 / T_cell
    inv_T_ref = 1.0 / 298.15
    arr_n = torch.exp(p['E_r_n'] / p['R_g'] * (inv_T_ref - inv_T))
    arr_p = torch.exp(p['E_r_p'] / p['R_g'] * (inv_T_ref - inv_T))
    
    theta_n = cs_n[-1:] / p['cs_max_n']
    theta_p = cs_p[-1:] / p['cs_max_p']
    Un = calculate_ocp(physics, theta_n, 'anode')
    Up = calculate_ocp(physics, theta_p, 'cathode')
    OCV = Up - Un
    
    R_electrolyte = physics.calculate_electrolyte_physics(ce_eps, p)
    V_conc = physics.calculate_conc_overpotential(ce_eps, T_cell, p)
    
    R_sei = Lsei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
    R_solid_ohm = (p['Ln']/p['sigma_n'] + p['Lp']/p['sigma_p'])/(3.0 * p['A'])
    V_ohm_total = I_app * (R_solid_ohm + R_electrolyte + R_sei)
    
    ce_real = ce_eps / (physics.eps_vec + 1e-12)
    sqrt_ce_n_avg = torch.sum(torch.sqrt(torch.clamp(ce_real[:physics.Nx_n], min=1e-12)) * physics.W_n)
    sqrt_ce_p_avg = torch.sum(torch.sqrt(torch.clamp(ce_real[-physics.Nx_p:], min=1e-12)) * physics.W_p)
    
    j0_n = p['m_ref_n'] * arr_n * sqrt_ce_n_avg * torch.sqrt(torch.clamp(cs_n[-1] * (p['cs_max_n'] - cs_n[-1]), min=1e-12))
    j0_p = p['m_ref_p'] * arr_p * sqrt_ce_p_avg * torch.sqrt(torch.clamp(cs_p[-1] * (p['cs_max_p'] - cs_p[-1]), min=1e-12))
    
    term_RTF = 2 * p['R_g'] * T_cell / p['F']

    # ── Intercalation current densities [A/m²] ───────────────────────────────
    i_n = I_app / (p['as_n'] * p['A'] * p['Ln'])
    i_p = -I_app / (p['as_p'] * p['A'] * p['Lp'])

    # ── 1. Stress model ───────────────────────────────────────────────────────
    stress_model_name = getattr(physics, 'stress_model_name', 'builtin')
    if stress_model_name == 'builtin':
        # Inline for speed (avoids import on every call)
        E_sei, nu_sei, Omega, sigma_intr = 10e9, 0.25, 3.17e-6, -0.5e9
        E_g, nu_g = 15e9, 0.3
        E_sei_b = E_sei / (1.0 - nu_sei)
        pf = E_g * Omega / (3.0 * (1.0 - nu_g))
        c_surf = cs_n[-1]
        c_surf_ref = 0.8 * p['cs_max_n']
        r_ref = physics.r_n_ref
        r_faces = torch.zeros(physics.Nr_n + 1, device=physics.device, dtype=torch.float64)
        r_faces[1:-1] = 0.5 * (r_ref[:-1] + r_ref[1:])
        r_faces[-1] = 1.0
        v = r_faces[1:]**3 - r_faces[:-1]**3
        c_bar = torch.sum(cs_n * v) / torch.sum(v)
        sth_surf = 3.0 * pf * (c_bar - c_surf)
        sigma_sei = E_sei_b * (Omega / 3.0) * (c_surf - c_surf_ref) + sigma_intr
        total_surf_stress = sth_surf + sigma_sei
        K_Ic = 0.3e6
        sigma_crack_threshold = K_Ic / torch.sqrt(torch.pi * torch.clamp(Lsei, min=1e-10))
        crack_flag = 0.5 * (1.0 + torch.tanh((total_surf_stress - sigma_crack_threshold) / 1e5))
        stress_enhancement = 1.0 + 3.0 * crack_flag
    else:
        from stress_models import get_stress_model
        total_surf_stress, stress_enhancement = get_stress_model(stress_model_name)(
            cs_n, Lsei, physics, p, physics.device)

    # ── 2. SEI model ──────────────────────────────────────────────────────────
    # phi_s_n: solid potential for reporting/context only.
    # PyBaMM models compute eta_sei internally using Un and i_n.
    # Builtin model needs phi_s_n for its legacy formula.
    phi_s_n = Un - i_n * (Lsei / p['kappa_sei'])  # local potential [V]

    sei_model_name = getattr(physics, 'sei_model_name', 'builtin')
    if sei_model_name == 'builtin':
        # Original empirical model (stress-enhanced exponential decay)
        i_side_base = (165.96e-6 * torch.exp(-6.3e9*Lsei) *
                       torch.exp(-0.55*p['F']*(phi_s_n-p['Us'])/(p['R_g']*T_cell)) +
                       p['F'] * (3.7398e-15/Lsei) * 0.015 *
                       torch.exp(-Un*p['F']/(p['R_g']*T_cell)))
        i_side   = i_side_base * stress_enhancement            # [A/m²], positive
        dLsei_dt = i_side * p['Msei'] / (2*p['F']*p['rho_sei'])  # [m/s]
    else:
        # PyBaMM-compatible models: return j_sei [A/m²], negative (reduction)
        from sei_models import get_sei_model
        j_sei = get_sei_model(sei_model_name)(Lsei, i_n, Un, T_cell, p, physics.device)
        # Convert j_sei → i_side (positive convention used in rest of code)
        i_side = -j_sei                                         # [A/m²], positive
        # PyBaMM growth formula: dL/dt = -(Vbar_sei / (F * z_sei)) * j_sei
        Vbar_sei = p.get('Vbar_sei', 9.585e-5)                 # m³/mol
        z_sei    = p.get('z_sei',    2.0)                      # electrons/reaction
        dLsei_dt = -(Vbar_sei / (p['F'] * z_sei)) * j_sei      # [m/s], positive

    g_side = p['as_n'] * p['Ln'] * p['A'] * i_side             # total side-rxn current [A]

    # ── 3. Butler-Volmer overpotentials ───────────────────────────────────────
    eta_n = term_RTF * torch.asinh(i_n / (2*j0_n + 1e-12))
    eta_p = term_RTF * torch.asinh(i_p / (2*j0_p + 1e-12))
    V_rxn = eta_n - eta_p

    V_cell = OCV - V_rxn - V_ohm_total - V_conc

    # ── 4. Thermal model ──────────────────────────────────────────────────────
    thermal_model_name = getattr(physics, 'thermal_model_name', 'builtin')
    if thermal_model_name == 'builtin':
        Q_gen  = I_app * (OCV - V_cell)
        Q_cool = p['hA'] * (T_cell - p['T_amb'])
        dT_dt = (Q_gen - Q_cool) / (p['rho_Cp'] * p['Vol_cell'])
    else:
        from temperature_models import get_thermal_model
        dT_dt = get_thermal_model(thermal_model_name)(I_app, OCV, V_cell, T_cell, p, physics.device)

    # ── 5. Lithium plating ────────────────────────────────────────────────────
    # (dispatched inside states/lithium_plating.py via context['phi_s_n'])
    # Override phi_s_n in context with the corrected value computed above.

    flux_n = -(I_app - g_side) / (p['A'] * p['Ln'] * p['F'] * p['as_n'])
    flux_p = I_app / (p['A'] * p['Lp'] * p['F'] * p['as_p'])

    ce_n = ce_real[:physics.Nx_n]
    ce_s = ce_real[physics.Nx_n:physics.Nx_n+physics.Nx_s]
    ce_p = ce_real[-physics.Nx_p:]
    
    def get_Deff(c, eps):
        c_m = c / 1000.0
        return (8.79e-11*c_m**2 - 3.97e-10*c_m + 4.86e-10) * (eps**p['b'])
        
    Deff_n = get_Deff(ce_n, p['eps_e_n'])
    Deff_s = get_Deff(ce_s, p['eps_e_s'])
    Deff_p = get_Deff(ce_p, p['eps_e_p'])

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
    if physics.discretization_methods['electrolyte'] == 'finite_volume':
        flux_coefficient = lambda ctx, coeff=deff_state: ctx.operators.face_coefficients(coeff)
    else:
        flux_coefficient = deff_state

    # This context holds all the calculated parameters (like fluxes and heat generation)
    # so that the STATE_EQUATIONS can fetch them instantly without recalculating.
    context = {
        'ce_state': ce_state, 'ce_real': ce_real, 'electrolyte_flux_coefficient': flux_coefficient,
        'electrolyte_source': src_state, 'flux_n': flux_n, 'flux_p': flux_p,
        'y_old': y_old, 'dt': dt, 'dLsei_dt': dLsei_dt, 'dT_dt': dT_dt,
        'phi_s_n': phi_s_n, 'i_n': i_n,   # i_n correctly normalized by (as_n*A*Ln)
        'deff_state': deff_state, 'src_state': src_state, 'g_side': g_side,
        'total_surf_stress': total_surf_stress,
    }

    dcs_n_out = physics.evaluate_operator_state('cs_n', y_flat, I_app, p, context)
    dcs_p_out = physics.evaluate_operator_state('cs_p', y_flat, I_app, p, context)
    electrolyte_eval = physics.evaluate_operator_state('electrolyte', y_flat, I_app, p, context, return_details=True)

    derivatives = {'cs_n': dcs_n_out, 'cs_p': dcs_p_out, 'electrolyte': electrolyte_eval.rhs}
    context.update({'dcs_n': dcs_n_out, 'dcs_p': dcs_p_out, 'dce': electrolyte_eval.rhs, 'flux_electrolyte': electrolyte_eval.flux})
    
    physics.evaluate_registered_operator_rhs(y_flat, I_app, p, context, derivatives)
    evaluate_registered_rhs(physics, y_flat, I_app, p, context, derivatives)
    return physics.state_layout.pack(derivatives, device=physics.device, dtype=y_flat.dtype)
