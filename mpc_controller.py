import numpy as np
from scipy.optimize import minimize
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Thin base so mpc_controller.py doesn't import from controllers.py (avoids
# circular import: controllers → mpc_controller → controllers).
# FastChargeMPCController re-implements the two interface methods directly.
# ─────────────────────────────────────────────────────────────────────────────

class _MpcBase:
    """Minimal BaseController-compatible interface used by the simulator."""
    def __init__(self, initial_current=0.0, min_voltage=2.4, max_voltage=4.5,
                 stop_on_voltage_limits=False):
        self.initial_current = float(initial_current)
        self.min_voltage = float(min_voltage)
        self.max_voltage = float(max_voltage)
        self.stop_on_voltage_limits = stop_on_voltage_limits
        self.current_stage = "MPC_FAST_CHARGE"

    def compute_current(self, t, y_state, model_solver, dt_sim):
        raise NotImplementedError

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        max_v = torch.max(cell_voltages).item()
        min_v = torch.min(cell_voltages).item()
        if self.stop_on_voltage_limits:
            if max_v >= self.max_voltage:
                return True, f"Max voltage {max_v:.4f} V reached."
            if min_v <= self.min_voltage:
                return True, f"Min voltage {min_v:.4f} V reached."
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: scalar OCP and physical side-reaction current (numpy, for ROM)
# ─────────────────────────────────────────────────────────────────────────────

def _ocp_anode_np(s: float) -> float:
    s = float(np.clip(s, 0.001, 0.999))
    return (1.9793 * np.exp(-39.3631*s) + 0.2482
            - 0.0909 * np.tanh(29.8538*(s - 0.1234))
            - 0.04478 * np.tanh(14.9159*(s - 0.2769))
            - 0.0205 * np.tanh(30.4444*(s - 0.6103)))


def _ocp_cathode_np(s: float) -> float:
    s = float(np.clip(s, 0.001, 0.999))
    return (-0.8090*s + 4.4875
            - 0.0428 * np.tanh(18.5138*(s - 0.5542))
            - 17.7326 * np.tanh(15.7890*(s - 0.3117))
            + 17.5842 * np.tanh(15.9308*(s - 0.3120)))


def _sei_side_current(phi_s_n: float, L_sei: float, T: float, p: dict) -> float:
    """
    SEI side-reaction current density [A/m²].

    Two components (same formula as in electrochemistry.py):
      1. Tafel leakage through the existing SEI film.
      2. Solvent-reduction limited by film thickness.
    """
    F, R_g = p['F'], p['R_g']
    Us = p['Us']
    i_tafel = 165.96e-6 * np.exp(-6.3e9 * L_sei) * np.exp(
        -0.55 * F * (phi_s_n - Us) / (R_g * T))
    U_n = _ocp_anode_np(0.5)   # rough reference; fine for constraint estimation
    i_solvent = F * (3.7398e-15 / max(L_sei, 1e-15)) * 0.015 * np.exp(
        -U_n * F / (R_g * T))
    return float(i_tafel + i_solvent)


def _stress_hydrostatic(soc: float, soc_avg_ref: float, p_mpc: dict) -> float:
    """
    Heuristic hydrostatic surface stress [Pa].

    The stress is proportional to the intercalation strain misfit between the
    surface and the particle average. During fast charging, the surface lithiates
    faster than the core, creating compressive (negative) surface stress.

    We use a simple lumped model:
        c_surf ≈ cs_max_n * soc
        c_avg  ≈ cs_max_n * soc_avg_ref   (lagged by diffusion time)
        sigma_th ≈ E * Omega / (3*(1-nu)) * (c_avg - c_surf)
    """
    E_g = p_mpc.get('E_g', 15e9)
    nu_g = p_mpc.get('nu_g', 0.3)
    Omega = p_mpc.get('Omega', 3.17e-6)
    cs_max = p_mpc['cs_max_n']

    pf = E_g * Omega / (3.0 * (1.0 - nu_g))
    c_surf = cs_max * soc
    c_avg  = cs_max * soc_avg_ref   # slow-moving average
    return float(pf * (c_avg - c_surf))


# ─────────────────────────────────────────────────────────────────────────────
# ROM state propagation
# ─────────────────────────────────────────────────────────────────────────────

def _rom_step(state: dict, I_charge: float, dt: float, p: dict, p_mpc: dict) -> dict:
    """
    Propagate ROM state forward by dt seconds at charging current I_charge [A].

    state keys:  soc, soc_avg, T, L_sei
    Returns:     new state dict + scalar observations (V_term, sigma_surf, dL_sei)
    """
    soc      = state['soc']
    soc_avg  = state['soc_avg']
    T        = state['T']
    L_sei    = state['L_sei']

    F    = p['F']
    R_g  = p['R_g']
    Q_nom = p_mpc['Q_nom']        # nominal capacity [A·s]

    # ── 1. SOC ──────────────────────────────────────────────────────────────
    # Charging is negative current convention in this simulator;
    # we receive positive I_charge as a charging magnitude.
    dsoc_dt = I_charge / Q_nom   # positive because charging
    soc_new = float(np.clip(soc + dsoc_dt * dt, 0.0, 1.0))

    # slow-moving average (lumped diffusion lag, tau ~ Rs²/Ds)
    tau_diff = p_mpc.get('tau_diff_n', 300.0)   # ~5 min lag for Nr=10
    alpha = dt / (dt + tau_diff)
    soc_avg_new = float(soc_avg + alpha * (soc_new - soc_avg))

    # ── 2. Stoichiometry ─────────────────────────────────────────────────────
    theta_n_0 = p_mpc.get('theta_n_0', 0.0)   # stoich at SOC=0 (empty)
    theta_n_1 = p_mpc.get('theta_n_1', 0.85)  # stoich at SOC=1 (full)
    theta_p_0 = p_mpc.get('theta_p_0', 0.95)
    theta_p_1 = p_mpc.get('theta_p_1', 0.45)

    theta_n = theta_n_0 + soc_new * (theta_n_1 - theta_n_0)
    theta_p = theta_p_0 + soc_new * (theta_p_1 - theta_p_0)

    Un = _ocp_anode_np(theta_n)
    Up = _ocp_cathode_np(theta_p)
    OCV = Up - Un

    # ── 3. Terminal voltage (ECM) ─────────────────────────────────────────────
    arr_n = np.exp(p['E_r_n'] / R_g * (1.0/298.15 - 1.0/T))
    arr_p = np.exp(p['E_r_p'] / R_g * (1.0/298.15 - 1.0/T))

    # ROM exchange-current densities (use scalar ce=1000 mol/m³ as reference)
    ce_ref = 1000.0
    j0_n_ref = p['m_ref_n'] * arr_n * np.sqrt(ce_ref)
    j0_p_ref = p['m_ref_p'] * arr_p * np.sqrt(ce_ref)

    # Include surface cs dependence
    cs_n_surf = theta_n * p['cs_max_n']
    cs_p_surf = theta_p * p['cs_max_p']
    j0_n = j0_n_ref * float(np.sqrt(max(cs_n_surf * (p['cs_max_n'] - cs_n_surf), 1e-12)))
    j0_p = j0_p_ref * float(np.sqrt(max(cs_p_surf * (p['cs_max_p'] - cs_p_surf), 1e-12)))

    I0_n = j0_n * p['A'] * p['Ln'] * p['as_n']
    I0_p = j0_p * p['A'] * p['Lp'] * p['as_p']

    # SEI & solid ohmic resistances
    R_sei  = L_sei / (p['as_n'] * p['A'] * p['Ln'] * p['kappa_sei'])
    R_solid = (p['Ln'] / p['sigma_n'] + p['Lp'] / p['sigma_p']) / (3.0 * p['A'])
    R_total = p_mpc.get('R_elec_ref', 0.005) + R_solid + R_sei   # electrolyte + ohmic

    RTF = 2.0 * R_g * T / F
    # Butler-Volmer over-potentials (charge = negative current in PDE = positive I_charge)
    eta_n = RTF * np.arcsinh((-I_charge) / (2.0 * I0_n + 1e-12))
    eta_p = RTF * np.arcsinh(I_charge  / (2.0 * I0_p + 1e-12))
    V_rxn = eta_n - eta_p

    V_term = OCV - V_rxn - I_charge * R_total

    # ── 4. Temperature ───────────────────────────────────────────────────────
    hA     = p['hA']
    rho_Cp = p['rho_Cp']
    Vol    = p['Vol_cell']
    T_amb  = p.get('T_amb', 298.15)

    V_cell = OCV - V_rxn - I_charge * R_solid   # approximation for heat
    Q_irr  = I_charge * (OCV - V_cell)           # irreversible heat [W]
    Q_cool = hA * (T - T_amb)
    dT_dt  = (Q_irr - Q_cool) / (rho_Cp * Vol)
    T_new  = float(T + dT_dt * dt)

    # ── 5. SEI growth ────────────────────────────────────────────────────────
    # Anode half-potential for SEI Tafel kinetics
    phi_s_n = Un - (-I_charge) * R_sei   # charging current is negative in anode convention
    i_side  = _sei_side_current(float(phi_s_n), float(L_sei), float(T), p)
    dL_dt   = i_side * p['Msei'] / (2.0 * F * p['rho_sei'])
    L_sei_new = float(L_sei + dL_dt * dt)

    # ── 6. Surface stress ────────────────────────────────────────────────────
    sigma_surf = _stress_hydrostatic(soc_new, soc_avg_new, {**p, **p_mpc})

    return (
        {'soc': soc_new, 'soc_avg': soc_avg_new, 'T': T_new, 'L_sei': L_sei_new},
        {
            'V_term': V_term,
            'OCV': OCV,
            'T': T_new,
            'sigma_surf': sigma_surf,
            'dL_sei': dL_dt * dt,
            'I0_n': I0_n,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# MPC cost and constraint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _mpc_evaluate(u_seq: np.ndarray, state0: dict, dt: float,
                  p: dict, p_mpc: dict) -> tuple:
    """
    Rollout u_seq (shape Np,) starting from state0.
    Returns (cost, list_of_obs_dicts).
    Constraints are embedded as penalty so scipy can use gradient-based solvers.
    """
    state = dict(state0)
    Np = len(u_seq)
    obs_list = []
    cost = 0.0

    w_time    = p_mpc.get('w_time', 1.0)
    w_sei     = p_mpc.get('w_sei', 5e9)
    w_stress  = p_mpc.get('w_stress', 1e-16)
    w_temp    = p_mpc.get('w_temp', 0.5)
    w_dcurr   = p_mpc.get('w_dcurr', 1e-3)

    V_max     = p_mpc.get('V_max', 4.2)
    T_max     = p_mpc.get('T_max', 318.15)   # 45 °C
    sigma_max = p_mpc.get('sigma_max', 200e6) # 200 MPa compressive limit

    penalty_hard = p_mpc.get('penalty_hard', 1e6)

    prev_I = state0.get('last_I', u_seq[0])

    for k in range(Np):
        I_k = u_seq[k]
        state, obs = _rom_step(state, I_k, dt, p, p_mpc)
        obs_list.append(obs)

        # ── objective terms ──────────────────────────────────────────────────
        # Minimise: charge time proxy (-SOC progress), SEI growth, temp rise, jerk
        cost += w_time   * (1.0 - state['soc'])        # maximise SOC gain
        cost += w_sei    * obs['dL_sei']                # penalise SEI growth
        cost += w_temp   * max(0.0, state['T'] - 298.15)  # penalise heating
        cost += w_stress * obs['sigma_surf']**2         # penalise stress
        cost += w_dcurr  * (I_k - prev_I)**2           # penalise current slew
        prev_I = I_k

        # ── hard constraint penalties ─────────────────────────────────────────
        if obs['V_term'] > V_max:
            cost += penalty_hard * (obs['V_term'] - V_max)**2
        if state['T'] > T_max:
            cost += penalty_hard * (state['T'] - T_max)**2
        if abs(obs['sigma_surf']) > sigma_max:
            cost += penalty_hard * (abs(obs['sigma_surf']) - sigma_max)**2

    return -cost, obs_list   # negate because scipy minimises


# ─────────────────────────────────────────────────────────────────────────────
# Fast-Charge MPC Controller
# ─────────────────────────────────────────────────────────────────────────────

class FastChargeMPCController(_MpcBase):
    """
    MPC Controller for battery fast charging.

    Parameters
    ----------
    n_parallel : int
        Number of parallel cells in the pack (scales capacity and current).
    Q_nom_cell : float
        Nominal capacity of a single cell in Ah. Default: 3 Ah.
    I_max_rate : float
        Maximum C-rate for charging. Default: 3 (i.e. 3C).
    I_min_rate : float
        Minimum C-rate (keeps some minimum current). Default: 0.1.
    v_max : float
        Per-cell voltage upper limit [V]. Default: 4.2 V.
    t_max_degC : float
        Maximum cell temperature in °C. Default: 45 °C.
    sigma_max_MPa : float
        Maximum absolute surface stress [MPa]. Default: 200 MPa.
    soc_target : float
        Target SOC to reach (0–1). Default: 0.95.
    Np : int
        MPC prediction horizon (number of steps). Default: 10.
    dt_mpc : float
        Prediction step size [s] (can differ from simulator dt). Default: 10 s.
    """

    def __init__(
        self,
        n_parallel: int = 1,
        Q_nom_cell: float = 3.0,        # Ah per cell
        I_max_rate: float = 3.0,         # C-rate
        I_min_rate: float = 0.1,         # C-rate
        v_max: float = 4.2,
        t_max_degC: float = 45.0,
        sigma_max_MPa: float = 200.0,
        soc_target: float = 0.95,
        Np: int = 10,
        dt_mpc: float = 10.0,
        **kwargs
    ):
        super().__init__(initial_current=0.0, min_voltage=kwargs.get('min_voltage', 2.4),
                         max_voltage=kwargs.get('max_voltage', 4.5),
                         stop_on_voltage_limits=False)
        self.n_parallel   = n_parallel
        self.Q_nom_cell   = Q_nom_cell   # Ah
        self.I_max        = I_max_rate * Q_nom_cell * n_parallel   # A (pack)
        self.I_min        = I_min_rate * Q_nom_cell * n_parallel   # A (pack)
        self.v_max        = v_max
        self.t_max        = t_max_degC + 273.15
        self.sigma_max    = sigma_max_MPa * 1e6
        self.soc_target   = soc_target
        self.Np           = Np
        self.dt_mpc       = dt_mpc
        self.current_stage = "MPC_FAST_CHARGE"

        # warm-start: initialise to 1C current
        self._u_prev = np.full(Np, Q_nom_cell * n_parallel)
        self._last_I = float(Q_nom_cell * n_parallel)
        self._rom_state: dict | None = None
        self._p: dict | None = None
        self._p_mpc: dict | None = None
        self._solve_count = 0
        self._opt_verbose = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def compute_current(self, t, y_state, model_solver, dt_sim):
        """Called by the simulator at each control step."""

        # Lazy initialise physics parameters from the solver
        if self._p is None:
            self._init_params(model_solver)

        # Extract observable state from the full PDE solution
        rom_state = self._observe(y_state, model_solver)
        self._rom_state = rom_state

        # Run MPC optimisation
        I_opt = self._solve_mpc(rom_state)
        self._last_I = I_opt
        rom_state['last_I'] = I_opt

        soc_pct = rom_state['soc'] * 100.0
        T_degC  = rom_state['T'] - 273.15
        L_nm    = rom_state['L_sei'] * 1e9
        print(f"  [MPC] t={t:.0f}s | SOC={soc_pct:.1f}% | "
              f"I={I_opt:.2f}A | T={T_degC:.1f}°C | Lsei={L_nm:.2f}nm")

        # Return as negative (convention: charging is negative in this simulator)
        return -I_opt

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        if self._rom_state is not None and self._rom_state['soc'] >= self.soc_target:
            return True, f"[MPC] Charge complete at t={t:.0f}s. SOC={self._rom_state['soc']*100:.1f}%"
        max_v = torch.max(cell_voltages).item()
        if max_v >= self.v_max + 0.05:  # hard safety margin
            return True, f"[MPC] Emergency stop: cell voltage {max_v:.3f} V."
        return False, ""

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_params(self, model_solver):
        """Pull physics parameters directly from the BatteryPhysics object."""
        p_raw = model_solver.raw_params[0]   # use cell-0 params
        self._p = dict(p_raw)

        # Derived quantities for the ROM
        Q_nom_As = self.Q_nom_cell * self.n_parallel * 3600.0   # A·s (pack level)

        # Estimate electrolyte resistance at reference ce=1000 mol/m³
        # (approx R_elec at SoC~50 %, good enough for MPC ROM)
        kappa_ref = 3.329 * (1.0)**1.5 - 2.51 * (1.0) + 0.1297  # ≈ 1.0 S/m
        R_elec_ref = (
            (p_raw['Ln'] / (3 * p_raw['eps_e_n']**p_raw['b']) +
             p_raw['Ls'] / (p_raw['eps_e_s']**p_raw['b']) +
             p_raw['Lp'] / (3 * p_raw['eps_e_p']**p_raw['b']))
            / (p_raw['A'] * (kappa_ref + 1e-9))
        )

        self._p_mpc = {
            'Q_nom':        Q_nom_As,
            'R_elec_ref':   R_elec_ref,
            'tau_diff_n':   (model_solver.physics.r_n_ref[-1].item()**2) / (
                             6.0 * p_raw['Ds_n'] + 1e-20),   # Rs²/(6Ds) [s]
            # stoichiometric window (typical graphite / NMC)
            'theta_n_0':    0.0,
            'theta_n_1':    0.85,
            'theta_p_0':    0.95,
            'theta_p_1':    0.45,
            # stress material constants (same as in electrochemistry.py)
            'E_g':          15e9,
            'nu_g':         0.3,
            'Omega':        3.17e-6,
            'cs_max_n':     p_raw['cs_max_n'],
            # MPC tuning weights
            'w_time':       1.0,
            'w_sei':        6e9,
            'w_stress':     1e-16,
            'w_temp':       0.3,
            'w_dcurr':      2e-4,
            'penalty_hard': 5e6,
            # constraints
            'V_max':        self.v_max,
            'T_max':        self.t_max,
            'sigma_max':    self.sigma_max,
        }

    def _observe(self, y_state, model_solver) -> dict:
        """Extract reduced-state from the full PDE solution."""
        with torch.no_grad():
            physics = model_solver.physics

            # physics.state() returns raw physical values (mol/m³, K, m, etc.)
            # cs_n shape: (n_cells, Nr_n)  units: mol/m³
            cs_n_real = physics.state(y_state, 'cs_n')

            # Surface stoichiometry (last Chebyshev node = particle surface)
            theta_n_surf = cs_n_real[:, -1] / float(self._p['cs_max_n'])
            # In graphite, theta_n at full charge ≈ 0.85 (high Li), at empty ≈ 0.03
            # SOC maps linearly: soc = (theta_n - theta_n_empty)/(theta_n_full - theta_n_empty)
            th_empty = self._p_mpc.get('theta_n_0', 0.0)   # ~0 for graphite discharged
            th_full  = self._p_mpc.get('theta_n_1', 0.85)
            soc = float(torch.mean(
                torch.clamp((theta_n_surf - th_empty) / (th_full - th_empty + 1e-9), 0.0, 1.0)
            ).item())

            # Average stoichiometry (diffusion lag proxy)
            theta_n_avg = cs_n_real.mean(dim=1) / float(self._p['cs_max_n'])
            soc_avg = float(torch.mean(
                torch.clamp((theta_n_avg - th_empty) / (th_full - th_empty + 1e-9), 0.0, 1.0)
            ).item())

            # Temperature [K] — clamped for safety
            T_raw = physics.state(y_state, 'temperature')
            T = float(torch.max(T_raw).item())
            T = max(270.0, min(T, 400.0))

            # SEI thickness [m]
            if physics.sei_enabled:
                L_raw = physics.state(y_state, 'Lsei')
                L_sei = float(torch.mean(L_raw).item())
                L_sei = max(float(self._p.get('Lsei_0', 5e-9)), L_sei)
            else:
                L_sei = float(self._p.get('Lsei_0', 5e-9))

        return {
            'soc':     soc,
            'soc_avg': soc_avg,
            'T':       T,
            'L_sei':   L_sei,
            'last_I':  self._last_I,
        }

    def _solve_mpc(self, state0: dict) -> float:
        """Run the receding-horizon optimisation and return I_0 [A]."""

        def objective(u_seq):
            cost, _ = _mpc_evaluate(u_seq, state0, self.dt_mpc, self._p, self._p_mpc)
            return cost   # negate inside _mpc_evaluate already negated → minimise

        bounds = [(self.I_min, self.I_max)] * self.Np

        result = minimize(
            objective,
            x0=self._u_prev,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 80, 'ftol': 1e-6, 'disp': self._opt_verbose},
        )

        if result.success or result.fun < _mpc_evaluate(self._u_prev, state0, self.dt_mpc, self._p, self._p_mpc)[0]:
            u_opt = result.x
        else:
            u_opt = self._u_prev   # fall back to previous solution

        # Warm-start: shift sequence and repeat last element
        self._u_prev = np.roll(u_opt, -1)
        self._u_prev[-1] = u_opt[-1]
        self._solve_count += 1

        return float(np.clip(u_opt[0], self.I_min, self.I_max))
