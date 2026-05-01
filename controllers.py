import numpy as np
import torch


class BaseController:
    def __init__(self, initial_current=0.0, min_voltage=2.4, max_voltage=4.5, stop_on_voltage_limits=False):
        self.initial_current = float(initial_current)
        self.min_voltage = float(min_voltage)
        self.max_voltage = float(max_voltage)
        self.stop_on_voltage_limits = stop_on_voltage_limits
        self.current_stage = self.__class__.__name__

    def compute_current(self, t, y_state, model_solver, dt_sim):
        raise NotImplementedError

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        max_voltage = torch.max(cell_voltages).item()
        min_voltage = torch.min(cell_voltages).item()
        max_idx = int(torch.argmax(cell_voltages).item())
        min_idx = int(torch.argmin(cell_voltages).item())

        if self.stop_on_voltage_limits and max_voltage >= self.max_voltage:
            return True, (
                f"Stopping controlled simulation: cell {max_idx} reached the "
                f"max voltage limit ({max_voltage:.4f} V, limit {self.max_voltage:.2f} V)."
            )

        if self.stop_on_voltage_limits and min_voltage <= self.min_voltage:
            return True, (
                f"Stopping controlled simulation: cell {min_idx} reached the "
                f"min voltage limit ({min_voltage:.4f} V, limit {self.min_voltage:.2f} V)."
            )

        return False, ""


class ConstantCurrentController(BaseController):
    def __init__(self, current, **kwargs):
        super().__init__(initial_current=current, stop_on_voltage_limits=True, **kwargs)
        self.current = float(current)
        self.current_stage = "CONSTANT_CURRENT"

    def compute_current(self, t, y_state, model_solver, dt_sim):
        return self.current


class CurrentProfileController(BaseController):
    def __init__(self, time_points, current_points, **kwargs):
        initial_current = float(current_points[0]) if len(current_points) > 0 else 0.0
        super().__init__(initial_current=initial_current, stop_on_voltage_limits=True, **kwargs)
        self.time_points = np.asarray(time_points, dtype=float)
        self.current_points = np.asarray(current_points, dtype=float)
        self.current_stage = "CURRENT_PROFILE"

    def compute_current(self, t, y_state, model_solver, dt_sim):
        return float(np.interp(t, self.time_points, self.current_points))


class CCCVController(BaseController):
    def __init__(self, cc_current=-5.0, cv_voltage=4.2, cutoff_current=1.0, kp=40.0, **kwargs):
        super().__init__(initial_current=cc_current, stop_on_voltage_limits=False, **kwargs)
        self.cc_current = float(cc_current)
        self.cv_voltage = float(cv_voltage)
        self.cutoff_current = float(cutoff_current)
        self.kp = float(kp)
        self.current_stage = "CC"

    def compute_current(self, t, y_state, model_solver, dt_sim):
        cell_voltages = model_solver.get_exact_terminal_voltages(y_state, self.cc_current)
        max_voltage = torch.max(cell_voltages).item()

        if self.current_stage == "CC" and max_voltage >= self.cv_voltage:
            print(f"[{t:.1f}s] Switching to CV mode.")
            self.current_stage = "CV"

        if self.current_stage == "CC":
            return self.cc_current

        voltage_error = self.cv_voltage - max_voltage
        current_magnitude = min(abs(self.cc_current), max(0.0, self.kp * max(voltage_error, 0.0)))
        return -current_magnitude

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        min_voltage = torch.min(cell_voltages).item()
        min_idx = int(torch.argmin(cell_voltages).item())

        if min_voltage <= self.min_voltage:
            return True, (
                f"Stopping controlled simulation: cell {min_idx} reached the "
                f"min voltage limit ({min_voltage:.4f} V, limit {self.min_voltage:.2f} V)."
            )

        if self.current_stage == "CV" and abs(pack_current) <= self.cutoff_current:
            return True, f"Charge complete at {t:.1f}s. CV current dropped below {self.cutoff_current:.2f} A."

        return False, ""


class MPCBatteryController(BaseController):
    def __init__(self, n_parallel, t_limit=313.15, target_c_rate=2.0, cutoff_v=2.5, **kwargs):
        super().__init__(initial_current=0.0, stop_on_voltage_limits=False, **kwargs)
        self.n_parallel = n_parallel
        self.t_limit = t_limit
        self.max_i = 5.0 * target_c_rate * n_parallel
        self.last_i = 0.0
        self.cutoff_v = cutoff_v
        self.current_stage = "DISCHARGE"
        self._setup_mpc()

    def _setup_mpc(self):
        try:
            import do_mpc
        except ImportError as exc:
            raise ImportError("MPC strategy requires 'do_mpc' to be installed.") from exc

        model_type = 'discrete'
        self.model = do_mpc.model.Model(model_type)

        s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))
        T = self.model.set_variable(var_type='_x', var_name='T', shape=(1, 1))
        L = self.model.set_variable(var_type='_x', var_name='L', shape=(1, 1))
        i = self.model.set_variable(var_type='_u', var_name='i')

        c_cap = 3600 * 3.0 * self.n_parallel
        r_int = 0.02 / self.n_parallel
        mc_p = 40.0 * self.n_parallel
        hA = 0.5
        t_amb = 298.15
        dt = 5.0

        self.model.set_rhs('s', s + (i / c_cap) * dt)
        self.model.set_rhs('T', T + ((i**2 * r_int - hA * (T - t_amb)) / mc_p) * dt)
        sei_growth = 1e-12 * np.exp((T - 273.15) / 15) * (i / self.n_parallel)
        self.model.set_rhs('L', L + sei_growth * dt)
        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(n_horizon=12, t_step=5.0, n_robust=1)
        self.mpc.set_objective(mterm=-s, lterm=1e9 * L + 0.5 * (T - 298.15) ** 2)
        self.mpc.set_rterm(i=1e-2)
        self.mpc.bounds['lower', '_u', 'i'] = 0.0
        self.mpc.bounds['upper', '_u', 'i'] = self.max_i
        self.mpc.bounds['upper', '_x', 'T'] = self.t_limit
        self.mpc.setup()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)

    def compute_current(self, t, y_state, model_solver, dt_sim):
        exact_v_cells = model_solver.get_exact_terminal_voltages(y_state, self.last_i)
        min_v = torch.min(exact_v_cells).item()

        if self.current_stage == "DISCHARGE":
            if min_v <= self.cutoff_v:
                print(f"[{t:.1f}s] Battery empty. Handing over to MPC optimizer.")
                self.current_stage = "MPC_CHARGE"
            current = self.max_i * 0.3
            self.last_i = current
            return current

        current_soc = torch.mean(y_state[:, 0] / 33133.0).item()
        current_temp = torch.max(y_state[:, -1]).item()
        current_sei = torch.mean(y_state[:, -2]).item()

        x0 = np.array([[current_soc], [current_temp], [current_sei]])
        self.mpc.x0 = x0
        self.estimator.x0 = x0
        self.mpc.set_initial_guess()

        u0 = self.mpc.make_step(x0)
        self.last_i = -float(u0[0])
        return self.last_i

    def should_stop(self, t, y_state, cell_voltages, pack_current):
        current_soc = torch.mean(y_state[:, 0] / 33133.0).item()
        if current_soc > 0.99:
            return True, f"Charge complete at {t:.1f}s | SOC: {current_soc * 100:.1f}%"
        return False, ""


def build_controller(strategy, **kwargs):
    strategy = strategy.lower()

    if strategy == "constant_current":
        return ConstantCurrentController(**kwargs)
    if strategy == "current_profile":
        return CurrentProfileController(**kwargs)
    if strategy == "cc_cv":
        return CCCVController(**kwargs)
    if strategy == "mpc":
        return MPCBatteryController(**kwargs)

    raise ValueError(
        "Unknown controller strategy. Use 'constant_current', 'current_profile', 'cc_cv', or 'mpc'."
    )
