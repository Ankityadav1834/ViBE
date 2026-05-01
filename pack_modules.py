import numpy as np
import torch


class BalancingStrategy:
    name = "none"

    def compute_balancing_currents(self, cell_voltages, string_currents):
        return torch.zeros_like(cell_voltages)


class NoBalancing(BalancingStrategy):
    name = "none"


class PassiveBalancing(BalancingStrategy):
    name = "passive"

    def __init__(self, r_bleed=10.0, v_threshold=4.0, **kwargs):
        self.r_bleed = float(r_bleed)
        self.v_threshold = float(v_threshold)

    def compute_balancing_currents(self, cell_voltages, string_currents):
        v_excess = torch.nn.functional.softplus((cell_voltages - self.v_threshold) * 50.0) / 50.0
        is_charging = torch.sigmoid(-string_currents * 1000.0)
        return (v_excess / self.r_bleed) * is_charging


class ActiveCapacitorBalancing(BalancingStrategy):
    name = "active_capacitor"

    def __init__(self, r_eq=0.5, **kwargs):
        self.r_eq = float(r_eq)

    def compute_balancing_currents(self, cell_voltages, string_currents):
        i_bal = torch.zeros_like(cell_voltages)
        for series_idx in range(cell_voltages.shape[0]):
            row = cell_voltages[series_idx]
            row_bal = torch.zeros_like(row)
            for i in range(row.shape[0] - 1):
                i_transfer = (row[i] - row[i + 1]) / self.r_eq
                row_bal[i] -= i_transfer
                row_bal[i + 1] += i_transfer
            i_bal[series_idx] = row_bal
        return i_bal


class ActiveInductorBalancing(BalancingStrategy):
    name = "active_inductor"

    def __init__(self, transfer_gain=2.0, **kwargs):
        self.transfer_gain = float(transfer_gain)

    def compute_balancing_currents(self, cell_voltages, string_currents):
        v_avg = torch.mean(cell_voltages, dim=1, keepdim=True)
        return self.transfer_gain * (v_avg - cell_voltages)


class ThermalStrategy:
    name = "ambient"

    def compute_temperature_rhs(self, battery, temperatures, heat_generation):
        raise NotImplementedError


class AmbientConvection(ThermalStrategy):
    name = "ambient"

    def __init__(self, hA_scale=1.0, ambient_temp=None, **kwargs):
        self.hA_scale = float(hA_scale)
        self.ambient_temp = ambient_temp

    def compute_temperature_rhs(self, battery, temperatures, heat_generation):
        t_amb = battery.T_amb_vec if self.ambient_temp is None else torch.full_like(temperatures, self.ambient_temp)
        q_cond = battery.Gth @ temperatures - torch.sum(battery.Gth, dim=1, keepdim=True) * temperatures
        q_conv = (battery.h_amb * self.hA_scale) * (temperatures - t_amb)
        return (q_cond - q_conv) / (battery.Cth + 1e-12)


class ActiveLiquidCooling(ThermalStrategy):
    name = "liquid"

    def __init__(self, hA_contact=2.5, m_dot_cp=5.0, inlet_temp=298.15, **kwargs):
        self.hA_contact = float(hA_contact)
        self.m_dot_cp = float(m_dot_cp)
        self.inlet_temp = float(inlet_temp)

    def compute_temperature_rhs(self, battery, temperatures, heat_generation):
        t_grid = temperatures.view(battery.n_series, battery.n_parallel)
        fluid_grid = torch.zeros_like(t_grid)
        effectiveness = 1.0 - np.exp(-self.hA_contact / max(self.m_dot_cp, 1e-12))

        for parallel_idx in range(battery.n_parallel):
            fluid_temp = torch.tensor(self.inlet_temp, device=temperatures.device, dtype=temperatures.dtype)
            for series_idx in range(battery.n_series):
                fluid_grid[series_idx, parallel_idx] = fluid_temp
                fluid_temp = fluid_temp + (t_grid[series_idx, parallel_idx] - fluid_temp) * effectiveness

        t_fluid = fluid_grid.reshape(-1, 1)
        q_cond = battery.Gth @ temperatures - torch.sum(battery.Gth, dim=1, keepdim=True) * temperatures
        q_cool = self.hA_contact * (temperatures - t_fluid)
        return (q_cond - q_cool) / (battery.Cth + 1e-12)


class PCMCooling(ThermalStrategy):
    name = "pcm"

    def __init__(
        self,
        hA_ambient=0.05,
        ambient_temp=298.15,
        melt_temp=308.15,
        latent_heat=50000.0,
        smoothing_width=1.5,
        **kwargs
    ):
        self.hA_ambient = float(hA_ambient)
        self.ambient_temp = float(ambient_temp)
        self.melt_temp = float(melt_temp)
        self.latent_heat = float(latent_heat)
        self.smoothing_width = float(smoothing_width)

    def compute_temperature_rhs(self, battery, temperatures, heat_generation):
        delta_smooth = torch.exp(-((temperatures - self.melt_temp) / self.smoothing_width) ** 2)
        delta_smooth = delta_smooth / (self.smoothing_width * np.sqrt(np.pi))
        c_eff = battery.Cth + self.latent_heat * delta_smooth

        q_cond = battery.Gth @ temperatures - torch.sum(battery.Gth, dim=1, keepdim=True) * temperatures
        q_conv = self.hA_ambient * (temperatures - self.ambient_temp)
        return (q_cond - q_conv) / (c_eff + 1e-12)


def build_balancing_module(enabled=False, strategy="none", **kwargs):
    if not enabled or strategy == "none":
        return NoBalancing()

    strategy = strategy.lower()
    if strategy == "passive":
        return PassiveBalancing(**kwargs)
    if strategy == "active_capacitor":
        return ActiveCapacitorBalancing(**kwargs)
    if strategy == "active_inductor":
        return ActiveInductorBalancing(**kwargs)
    raise ValueError("Unknown balancing strategy. Use 'none', 'passive', 'active_capacitor', or 'active_inductor'.")


def build_thermal_module(enabled=False, strategy="ambient", **kwargs):
    if not enabled:
        return AmbientConvection()

    strategy = strategy.lower()
    if strategy == "ambient":
        return AmbientConvection(**kwargs)
    if strategy == "liquid":
        return ActiveLiquidCooling(**kwargs)
    if strategy == "pcm":
        return PCMCooling(**kwargs)
    raise ValueError("Unknown thermal strategy. Use 'ambient', 'liquid', or 'pcm'.")
