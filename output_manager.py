import pandas as pd
import torch


class SimulationOutputManager:
    """
    Central place for saving raw states and derived outputs.

    States come from StateLayout automatically. Derived outputs are registered
    as functions so future PDE outputs can be added without touching solver code.
    """

    def __init__(self, battery_solver):
        self.battery = battery_solver
        self.outputs = {}
        self._register_default_outputs()

    def register_output(self, name, fn):
        self.outputs[name] = fn

    def _register_default_outputs(self):
        self.register_output("terminal_voltage", self._terminal_voltage)
        self.register_output("cell_current", self._cell_current)
        self.register_output("pack_voltage", self._pack_voltage)
        self.register_output("temperature", lambda y, i: self.battery.physics.state(y, "temperature").flatten())
        self.register_output("soc", self._soc)
        self.register_output("sei_thickness_nm", lambda y, i: self.battery.physics.state(y, "Lsei").flatten() * 1e9)
        self.register_output("capacity_fade_pct", self._capacity_fade_pct)
        self.register_output("dis_stress_vm_peak", self._dis_stress_vm_peak)
        self.register_output("dis_stress_th_surf", self._dis_stress_th_surf)
        self.register_output("sei_mismatch_stress", self._sei_mismatch_stress)
        self.register_output("total_surf_stress", self._total_surf_stress)

        if "stress" in self.battery.physics.state_layout:
            self.register_output("stress_mean", lambda y, i: torch.mean(self.battery.physics.state(y, "stress"), dim=1))
            self.register_output("stress_min", lambda y, i: torch.min(self.battery.physics.state(y, "stress"), dim=1).values)
            self.register_output("stress_max", lambda y, i: torch.max(self.battery.physics.state(y, "stress"), dim=1).values)
            self.register_output("force_from_stress", self._force_from_stress)

    def _cell_current(self, y, i_pack):
        if torch.is_tensor(i_pack):
            return i_pack.view(self.battery.n_cells)
        return self.battery.compute_effective_cell_currents(y, float(i_pack)).flatten()

    def _terminal_voltage(self, y, i_pack):
        current = self._cell_current(y, i_pack).view(self.battery.n_cells, 1)
        return self.battery.get_exact_terminal_voltages(y, current)

    def _pack_voltage(self, y, i_pack):
        cell_voltages = self._terminal_voltage(y, i_pack).view(self.battery.n_series, self.battery.n_parallel)
        return torch.sum(torch.mean(cell_voltages, dim=1)).reshape(1)

    def _soc(self, y, i_pack):
        cs_n = self.battery.physics.state(y, "cs_n")
        cs_max = self.battery.physics.params["cs_max_n"]
        return ((cs_n[:, -1:] / cs_max - 0.01) / 0.94).flatten()

    def _capacity_fade_pct(self, y, i_pack):
        Lsei = self.battery.physics.state(y, "Lsei")
        p = self.battery.physics.params
        rho_sei = p["rho_sei"]
        Msei = p["Msei"]
        cs_max = p["cs_max_n"]
        Rs_n = p["Rs_n"]
        Lsei_0 = p["Lsei_0"]
        delta_Lsei = Lsei - Lsei_0
        
        # Capacity fade fraction from SEI Li consumption: 
        # fade = 2 * (rho_sei / Msei) * delta_Lsei * (3 / Rs_n) / cs_max
        fade = 2.0 * (rho_sei / Msei) * delta_Lsei * (3.0 / Rs_n) / cs_max
        return (fade * 100.0).flatten()

    def _get_c_bar_data(self, y):
        cs_n = self.battery.physics.state(y, "cs_n")
        r_ref = self.battery.physics.r_n_ref.to(self.battery.device).unsqueeze(0)
        
        # Construct r_faces to compute spherical volume elements
        r_faces = torch.zeros((1, r_ref.shape[1] + 1), device=r_ref.device)
        r_faces[:, 1:-1] = 0.5 * (r_ref[:, :-1] + r_ref[:, 1:])
        r_faces[:, -1] = 1.0
        
        v = r_faces[:, 1:]**3 - r_faces[:, :-1]**3
        cv = torch.cumsum(v, dim=1)
        cv2 = torch.cumsum(cs_n * v, dim=1)
        
        c_bar_r = cv2 / cv
        c_bar = c_bar_r[:, -1:]
        return cs_n, c_bar_r, c_bar

    def _dis_stress_vm_peak(self, y, i_pack):
        cs_n, c_bar_r, c_bar = self._get_c_bar_data(y)
        
        E = 15e9
        nu = 0.3
        Omega = 3.17e-6
        pf = E * Omega / (3.0 * (1.0 - nu))
        
        sr = 2.0 * pf * (c_bar - c_bar_r)
        sth = pf * (2.0 * c_bar + c_bar_r - 3.0 * cs_n)
        svm = torch.abs(sr - sth)
        return torch.max(svm, dim=1).values.flatten()

    def _dis_stress_th_surf(self, y, i_pack):
        cs_n, c_bar_r, c_bar = self._get_c_bar_data(y)
        
        E = 15e9
        nu = 0.3
        Omega = 3.17e-6
        pf = E * Omega / (3.0 * (1.0 - nu))
        
        c_surf = cs_n[:, -1]
        c_bar_surf = c_bar.squeeze(1)
        
        sth_surf = pf * (2.0 * c_bar_surf + c_bar_surf - 3.0 * c_surf)
        return sth_surf.flatten()

    def _sei_mismatch_stress(self, y, i_pack):
        cs_n = self.battery.physics.state(y, "cs_n")
        c_surf = cs_n[:, -1]
        c_max = self.battery.physics.params["cs_max_n"].flatten()
        c_surf_ref = 0.8 * c_max # Reference concentration at formation
        
        E_sei = 10e9
        nu_sei = 0.25
        E_sei_b = E_sei / (1.0 - nu_sei)
        Omega = 3.17e-6
        sigma_intr = -0.5e9
        
        delta_c = c_surf - c_surf_ref
        sigma_mism = E_sei_b * (Omega / 3.0) * delta_c + sigma_intr
        return sigma_mism.flatten()

    def _total_surf_stress(self, y, i_pack):
        sth_surf = self._dis_stress_th_surf(y, i_pack)
        sei_mism = self._sei_mismatch_stress(y, i_pack)
        return (sth_surf + sei_mism).flatten()

    def _force_from_stress(self, y, i_pack):
        stress = self.battery.physics.state(y, "stress")
        area = float(self.battery.config.get("stress_options", {}).get("force_area", self.battery.raw_params[0]["A"]))
        return torch.mean(stress, dim=1) * area

    def evaluate(self, y, i_pack):
        y = y.to(self.battery.device)
        if torch.is_tensor(i_pack):
            i_pack = i_pack.to(self.battery.device)
        values = {}
        with torch.no_grad():
            for name, fn in self.outputs.items():
                out = fn(y, i_pack)
                if not torch.is_tensor(out):
                    out = torch.as_tensor(out, device=self.battery.device, dtype=y.dtype)
                values[name] = out.detach().cpu().reshape(-1)
        return values

    def state_records(self, times, y_list):
        rows = []
        layout = self.battery.physics.state_layout
        y_hist = torch.stack(y_list)
        for step_idx, time_value in enumerate(times):
            y_step = y_hist[step_idx]
            for cell_idx in range(self.battery.n_cells):
                y_cell = y_step[cell_idx]
                for field in layout.fields:
                    values = layout.get(y_cell, field.name).reshape(-1)
                    for local_idx, value in enumerate(values):
                        rows.append({
                            "time": float(time_value),
                            "cell": cell_idx,
                            "kind": "state",
                            "name": field.name,
                            "local_index": local_idx,
                            "value": float(value),
                        })
        return rows

    def output_records(self, times, y_list, i_pack_history):
        rows = []
        y_hist = torch.stack(y_list)
        for step_idx, time_value in enumerate(times):
            y_step = y_hist[step_idx].to(self.battery.device)
            i_pack = self._history_value(i_pack_history, step_idx)
            values = self.evaluate(y_step, i_pack)
            for name, output in values.items():
                if output.numel() == 1:
                    rows.append({
                        "time": float(time_value),
                        "cell": -1,
                        "kind": "output",
                        "name": name,
                        "local_index": 0,
                        "value": float(output[0]),
                    })
                else:
                    for cell_idx, value in enumerate(output):
                        rows.append({
                            "time": float(time_value),
                            "cell": cell_idx,
                            "kind": "output",
                            "name": name,
                            "local_index": 0,
                            "value": float(value),
                        })
        return rows

    def save(self, times, y_list, i_pack_history, filename="all_results.csv"):
        rows = self.state_records(times, y_list)
        rows.extend(self.output_records(times, y_list, i_pack_history))
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return df

    @staticmethod
    def _history_value(history, step_idx):
        if isinstance(history, (list, tuple)):
            return history[step_idx]
        if hasattr(history, "__len__") and not isinstance(history, (str, bytes)):
            try:
                return history[step_idx]
            except TypeError:
                pass
        return history
