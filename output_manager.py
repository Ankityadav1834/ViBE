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
