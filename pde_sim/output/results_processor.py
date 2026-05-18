"""
ResultsProcessor — shared post-processing engine for all VIBE solvers.

All three solvers (BasicSolver, AdvancedSolver, ControlledSolver) delegate
their `process_results` calls here so the output logic lives in one place
and respects the user-supplied OutputSpec.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from pde_sim.output.output_spec import OutputSpec, _CATALOGUE


class ResultsProcessor:
    """
    Extracts, filters, and saves simulation results according to an OutputSpec.

    Parameters
    ----------
    battery : ImplicitBatterySolver
        The pack solver (for access to physics, n_cells, raw_params, etc.)
    core_solver : BasicSolver | AdvancedSolver
        Used for `_get_voltage_breakdown_numpy`.
    output_spec : OutputSpec
        Declares which variables to compute and save.
    out_dir : str
        Directory where output files are written.
    """

    def __init__(self, battery, core_solver, output_spec: OutputSpec, out_dir: str):
        self.battery = battery
        self.core_solver = core_solver
        self.spec = output_spec
        self.out_dir = out_dir

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        times: list,
        y_list: list,
        I_pack_hist,   # float | array-like
        I_cells_hist=None,
    ) -> dict:
        """
        Post-process a completed simulation.

        Returns a dict of arrays keyed by the storage keys in OutputSpec
        (e.g. ``res['TermV']``, ``res['Curr']``, …).
        """
        battery = self.battery
        spec = self.spec
        n_cells = battery.n_cells

        times_np = np.array(times)
        y = torch.stack(y_list).numpy()
        n_steps = len(times_np)

        # Reconstruct per-cell currents when not pre-computed
        if I_cells_hist is None:
            I_pack_arr = np.asarray(I_pack_hist if np.ndim(I_pack_hist) > 0
                                    else [float(I_pack_hist)] * n_steps)
        else:
            I_cells_np = torch.stack(I_cells_hist).numpy()

        # ── Allocate result arrays ─────────────────────────────────────────
        res: dict[str, np.ndarray] = {}

        # Per-cell arrays
        cell_vars = [v for v in spec.variables
                     if _CATALOGUE[v].shape == "cell"]
        for v in cell_vars:
            res[_CATALOGUE[v].key] = np.zeros((n_steps, n_cells))

        # Pack-level arrays
        if spec.wants("pack_voltage"):
            res["PackVoltage"] = np.zeros(n_steps)
        if spec.wants("pack_current"):
            pack_arr = (I_pack_hist if hasattr(I_pack_hist, '__len__')
                        else np.full(n_steps, float(I_pack_hist)))
            res["PackCurrent"] = np.asarray(pack_arr, dtype=float)
        if spec.wants("contact_resistance_drop"):
            res["OhmRC"] = np.zeros((n_steps, n_cells))
        if spec.wants("busbar_resistance_drop"):
            res["OhmRB"] = np.zeros((n_steps, n_cells))

        # ── Step-by-step extraction ────────────────────────────────────────
        needs_bd = spec.needs_breakdown
        has_sei = "Lsei" in battery.physics.state_layout.slices
        cs_n_slc = battery.physics.state_layout.slice("cs_n")
        Lsei_slc = (battery.physics.state_layout.slice("Lsei")
                    if has_sei else None)

        print(f"Post-processing {n_steps} steps for "
              f"{len(spec.variables)} output(s): {spec.variables}")

        with torch.no_grad():
            for i in range(n_steps):
                # Resolve per-cell currents for this step
                if I_cells_hist is not None:
                    I_c = I_cells_np[i].flatten()
                else:
                    y_t = torch.from_numpy(y[i]).to(battery.device)
                    I_c = self.core_solver.solve_current_distribution(
                        y_t, float(I_pack_arr[i])
                    ).cpu().numpy().flatten()

                # Voltage breakdown (computed once per step if any BD var wanted)
                bd_list: list[dict | None] = [None] * n_cells
                if needs_bd:
                    for k in range(n_cells):
                        bd_list[k] = self.core_solver._get_voltage_breakdown_numpy(
                            y[i, k], I_c[k], battery.raw_params[k]
                        )

                # Fill per-cell values
                pack_volts: list[float] = []
                for k in range(n_cells):
                    yc = y[i, k]
                    pk = battery.raw_params[k]
                    bd = bd_list[k]

                    if spec.wants("terminal_voltage"):
                        v = bd["TermV"] if bd else 0.0
                        res["TermV"][i, k] = v
                        pack_volts.append(v)
                    if spec.wants("cell_current"):
                        res["Curr"][i, k] = I_c[k]
                    if spec.wants("soc"):
                        cs_surf = yc[cs_n_slc.stop - 1]
                        res["SOC"][i, k] = (cs_surf / pk["cs_max_n"] - 0.01) / 0.94
                    if spec.wants("temperature"):
                        res["Temp"][i, k] = yc[-1]
                    if spec.wants("sei_thickness"):
                        lsei = yc[Lsei_slc.start] if has_sei else pk["Lsei_0"]
                        res["SEI_Thick"][i, k] = lsei * 1e9
                    if spec.wants("ocv"):
                        res["OCV"][i, k] = bd["OCV"]
                    if spec.wants("rxn_overpotential"):
                        res["Rxn"][i, k] = bd["Rxn"]
                    if spec.wants("ohmic_solid"):
                        res["OhmS"][i, k] = bd["OhmSolid"]
                    if spec.wants("ohmic_electrolyte"):
                        res["OhmE"][i, k] = bd["OhmElec"]
                    if spec.wants("concentration_overpotential"):
                        res["Conc"][i, k] = bd["Conc"]
                    if spec.wants("sei_voltage"):
                        res["SEI"][i, k] = bd["SEI"]
                    if spec.wants("capacity_fade"):
                        lsei = yc[Lsei_slc.start] if has_sei else pk["Lsei_0"]
                        delta = lsei - pk["Lsei_0"]
                        fade = (2 * pk["rho_sei"] / pk["Msei"] * delta
                                * 3 / pk["Rs_n"] / pk["cs_max_n"])
                        res["CapFade"][i, k] = fade * 100.0
                    # External circuit losses (liionpack-comparable)
                    if spec.wants("contact_resistance_drop"):
                        res["OhmRC"][i, k] = I_c[k] * pk.get("R_contact", 0.0)
                    if spec.wants("busbar_resistance_drop"):
                        res["OhmRB"][i, k] = I_c[k] * pk.get("R_bus", 0.0)

                if spec.wants("pack_voltage") and pack_volts:
                    arr = np.array(pack_volts).reshape(
                        battery.n_series, battery.n_parallel
                    )
                    res["PackVoltage"][i] = np.sum(np.mean(arr, axis=1))

        # ── Save to .npz ──────────────────────────────────────────────────
        npz_path = os.path.join(self.out_dir, "simulation_results.npz")
        # Include the list of saved variable keys as metadata so loaders know
        # what was requested without re-inspecting every array name.
        saved_keys = list(res.keys())
        np.savez_compressed(
            npz_path,
            times=times_np,
            _saved_keys=np.array(saved_keys),
            **res,
        )
        size_kb = os.path.getsize(npz_path) / 1024
        print(
            f"Saved {npz_path}  ({size_kb:.1f} KB)\n"
            f"  Variables: {', '.join(saved_keys)}"
        )
        return res
