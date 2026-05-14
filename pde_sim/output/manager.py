"""
Output manager — saves states, derived quantities, and diagnostics.

Researchers can register custom derived quantities (e.g. flux magnitudes,
energy integrals, custom expressions) that are computed and saved alongside
the raw state variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import pandas as pd

from pde_sim.state.layout import StateLayout


@dataclass
class DerivedQuantity:
    """
    A derived output computed from the state.

    Parameters
    ----------
    name : str
        Column name in the output CSV.
    fn : callable(state_dict, params, time) → Tensor
        Computes the derived value from unpacked state fields.
    requires : tuple of str
        Field names this quantity depends on.
    description : str
        Human-readable description.
    """
    name: str
    fn: Callable
    requires: tuple = ()
    description: str = ""


class OutputManager:
    """
    Manages saving simulation results — states, derived quantities, and diagnostics.

    Usage
    -----
    ```python
    out = OutputManager(layout)
    out.register(DerivedQuantity("energy", compute_energy))
    out.save(times, states, params, "results.csv")
    ```
    """

    def __init__(self, state_layout: StateLayout):
        self.layout = state_layout
        self._derived: Dict[str, DerivedQuantity] = {}

    def register(self, quantity: DerivedQuantity) -> "OutputManager":
        """Register a derived quantity for output. Returns self for chaining."""
        self._derived[quantity.name] = quantity
        return self

    def register_expression(
        self,
        name: str,
        fn: Callable,
        requires: tuple = (),
        description: str = "",
    ) -> "OutputManager":
        """Convenience: register a derived quantity from a function."""
        return self.register(
            DerivedQuantity(name, fn, requires, description)
        )

    def evaluate_derived(
        self,
        state: torch.Tensor,
        params: Optional[Dict[str, Any]] = None,
        time: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate all registered derived quantities.

        Parameters
        ----------
        state : Tensor
            Flat state vector.
        params : dict
            Runtime parameters.
        time : float
            Current simulation time.

        Returns
        -------
        dict mapping name → Tensor
        """
        fields = self.layout.unpack(state)
        results = {}
        for name, dq in self._derived.items():
            # Check dependencies
            missing = [r for r in dq.requires if r not in fields]
            if missing:
                continue
            with torch.no_grad():
                val = dq.fn(fields, params or {}, time)
                if not torch.is_tensor(val):
                    val = torch.tensor(float(val))
                results[name] = val.detach().cpu().reshape(-1)
        return results

    def save(
        self,
        times: List[float],
        states: List[torch.Tensor],
        params: Optional[Dict[str, Any]] = None,
        filename: str = "results.csv",
        save_raw_states: bool = True,
    ) -> pd.DataFrame:
        """
        Save simulation results to CSV.

        Output format: long-form table with columns:
        time, kind, name, index, value
        """
        rows = []

        for step_idx, (t, state) in enumerate(zip(times, states)):
            if torch.is_tensor(state) and state.device.type != "cpu":
                state = state.cpu()

            # Raw state fields
            if save_raw_states:
                for spec in self.layout.fields:
                    vals = self.layout.get(state, spec.name).reshape(-1)
                    for i, v in enumerate(vals):
                        rows.append({
                            "time": float(t),
                            "kind": "state",
                            "name": spec.name,
                            "index": i,
                            "value": float(v),
                        })

            # Derived quantities
            derived = self.evaluate_derived(state, params, float(t))
            for name, tensor in derived.items():
                for i, v in enumerate(tensor):
                    rows.append({
                        "time": float(t),
                        "kind": "derived",
                        "name": name,
                        "index": i,
                        "value": float(v),
                    })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return df

    def __repr__(self):
        derived_names = list(self._derived.keys())
        return (
            f"OutputManager(fields={self.layout.names()}, "
            f"derived={derived_names})"
        )
