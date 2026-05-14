"""
State layout — flat state-vector management for coupled PDE systems.

Maps named fields to slices of a monolithic state tensor, handles
initial conditions, scaling, and non-negativity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class FieldSpec:
    """
    Metadata for one field in the state vector.

    Parameters
    ----------
    name : str
        Identifier matching the Field/equation name.
    size : int
        Number of DOFs for this field.
    initial : any
        Initial condition (scalar, tensor, or callable).
    scale : float
        Scaling factor for Newton solver conditioning.
    nonnegative : bool
        Whether to enforce u ≥ 0.
    """
    name: str
    size: int
    initial: Any = 0.0
    scale: float = 1.0
    nonnegative: bool = False


class StateLayout:
    """
    Maps named fields to slices of a flat state tensor.

    Usage
    -----
    ```python
    layout = StateLayout()
    layout.register("c", 30, initial=1000.0, scale=1000.0)
    layout.register("T", 1, initial=298.15, scale=300.0)

    y0 = layout.initial_state(device="cuda")
    c_values = layout.get(y, "c")
    ```
    """

    def __init__(self):
        self._fields: List[FieldSpec] = []
        self._slices: Dict[str, slice] = {}
        self._total_size: int = 0

    def register(
        self,
        name: str,
        size: int,
        initial: Any = 0.0,
        scale: float = 1.0,
        nonnegative: bool = False,
    ) -> slice:
        """
        Register a new field and return its slice.

        Raises ValueError if the name is already registered.
        """
        if name in self._slices:
            raise ValueError(f"Field '{name}' is already registered.")
        if size <= 0:
            raise ValueError(f"Field '{name}' must have positive size, got {size}.")

        start = self._total_size
        stop = start + size
        spec = FieldSpec(name, size, initial, scale, nonnegative)
        self._fields.append(spec)
        self._slices[name] = slice(start, stop)
        self._total_size = stop
        return self._slices[name]

    @property
    def total_size(self) -> int:
        return self._total_size

    @property
    def fields(self) -> List[FieldSpec]:
        return list(self._fields)

    def names(self) -> List[str]:
        return [f.name for f in self._fields]

    def __contains__(self, name: str) -> bool:
        return name in self._slices

    def __len__(self) -> int:
        return len(self._fields)

    def get_slice(self, name: str) -> slice:
        """Get the slice for a named field."""
        return self._slices[name]

    def get(self, state: torch.Tensor, name: str) -> torch.Tensor:
        """Extract a field from the flat state vector."""
        return state[..., self._slices[name]]

    def set(self, state: torch.Tensor, name: str, values: torch.Tensor):
        """Write values into the field's slice of the state vector."""
        state[..., self._slices[name]] = values

    def pack(
        self,
        values: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Pack a dict of field tensors into a flat state vector.

        Missing fields are zero-filled.
        """
        chunks = []
        for spec in self._fields:
            if spec.name in values:
                v = values[spec.name]
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, device=device, dtype=dtype)
                else:
                    v = v.to(device=device, dtype=dtype)
                chunks.append(v.reshape(spec.size))
            else:
                chunks.append(
                    torch.zeros(spec.size, device=device, dtype=dtype)
                )
        return torch.cat(chunks, dim=0)

    def unpack(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unpack a flat state vector into a dict of field tensors."""
        return {
            spec.name: state[..., self._slices[spec.name]]
            for spec in self._fields
        }

    def initial_state(
        self,
        params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build the initial state vector from registered field specs."""
        params = params or {}
        y = torch.zeros(self._total_size, device=device, dtype=dtype)
        for spec in self._fields:
            value = spec.initial
            if callable(value):
                value = value(params, device, dtype)
            elif isinstance(value, str):
                value = params.get(value, 0.0)
            if torch.is_tensor(value):
                value = value.to(device=device, dtype=dtype).reshape(-1)
            else:
                value = torch.tensor(float(value), device=device, dtype=dtype)
            if value.numel() == 1 and spec.size > 1:
                value = value.expand(spec.size)
            y[self._slices[spec.name]] = value
        return y

    def scale_vector(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build a scaling vector for Newton solver conditioning."""
        s = torch.ones(self._total_size, device=device, dtype=dtype)
        for spec in self._fields:
            s[self._slices[spec.name]] = spec.scale
        return s

    def nonnegative_mask(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Boolean mask: True where the field is constrained ≥ 0."""
        mask = torch.zeros(self._total_size, device=device, dtype=torch.bool)
        for spec in self._fields:
            if spec.nonnegative:
                mask[self._slices[spec.name]] = True
        return mask

    def __repr__(self):
        parts = ", ".join(f"{s.name}({s.size})" for s in self._fields)
        return f"StateLayout([{parts}], total={self._total_size})"
