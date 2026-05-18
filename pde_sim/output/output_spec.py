"""
OutputSpec — Selective output variable declaration for VIBE simulations.

Inspired by liionpack's ``output_variables`` list, but extended to support
units, descriptions, and post-processing hooks so each variable is
self-documenting.

Usage
-----
# Default (current, voltage, SOC, SEI thickness, temperature, overpotentials)
solver.simulate(t_end=3600, dt_init=1.0, controller=ctrl)

# Only save terminal voltage and temperature
from pde_sim.output import OutputSpec
spec = OutputSpec(["terminal_voltage", "temperature"])
solver.simulate(t_end=3600, dt_init=1.0, controller=ctrl, output_spec=spec)

# Full output including every overpotential component
spec = OutputSpec("all")
solver.simulate(t_end=3600, dt_init=1.0, controller=ctrl, output_spec=spec)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

# ── Variable catalogue ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _VarDef:
    """Internal descriptor for one output variable."""
    key: str           # key in the results dict (also the array name in .npz)
    label: str         # human-readable label for plots / CSV headers
    unit: str          # physical unit string
    shape: str         # 'cell' | 'pack' | 'cell_x_nr'
    description: str
    # Optional: a callable (y_cell, I_cell, physics, p) -> float used during
    # post-processing.  None means the value is extracted directly from the
    # state vector or a voltage breakdown dict.
    extractor: Callable | None = field(default=None, compare=False, hash=False)


# Every variable the framework knows how to produce
_CATALOGUE: dict[str, _VarDef] = {
    # ── Primary state outputs ────────────────────────────────────────────────
    "terminal_voltage": _VarDef(
        key="TermV",
        label="Terminal Voltage",
        unit="V",
        shape="cell",
        description="Cell terminal voltage including all ohmic and kinetic losses.",
    ),
    "cell_current": _VarDef(
        key="Curr",
        label="Cell Current",
        unit="A",
        shape="cell",
        description="Per-cell current (negative = charge, positive = discharge).",
    ),
    "soc": _VarDef(
        key="SOC",
        label="State of Charge",
        unit="-",
        shape="cell",
        description="Estimated SOC from surface stoichiometry of negative electrode.",
    ),
    "temperature": _VarDef(
        key="Temp",
        label="Temperature",
        unit="K",
        shape="cell",
        description="Cell temperature (last state in the state vector).",
    ),
    "sei_thickness": _VarDef(
        key="SEI_Thick",
        label="SEI Thickness",
        unit="nm",
        shape="cell",
        description="Total SEI layer thickness in nanometres.",
    ),
    # ── Voltage breakdown / overpotentials ───────────────────────────────────
    "ocv": _VarDef(
        key="OCV",
        label="Open Circuit Voltage",
        unit="V",
        shape="cell",
        description="Open circuit voltage = U_p(theta_p) - U_n(theta_n).",
    ),
    "rxn_overpotential": _VarDef(
        key="Rxn",
        label="Reaction Overpotential",
        unit="V",
        shape="cell",
        description="Sum of Butler-Volmer activation overpotentials at anode and cathode.",
    ),
    "ohmic_solid": _VarDef(
        key="OhmS",
        label="Solid Ohmic Loss",
        unit="V",
        shape="cell",
        description="Voltage drop due to finite electronic conductivity in solid electrodes.",
    ),
    "ohmic_electrolyte": _VarDef(
        key="OhmE",
        label="Electrolyte Ohmic Loss",
        unit="V",
        shape="cell",
        description="Ionic resistance loss through electrolyte (Bruggeman-corrected).",
    ),
    "concentration_overpotential": _VarDef(
        key="Conc",
        label="Concentration Overpotential",
        unit="V",
        shape="cell",
        description="Nernst-type loss from electrolyte concentration gradient.",
    ),
    "sei_voltage": _VarDef(
        key="SEI",
        label="SEI Voltage Drop",
        unit="V",
        shape="cell",
        description="Ohmic drop across the SEI film: I * L_SEI / (kappa_SEI * a_s * L_n * A).",
    ),
    # ── Pack-level outputs ───────────────────────────────────────────────────
    "pack_voltage": _VarDef(
        key="PackVoltage",
        label="Pack Voltage",
        unit="V",
        shape="pack",
        description="Sum of series-group mean voltages.",
    ),
    "pack_current": _VarDef(
        key="PackCurrent",
        label="Pack Current",
        unit="A",
        shape="pack",
        description="Applied pack current (positive = discharge).",
    ),
    # ── Capacity fade (derived) ───────────────────────────────────────────────
    "capacity_fade": _VarDef(
        key="CapFade",
        label="Capacity Fade",
        unit="%",
        shape="cell",
        description="Estimated capacity fade from SEI growth (active lithium loss).",
    ),
    # ── External circuit resistance losses (liionpack-comparable) ─────────────
    "contact_resistance_drop": _VarDef(
        key="OhmRC",
        label="Contact Resistance Drop",
        unit="V",
        shape="cell",
        description="Voltage drop across R_contact (tab-to-busbar weld). "
                    "Equivalent to liionpack Rc element (default 10 mΩ).",
    ),
    "busbar_resistance_drop": _VarDef(
        key="OhmRB",
        label="Busbar Resistance Drop",
        unit="V",
        shape="cell",
        description="Voltage drop across R_bus (busbar segment + terminal lug). "
                    "Equivalent to liionpack Rb+Rt elements (default 0.1 mΩ + 0.01 mΩ).",
    ),
}

# Convenience aliases
_ALIASES: dict[str, str] = {
    "voltage":           "terminal_voltage",
    "v":                 "terminal_voltage",
    "current":           "cell_current",
    "i":                 "cell_current",
    "T":                 "temperature",
    "temp":              "temperature",
    "sei":               "sei_thickness",
    "lsei":              "sei_thickness",
    "sei_thick":         "sei_thickness",
    "ocv":               "ocv",
    "rxn":               "rxn_overpotential",
    "eta_rxn":           "rxn_overpotential",
    "ohm_s":             "ohmic_solid",
    "ohm_e":             "ohmic_electrolyte",
    "conc":              "concentration_overpotential",
    "sei_v":             "sei_voltage",
    "pack_v":            "pack_voltage",
    "pack_i":            "pack_current",
    "fade":              "capacity_fade",
    # External circuit aliases (liionpack-equivalent)
    "ohm_rc":            "contact_resistance_drop",
    "contact":           "contact_resistance_drop",
    "r_contact":         "contact_resistance_drop",
    "ohm_rb":            "busbar_resistance_drop",
    "busbar":            "busbar_resistance_drop",
    "r_bus":             "busbar_resistance_drop",
}

# Logical groups
_GROUPS: dict[str, list[str]] = {
    "overpotentials": [
        "rxn_overpotential",
        "ohmic_solid",
        "ohmic_electrolyte",
        "concentration_overpotential",
        "sei_voltage",
        "contact_resistance_drop",
        "busbar_resistance_drop",
    ],
    "breakdown": [
        "ocv",
        "rxn_overpotential",
        "ohmic_solid",
        "ohmic_electrolyte",
        "concentration_overpotential",
        "sei_voltage",
        "contact_resistance_drop",
        "busbar_resistance_drop",
        "terminal_voltage",
    ],
    "primary": [
        "terminal_voltage",
        "cell_current",
        "soc",
        "temperature",
        "sei_thickness",
        "pack_voltage",
        "pack_current",
    ],
    "all": list(_CATALOGUE.keys()),
}
# 'default' is an alias for the primary group + all overpotentials (incl. external circuit)
_GROUPS["default"] = _GROUPS["primary"] + _GROUPS["overpotentials"]

# Public name sets for external use
ALL_OUTPUTS: frozenset[str] = frozenset(_CATALOGUE.keys())
DEFAULT_OUTPUTS: tuple[str, ...] = (
    "terminal_voltage",
    "cell_current",
    "soc",
    "temperature",
    "sei_thickness",
    "pack_voltage",
    "pack_current",
    "overpotentials",   # expands to the 5-component breakdown
)


# ── OutputSpec ────────────────────────────────────────────────────────────────

class OutputSpec:
    """
    Declares which variables to compute and save during a simulation.

    Parameters
    ----------
    variables : str | list[str], optional
        Variable names to record.  Can be:
        - ``"all"``               → every available output
        - ``"default"``           → the DEFAULT_OUTPUTS set (recommended)
        - A list of canonical names, aliases, or group names (e.g.
          ``["voltage", "temperature", "overpotentials"]``).
        Default is ``"default"``.
    include_breakdown : bool, optional
        If True, always include all voltage breakdown components.

    Examples
    --------
    >>> # Minimal — just voltage and temperature
    >>> spec = OutputSpec(["terminal_voltage", "temperature"])

    >>> # Everything
    >>> spec = OutputSpec("all")

    >>> # Default set (matches VIBE's previous hard-coded behaviour)
    >>> spec = OutputSpec()
    """

    def __init__(
        self,
        variables: str | Sequence[str] = "default",
        include_breakdown: bool = False,
    ):
        self._requested = variables
        self._include_breakdown = include_breakdown
        self._resolved: list[str] = []
        self._resolve()

    # ── Resolution ───────────────────────────────────────────────────────────

    def _resolve(self):
        req = self._requested

        if isinstance(req, str):
            req = [req]

        resolved: list[str] = []
        for name in req:
            name_lower = name.lower().strip()

            # Group expansion
            if name_lower in _GROUPS:
                resolved.extend(_GROUPS[name_lower])
                continue

            # Alias resolution
            canonical = _ALIASES.get(name_lower, name_lower)
            if canonical not in _CATALOGUE:
                raise ValueError(
                    f"Unknown output variable {name!r}. "
                    f"Valid names: {sorted(_CATALOGUE.keys())} "
                    f"or groups: {sorted(_GROUPS.keys())}."
                )
            resolved.append(canonical)

        if self._include_breakdown:
            for v in _GROUPS["breakdown"]:
                if v not in resolved:
                    resolved.append(v)

        # Deduplicate while preserving order
        seen: set[str] = set()
        self._resolved = [v for v in resolved if not (v in seen or seen.add(v))]

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def variables(self) -> list[str]:
        """Ordered list of resolved canonical variable names."""
        return list(self._resolved)

    def wants(self, name: str) -> bool:
        """Check if a canonical variable name was requested."""
        canonical = _ALIASES.get(name.lower(), name.lower())
        return canonical in self._resolved

    @property
    def needs_breakdown(self) -> bool:
        """True if any voltage-breakdown component was requested."""
        breakdown_keys = {
            "ocv", "rxn_overpotential", "ohmic_solid",
            "ohmic_electrolyte", "concentration_overpotential", "sei_voltage",
            "terminal_voltage",
        }
        return bool(set(self._resolved) & breakdown_keys)

    def result_keys(self) -> list[str]:
        """Return the storage keys (e.g. 'TermV', 'Curr') for requested vars."""
        return [_CATALOGUE[v].key for v in self._resolved]

    def label(self, var: str) -> str:
        canonical = _ALIASES.get(var.lower(), var.lower())
        return _CATALOGUE[canonical].label

    def unit(self, var: str) -> str:
        canonical = _ALIASES.get(var.lower(), var.lower())
        return _CATALOGUE[canonical].unit

    def shape(self, var: str) -> str:
        canonical = _ALIASES.get(var.lower(), var.lower())
        return _CATALOGUE[canonical].shape

    def __repr__(self) -> str:
        return f"OutputSpec({self._resolved})"

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "OutputSpec":
        """Return the default output specification."""
        return cls("default")

    @classmethod
    def minimal(cls) -> "OutputSpec":
        """Return a minimal spec (voltage + current + SOC + temperature)."""
        return cls(["terminal_voltage", "cell_current", "soc", "temperature"])

    @classmethod
    def full(cls) -> "OutputSpec":
        """Return a full spec with every available output."""
        return cls("all")
