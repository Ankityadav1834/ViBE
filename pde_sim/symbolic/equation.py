"""
Equation containers — PDEEquation, AlgebraicEquation, and System.

A System is the top-level container a researcher defines in their
equations file. It holds all coupled PDEs, algebraic relations, and
metadata needed by the assembly pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List

from pde_sim.symbolic.expressions import (
    Dt,
    Expression,
    EquationPair,
    Field,
    ensure_expression,
)


@dataclass
class PDEEquation:
    """
    A time-dependent PDE: Dt(field) == rhs_expression.

    Attributes
    ----------
    name : str
        Human-readable name (also the key in the state vector).
    field : Field
        The unknown being evolved.
    rhs : Expression
        Right-hand side of the PDE after isolating Dt(field).
    domain : str
        Mesh domain this equation lives on.
    boundary_conditions : dict
        Maps location names → BC objects.
    initial_condition : any
        Scalar, tensor, or callable for the initial state.
    scale : float
        Scaling factor for Newton solver conditioning.
    nonnegative : bool
        Whether to clamp the field ≥ 0 during solving.
    notes : str
        Human-readable description for documentation.
    """
    name: str
    field: Field
    rhs: Expression
    domain: str = "default"
    boundary_conditions: dict = field(default_factory=dict)
    initial_condition: Any = 0.0
    scale: float = 1.0
    nonnegative: bool = False
    notes: str = ""


@dataclass
class AlgebraicEquation:
    """
    A stationary (algebraic) equation: 0 == residual_expression.

    No Dt() — the field is determined at each time step by satisfying
    the algebraic constraint.

    Attributes
    ----------
    name : str
        Identifier.
    field : Field
        The unknown.
    residual : Expression
        The expression whose root defines the field value.
    domain : str
        Mesh domain.
    boundary_conditions : dict
        Maps location names → BC objects.
    initial_condition : any
        Initial guess.
    scale : float
        Newton scaling.
    notes : str
        Description.
    """
    name: str
    field: Field
    residual: Expression
    domain: str = "default"
    boundary_conditions: dict = field(default_factory=dict)
    initial_condition: Any = 0.0
    scale: float = 1.0
    notes: str = ""


class System:
    """
    A coupled system of PDEs and algebraic equations.

    Usage
    -----
    ```python
    c = Field("c", domain="electrolyte")
    T = Field("T", domain="cell")

    system = System({
        "c": Dt(c) == Div(D * Grad(c)),
        "T": Dt(T) == Q / (rho * Cp),
    })
    ```

    Each value can be:
    - An ``EquationPair`` (from ``lhs == rhs``)
    - A ``PDEEquation`` or ``AlgebraicEquation`` (pre-built)
    """

    def __init__(
        self,
        equations: Dict[str, Any],
        metadata: Optional[Dict[str, dict]] = None,
    ):
        self.pdes: Dict[str, PDEEquation] = {}
        self.algebraic: Dict[str, AlgebraicEquation] = {}
        self._ordering: List[str] = []
        metadata = metadata or {}

        for name, eq_def in equations.items():
            self._ordering.append(name)
            meta = metadata.get(name, {})

            if isinstance(eq_def, PDEEquation):
                self.pdes[name] = eq_def
            elif isinstance(eq_def, AlgebraicEquation):
                self.algebraic[name] = eq_def
            elif isinstance(eq_def, EquationPair):
                self._parse_equation_pair(name, eq_def, meta)
            else:
                raise TypeError(
                    f"Equation '{name}': expected EquationPair, PDEEquation, "
                    f"or AlgebraicEquation, got {type(eq_def).__name__}."
                )

    def _parse_equation_pair(self, name: str, pair: EquationPair, meta: dict):
        """
        Decompose ``lhs == rhs`` into PDEEquation or AlgebraicEquation.

        If the LHS is ``Dt(field)``, it's a PDE; otherwise algebraic.
        """
        lhs, rhs = pair.lhs, pair.rhs

        if isinstance(lhs, Dt):
            # PDE:  Dt(field) == rhs
            field_expr = lhs.child
            if not isinstance(field_expr, Field):
                raise TypeError(
                    f"Equation '{name}': Dt() must wrap a Field, "
                    f"got {type(field_expr).__name__}."
                )
            self.pdes[name] = PDEEquation(
                name=name,
                field=field_expr,
                rhs=rhs,
                domain=meta.get("domain", field_expr.domain),
                boundary_conditions=meta.get("boundary_conditions", {}),
                initial_condition=meta.get("initial_condition", 0.0),
                scale=meta.get("scale", 1.0),
                nonnegative=meta.get("nonnegative", False),
                notes=meta.get("notes", ""),
            )
        else:
            # Algebraic:  lhs == rhs  →  residual = lhs - rhs
            residual = lhs - rhs
            # Try to find the field
            field_expr = self._extract_field(lhs, rhs)
            self.algebraic[name] = AlgebraicEquation(
                name=name,
                field=field_expr,
                residual=residual,
                domain=meta.get("domain", getattr(field_expr, "domain", "default")),
                boundary_conditions=meta.get("boundary_conditions", {}),
                initial_condition=meta.get("initial_condition", 0.0),
                scale=meta.get("scale", 1.0),
                notes=meta.get("notes", ""),
            )

    @staticmethod
    def _extract_field(lhs: Expression, rhs: Expression) -> Field:
        """Walk both sides to find a Field node."""
        for expr in (lhs, rhs):
            found = _find_fields(expr)
            if found:
                return found[0]
        # Fallback: create a placeholder
        return Field("_unknown")

    @property
    def names(self) -> List[str]:
        """All equation names in definition order."""
        return list(self._ordering)

    @property
    def all_fields(self) -> Dict[str, Field]:
        """All Field objects across every equation."""
        fields = {}
        for name, eq in self.pdes.items():
            fields[name] = eq.field
        for name, eq in self.algebraic.items():
            fields[name] = eq.field
        return fields

    def __len__(self):
        return len(self.pdes) + len(self.algebraic)

    def __repr__(self):
        parts = []
        for name in self._ordering:
            if name in self.pdes:
                eq = self.pdes[name]
                parts.append(f"  {name}: Dt({eq.field.name}) == {eq.rhs._repr()}")
            else:
                eq = self.algebraic[name]
                parts.append(f"  {name}: 0 == {eq.residual._repr()}")
        body = "\n".join(parts)
        return f"System({{\n{body}\n}})"


def _find_fields(expr: Expression) -> List[Field]:
    """Recursively collect all Field nodes in an expression tree."""
    if isinstance(expr, Field):
        return [expr]
    result = []
    for attr_name in ("child", "left", "right", "radius",
                       "condition", "if_true", "if_false"):
        child = getattr(expr, attr_name, None)
        if isinstance(child, Expression):
            result.extend(_find_fields(child))
    return result
