"""
Symbolic expression AST for PDE definitions.

Every node is a frozen dataclass — immutable and hashable. Python operator
overloading lets researchers write natural mathematical expressions:

    c = Field("c")
    D = Param("D")
    Dt(c) == Div(D * Grad(c))

The AST is later walked by the discretization backend to produce tensor
operations on the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union


# ═══════════════════════════════════════════════════════════════════════════
#  Base expression with operator overloading
# ═══════════════════════════════════════════════════════════════════════════

class Expression:
    """
    Abstract base for every node in the symbolic expression tree.

    Arithmetic operators are overloaded so users can write:
        D * Grad(c)  +  source
    and get the correct AST automatically.
    """

    # ── Arithmetic ──────────────────────────────────────────────────────────
    def __add__(self, other):
        return Add(self, ensure_expression(other))

    def __radd__(self, other):
        return Add(ensure_expression(other), self)

    def __sub__(self, other):
        return Subtract(self, ensure_expression(other))

    def __rsub__(self, other):
        return Subtract(ensure_expression(other), self)

    def __mul__(self, other):
        return Multiply(self, ensure_expression(other))

    def __rmul__(self, other):
        return Multiply(ensure_expression(other), self)

    def __truediv__(self, other):
        return Divide(self, ensure_expression(other))

    def __rtruediv__(self, other):
        return Divide(ensure_expression(other), self)

    def __pow__(self, other):
        return Power(self, ensure_expression(other))

    def __rpow__(self, other):
        return Power(ensure_expression(other), self)

    def __neg__(self):
        return Negate(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return Abs(self)

    # ── Comparison (for equation LHS == RHS syntax) ─────────────────────────
    def __eq__(self, other):
        """
        Overloaded for equation definition syntax:
            Dt(c) == Div(D * Grad(c))
        Returns an EquationPair, not a boolean.
        """
        return EquationPair(lhs=self, rhs=ensure_expression(other))

    def __hash__(self):
        return id(self)

    # ── Pretty printing ────────────────────────────────────────────────────
    def __repr__(self):
        return self._repr()

    def _repr(self) -> str:
        return f"{self.__class__.__name__}(...)"


# ═══════════════════════════════════════════════════════════════════════════
#  Leaf nodes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class Constant(Expression):
    """A literal numeric value."""
    value: float

    def _repr(self):
        return f"{self.value}"


@dataclass(frozen=True, eq=False)
class Field(Expression):
    """
    A solved state variable (unknown).

    Parameters
    ----------
    name : str
        Identifier used to index into the state vector.
    domain : str, optional
        Which mesh domain this field lives on (e.g. "electrolyte", "particle").
    size : int or None
        Number of DOFs. If None, inferred from the mesh at build time.
    """
    name: str
    domain: str = "default"
    size: Optional[int] = None

    def _repr(self):
        return self.name


@dataclass(frozen=True, eq=False)
class Param(Expression):
    """
    A named parameter — resolved at runtime from the parameter dict.

    Can be a scalar, a tensor, or a callable (state-dependent parameter).
    """
    name: str
    default: Any = None

    def _repr(self):
        return self.name


# ═══════════════════════════════════════════════════════════════════════════
#  Differential operators
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class Grad(Expression):
    """Gradient operator:  ∇(child)"""
    child: Expression

    def _repr(self):
        return f"Grad({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Div(Expression):
    """Divergence operator:  ∇·(child)"""
    child: Expression

    def _repr(self):
        return f"Div({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Laplacian(Expression):
    """Laplacian operator:  ∇²(child) = Div(Grad(child))"""
    child: Expression

    def _repr(self):
        return f"Laplacian({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Dt(Expression):
    """
    Time derivative operator:  ∂(child)/∂t

    Used on the LHS of a PDE:
        Dt(c) == Div(D * Grad(c))
    """
    child: Expression

    def _repr(self):
        return f"Dt({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Curl(Expression):
    """Curl operator:  ∇×(child)  (for 2D/3D vector fields)"""
    child: Expression

    def _repr(self):
        return f"Curl({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class SphericalDiv(Expression):
    """
    Spherical divergence:  (1/r²) ∂/∂r (r² · child)

    Used for radially symmetric PDEs in spherical coordinates.
    """
    child: Expression
    radius: Expression

    def _repr(self):
        return f"SphericalDiv({self.child._repr()}, r={self.radius._repr()})"


# ═══════════════════════════════════════════════════════════════════════════
#  Binary arithmetic nodes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class Add(Expression):
    left: Expression
    right: Expression

    def _repr(self):
        return f"({self.left._repr()} + {self.right._repr()})"


@dataclass(frozen=True, eq=False)
class Subtract(Expression):
    left: Expression
    right: Expression

    def _repr(self):
        return f"({self.left._repr()} - {self.right._repr()})"


@dataclass(frozen=True, eq=False)
class Multiply(Expression):
    left: Expression
    right: Expression

    def _repr(self):
        return f"({self.left._repr()} * {self.right._repr()})"


@dataclass(frozen=True, eq=False)
class Divide(Expression):
    left: Expression
    right: Expression

    def _repr(self):
        return f"({self.left._repr()} / {self.right._repr()})"


@dataclass(frozen=True, eq=False)
class Power(Expression):
    left: Expression
    right: Expression

    def _repr(self):
        return f"({self.left._repr()} ** {self.right._repr()})"


@dataclass(frozen=True, eq=False)
class Negate(Expression):
    child: Expression

    def _repr(self):
        return f"(-{self.child._repr()})"


# ═══════════════════════════════════════════════════════════════════════════
#  Mathematical functions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class Abs(Expression):
    child: Expression

    def _repr(self):
        return f"Abs({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Sqrt(Expression):
    child: Expression

    def _repr(self):
        return f"Sqrt({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Exp(Expression):
    child: Expression

    def _repr(self):
        return f"Exp({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Log(Expression):
    child: Expression

    def _repr(self):
        return f"Log({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Tanh(Expression):
    child: Expression

    def _repr(self):
        return f"Tanh({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Sinh(Expression):
    child: Expression

    def _repr(self):
        return f"Sinh({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Cosh(Expression):
    child: Expression

    def _repr(self):
        return f"Cosh({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Sign(Expression):
    child: Expression

    def _repr(self):
        return f"Sign({self.child._repr()})"


@dataclass(frozen=True, eq=False)
class Clamp(Expression):
    child: Expression
    low: Optional[Expression] = None
    high: Optional[Expression] = None

    def _repr(self):
        lo = self.low._repr() if self.low else "None"
        hi = self.high._repr() if self.high else "None"
        return f"Clamp({self.child._repr()}, {lo}, {hi})"


@dataclass(frozen=True, eq=False)
class Conditional(Expression):
    """
    Smooth conditional:  if condition > 0 then if_true else if_false
    Uses a smooth tanh approximation for differentiability.
    """
    condition: Expression
    if_true: Expression
    if_false: Expression
    sharpness: float = 100.0

    def _repr(self):
        return (f"Conditional({self.condition._repr()}, "
                f"{self.if_true._repr()}, {self.if_false._repr()})")


# ═══════════════════════════════════════════════════════════════════════════
#  Equation pair (from == overloading)
# ═══════════════════════════════════════════════════════════════════════════

class EquationPair:
    """
    Container returned by ``expr_lhs == expr_rhs``.

    Not an Expression itself — it's a specification that the System
    class consumes to build PDEEquation / AlgebraicEquation objects.
    """

    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs._repr()} == {self.rhs._repr()}"


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def ensure_expression(value) -> Expression:
    """Wrap Python scalars as Constant nodes."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(
        f"Cannot convert {type(value).__name__} to Expression. "
        f"Use Constant(...), Field(...), or Param(...) instead."
    )
