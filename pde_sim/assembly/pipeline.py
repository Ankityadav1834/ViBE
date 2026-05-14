"""
Assembly pipeline — transforms symbolic expressions into discrete tensors.

The pipeline walks the expression AST and dispatches each node to the
appropriate discretization backend operation. This is the heart of the
"symbolic → discrete" translation.

Usage
-----
```python
ctx = DiscretizationContext(mesh, backend, fields, params)
rhs_tensor = ctx.evaluate(equation.rhs)
```
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from pde_sim.symbolic.expressions import (
    Expression,
    Constant,
    Field,
    Param,
    Grad,
    Div,
    Laplacian,
    Dt,
    Curl,
    SphericalDiv,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Tanh,
    Sinh,
    Cosh,
    Sign,
    Clamp,
    Conditional,
)
from pde_sim.discretization.base import DiscretizationBackend
from pde_sim.discretization.fvm import FiniteVolumeBackend
from pde_sim.mesh.mesh import CompositeMesh1D
from pde_sim.boundary.conditions import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundarySet,
)


class DiscretizationContext:
    """
    Runtime context for evaluating symbolic expressions into tensors.

    Holds the mesh, backend, current field values, parameters,
    boundary conditions, and solver state (y_old, dt).

    Parameters
    ----------
    mesh : CompositeMesh1D
        The mesh to discretize on.
    backend : DiscretizationBackend
        The spatial discretization backend.
    fields : dict
        Maps field name → current tensor values.
    params : dict
        Maps parameter name → value (scalar, tensor, or callable).
    boundary_conditions : dict
        Maps field name → BoundarySet.
    y_old : dict or None
        Previous time-step field values (for implicit BCs).
    dt : float or None
        Current time step.
    """

    def __init__(
        self,
        mesh: CompositeMesh1D,
        backend: DiscretizationBackend,
        fields: Optional[Dict[str, torch.Tensor]] = None,
        params: Optional[Dict[str, Any]] = None,
        boundary_conditions: Optional[Dict[str, BoundarySet]] = None,
        y_old: Optional[Dict[str, torch.Tensor]] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ):
        self.mesh = mesh
        self.backend = backend
        self.fields = fields or {}
        self.params = params or {}
        self.boundary_conditions = boundary_conditions or {}
        self.y_old = y_old
        self.dt = dt
        self.time = time

    def evaluate(self, expr: Expression) -> torch.Tensor:
        """Recursively evaluate a symbolic expression to a tensor."""
        return _evaluate(expr, self)

    def resolve_param(self, name: str, default: Any = None) -> Any:
        """
        Look up a parameter by name.

        If the value is callable, invoke it with this context.
        """
        value = self.params.get(name, default)
        if callable(value):
            return value(self)
        return value

    def get_field(self, name: str) -> torch.Tensor:
        """Get current values of a named field."""
        if name not in self.fields:
            raise KeyError(
                f"Field '{name}' not found in context. "
                f"Available: {list(self.fields.keys())}"
            )
        value = self.fields[name]
        if callable(value):
            return value()
        return value


# ═══════════════════════════════════════════════════════════════════════════
#  Expression evaluator
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate(expr: Expression, ctx: DiscretizationContext) -> torch.Tensor:
    """
    Recursive AST walker that dispatches each node type.
    """
    device = ctx.mesh.nodes.device
    dtype = ctx.mesh.nodes.dtype

    # ── Leaf nodes ──────────────────────────────────────────────────────
    if isinstance(expr, Constant):
        return torch.tensor(expr.value, device=device, dtype=dtype)

    if isinstance(expr, Field):
        return ctx.get_field(expr.name)

    if isinstance(expr, Param):
        value = ctx.resolve_param(expr.name, expr.default)
        if value is None:
            raise KeyError(
                f"Parameter '{expr.name}' not found and no default set."
            )
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(float(value), device=device, dtype=dtype)

    # ── Differential operators ──────────────────────────────────────────
    if isinstance(expr, Grad):
        child_val = _evaluate(expr.child, ctx)
        return ctx.backend.gradient(child_val)

    if isinstance(expr, Div):
        child_val = _evaluate(expr.child, ctx)
        # For FVM: if the child is a Multiply(coefficient, Grad(field)),
        # handle face interpolation automatically
        if (isinstance(expr.child, Multiply)
                and isinstance(ctx.backend, FiniteVolumeBackend)):
            return _fvm_div_product(expr.child, ctx)
        return ctx.backend.divergence(child_val)

    if isinstance(expr, Laplacian):
        child_val = _evaluate(expr.child, ctx)
        return ctx.backend.laplacian(child_val)

    if isinstance(expr, Dt):
        # Dt() is handled by the solver, not the assembly pipeline.
        # When evaluating the RHS of Dt(u) == f, we never see the Dt node
        # because System already separated it. If we get here, it's a
        # nested time derivative (unusual) — just return the child.
        return _evaluate(expr.child, ctx)

    if isinstance(expr, SphericalDiv):
        r = _evaluate(expr.radius, ctx)
        child_val = _evaluate(expr.child, ctx)
        r_safe = r.clone()
        r_safe[r_safe == 0] = 1.0
        r_sq = r_safe ** 2
        div_r2f = ctx.backend.divergence(r_sq * child_val)
        result = div_r2f / r_sq
        # L'Hôpital at r=0
        if r[0] == 0:
            result[0] = 3.0 * ctx.backend.gradient(child_val)[0]
        return result

    if isinstance(expr, Curl):
        raise NotImplementedError("Curl is not implemented for 1D.")

    # ── Arithmetic ──────────────────────────────────────────────────────
    if isinstance(expr, Add):
        return _evaluate(expr.left, ctx) + _evaluate(expr.right, ctx)

    if isinstance(expr, Subtract):
        return _evaluate(expr.left, ctx) - _evaluate(expr.right, ctx)

    if isinstance(expr, Multiply):
        return _evaluate(expr.left, ctx) * _evaluate(expr.right, ctx)

    if isinstance(expr, Divide):
        return _evaluate(expr.left, ctx) / _evaluate(expr.right, ctx)

    if isinstance(expr, Power):
        return _evaluate(expr.left, ctx) ** _evaluate(expr.right, ctx)

    if isinstance(expr, Negate):
        return -_evaluate(expr.child, ctx)

    # ── Math functions ──────────────────────────────────────────────────
    if isinstance(expr, Abs):
        return torch.abs(_evaluate(expr.child, ctx))

    if isinstance(expr, Sqrt):
        return torch.sqrt(torch.clamp(_evaluate(expr.child, ctx), min=1e-30))

    if isinstance(expr, Exp):
        return torch.exp(_evaluate(expr.child, ctx))

    if isinstance(expr, Log):
        return torch.log(torch.clamp(_evaluate(expr.child, ctx), min=1e-30))

    if isinstance(expr, Tanh):
        return torch.tanh(_evaluate(expr.child, ctx))

    if isinstance(expr, Sinh):
        return torch.sinh(_evaluate(expr.child, ctx))

    if isinstance(expr, Cosh):
        return torch.cosh(_evaluate(expr.child, ctx))

    if isinstance(expr, Sign):
        return torch.sign(_evaluate(expr.child, ctx))

    if isinstance(expr, Clamp):
        val = _evaluate(expr.child, ctx)
        lo = float(_evaluate(expr.low, ctx)) if expr.low else None
        hi = float(_evaluate(expr.high, ctx)) if expr.high else None
        return torch.clamp(val, min=lo, max=hi)

    if isinstance(expr, Conditional):
        cond = _evaluate(expr.condition, ctx)
        t = _evaluate(expr.if_true, ctx)
        f = _evaluate(expr.if_false, ctx)
        # Smooth switch for differentiability
        w = 0.5 * (1.0 + torch.tanh(expr.sharpness * cond))
        return w * t + (1.0 - w) * f

    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _fvm_div_product(multiply_expr: Multiply, ctx: DiscretizationContext) -> torch.Tensor:
    """
    Special FVM path: Div(coeff * Grad(field)).

    Uses harmonic-mean face interpolation for the coefficient.
    Handles scalar coefficients by expanding to mesh size.
    """
    # Try to identify coeff * Grad(field) pattern
    left = multiply_expr.left
    right = multiply_expr.right

    def _expand_coeff(coeff, n_nodes):
        """Expand 0-dim or scalar coefficient to a vector."""
        if coeff.dim() == 0:
            return coeff.expand(n_nodes)
        return coeff

    if isinstance(right, Grad):
        coeff = _evaluate(left, ctx)
        child_val = _evaluate(right.child, ctx)
        coeff = _expand_coeff(coeff, child_val.shape[0])
        grad_val = ctx.backend.gradient(child_val)
        face_coeff = ctx.backend.face_coefficients(coeff)
        return ctx.backend.divergence(face_coeff * grad_val)
    elif isinstance(left, Grad):
        coeff = _evaluate(right, ctx)
        child_val = _evaluate(left.child, ctx)
        coeff = _expand_coeff(coeff, child_val.shape[0])
        grad_val = ctx.backend.gradient(child_val)
        face_coeff = ctx.backend.face_coefficients(coeff)
        return ctx.backend.divergence(face_coeff * grad_val)
    else:
        # Not a simple coeff * Grad pattern — fall back
        val = _evaluate(multiply_expr, ctx)
        return ctx.backend.divergence(val)


# ═══════════════════════════════════════════════════════════════════════════
#  Assembly Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class AssemblyPipeline:
    """
    Orchestrates the full symbolic → discrete → boundary → residual pipeline.

    For each equation in a System, it:
    1. Evaluates the RHS expression to a tensor via DiscretizationContext
    2. Applies boundary conditions
    3. Returns the residual or derivative vector

    Parameters
    ----------
    system : System
        The coupled equation system.
    meshes : dict
        Maps domain name → CompositeMesh1D.
    backends : dict
        Maps domain name → DiscretizationBackend.
    """

    def __init__(self, system, meshes, backends):
        self.system = system
        self.meshes = meshes
        self.backends = backends

    def evaluate_rhs(
        self,
        equation_name: str,
        fields: Dict[str, torch.Tensor],
        params: Dict[str, Any],
        boundary_conditions: Optional[Dict[str, BoundarySet]] = None,
        y_old: Optional[Dict[str, torch.Tensor]] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ) -> torch.Tensor:
        """
        Evaluate the RHS of a single PDE.

        Returns the discrete RHS tensor (du/dt = rhs).
        """
        if equation_name in self.system.pdes:
            eq = self.system.pdes[equation_name]
        elif equation_name in self.system.algebraic:
            eq = self.system.algebraic[equation_name]
        else:
            raise KeyError(f"Equation '{equation_name}' not in system.")

        domain = eq.domain if hasattr(eq, 'domain') else "default"
        mesh = self.meshes.get(domain, list(self.meshes.values())[0])
        backend = self.backends.get(domain, list(self.backends.values())[0])

        ctx = DiscretizationContext(
            mesh=mesh,
            backend=backend,
            fields=fields,
            params=params,
            boundary_conditions=boundary_conditions or {},
            y_old=y_old,
            dt=dt,
            time=time,
        )

        if hasattr(eq, 'rhs'):
            rhs = ctx.evaluate(eq.rhs)
        else:
            rhs = ctx.evaluate(eq.residual)

        # Apply BCs
        bc_set = (boundary_conditions or {}).get(equation_name)
        if bc_set is not None and y_old is not None and dt is not None:
            rhs = self._apply_boundary_conditions(
                rhs, bc_set, fields, y_old, dt,
                equation_name, mesh, backend, ctx,
            )

        return rhs

    def evaluate_all(
        self,
        fields: Dict[str, torch.Tensor],
        params: Dict[str, Any],
        boundary_conditions: Optional[Dict[str, BoundarySet]] = None,
        y_old: Optional[Dict[str, torch.Tensor]] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the RHS of all equations in the system.

        Returns a dict mapping equation name → RHS tensor.
        """
        results = {}
        for name in self.system.names:
            results[name] = self.evaluate_rhs(
                name, fields, params,
                boundary_conditions=boundary_conditions,
                y_old=y_old, dt=dt, time=time,
            )
        return results

    def _apply_boundary_conditions(
        self,
        rhs: torch.Tensor,
        bc_set: BoundarySet,
        fields: Dict[str, torch.Tensor],
        y_old: Dict[str, torch.Tensor],
        dt: float,
        eq_name: str,
        mesh: CompositeMesh1D,
        backend: DiscretizationBackend,
        ctx: DiscretizationContext,
    ) -> torch.Tensor:
        """
        Apply boundary conditions by modifying the residual at boundary nodes.

        The approach converts BCs into DAE-style algebraic constraints at
        boundary points, consistent with implicit Newton solvers.
        """
        constrained = rhs.clone()
        field_vals = fields.get(eq_name, fields.get(list(fields.keys())[0]))
        old_vals = y_old.get(eq_name)

        if old_vals is None:
            return constrained

        # Map location names to indices
        location_map = {"left": 0, "right": -1}

        for location, bc in bc_set.items():
            if location not in location_map:
                continue

            idx = location_map[location]
            target = bc.evaluate(ctx) if hasattr(bc, 'evaluate') else bc.value
            if callable(target):
                target = target(ctx)

            if bc.kind == "dirichlet":
                residual = field_vals[idx] - target
            elif bc.kind == "neumann":
                grad = backend.gradient(field_vals)
                residual = grad[idx] - target
            elif bc.kind == "robin":
                grad = backend.gradient(field_vals)
                residual = bc.alpha * field_vals[idx] + bc.beta * grad[idx] - target
            else:
                # Custom residual
                residual = target

            # DAE-style enforcement
            constrained[idx] = (
                (field_vals[idx] - old_vals[idx]) / dt - residual / dt
            )

        return constrained
