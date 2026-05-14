"""
Implicit solver — Newton iteration for nonlinear PDE systems.

Uses backward Euler (BDF-1) with damped Newton iteration.
Jacobian computation via ``torch.func.jacrev`` for automatic
differentiation (GPU-accelerated).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from pde_sim.state.layout import StateLayout


class ImplicitSolver:
    """
    Implicit backward-Euler Newton solver for coupled PDE systems.

    Parameters
    ----------
    state_layout : StateLayout
        The flat state-vector layout.
    rhs_fn : callable(y_flat, params, y_old, dt) → Tensor
        Computes the full RHS (du/dt) given the flat state vector.
    device : torch.device
        Computation device.
    dtype : torch.dtype
        Floating-point precision.
    tol : float
        Newton convergence tolerance (inf-norm of scaled residual).
    max_iter : int
        Maximum Newton iterations per step.
    damping : float
        Jacobian regularization factor.
    """

    def __init__(
        self,
        state_layout: StateLayout,
        rhs_fn: Callable,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        tol: float = 1e-6,
        max_iter: int = 15,
        damping: float = 1e-12,
    ):
        self.layout = state_layout
        self.rhs_fn = rhs_fn
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping

        self.scale = state_layout.scale_vector(self.device, self.dtype)
        self.inv_scale = 1.0 / self.scale
        self.nn_mask = state_layout.nonnegative_mask(self.device)

    def step(
        self,
        y_curr: torch.Tensor,
        dt: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Take one implicit Euler step.

        Parameters
        ----------
        y_curr : Tensor, shape (state_size,)
            Current state vector.
        dt : float
            Time step size.
        params : dict or None
            Runtime parameters.

        Returns
        -------
        y_next : Tensor, shape (state_size,)
            New state vector.
        converged : bool
            Whether Newton converged.
        n_iters : int
            Number of Newton iterations taken.
        """
        y_next = y_curr.clone()
        params = params or {}

        for k in range(self.max_iter):
            # Compute residual: R = y_next - y_curr - dt * f(y_next)
            dy = self.rhs_fn(y_next, params, y_curr, dt)
            R = y_next - y_curr - dt * dy

            # Check convergence
            res_norm = torch.norm(R * self.inv_scale, p=float('inf'))
            if res_norm < self.tol:
                return y_next, True, k + 1

            # Compute Jacobian via AD
            def residual_fn(y):
                return y - y_curr - dt * self.rhs_fn(y, params, y_curr, dt)

            try:
                J = torch.func.jacrev(residual_fn)(y_next)
                J = J + torch.eye(J.shape[0], device=self.device, dtype=self.dtype) * self.damping
                delta = torch.linalg.solve(J, -R)
            except (torch.linalg.LinAlgError, RuntimeError):
                return y_next, False, k + 1

            # Line search with non-negativity enforcement
            alpha = 1.0
            for _ in range(5):
                y_test = y_next + alpha * delta
                if torch.any(y_test[self.nn_mask] < 0):
                    alpha *= 0.5
                    continue
                dy_test = self.rhs_fn(y_test, params, y_curr, dt)
                R_test = y_test - y_curr - dt * dy_test
                if torch.norm(R_test * self.inv_scale) < torch.norm(R * self.inv_scale):
                    break
                alpha *= 0.5

            y_next = y_next + alpha * delta

        return y_next, False, self.max_iter

    def step_batched(
        self,
        y_batch: torch.Tensor,
        dt: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Batched Newton step using torch.vmap for GPU parallelism.

        Parameters
        ----------
        y_batch : Tensor, shape (batch, state_size)
            Current states for all cells/instances.
        dt : float
            Time step size.
        params : dict or None
            Parameters (tensors should have batch dimension).

        Returns
        -------
        y_next : Tensor, shape (batch, state_size)
        converged : bool
        """
        y_next = y_batch.clone()
        inv_S = self.inv_scale

        for k in range(self.max_iter):
            dy = self._batched_rhs(y_next, params, y_batch, dt)
            R = y_next - y_batch - dt * dy

            res_norm = torch.norm(R * inv_S, p=float('inf'))
            if res_norm < self.tol:
                return y_next, True

            def residual_fn(y, y_old):
                return y - y_old - dt * self._single_rhs(y, params, y_old, dt)

            try:
                J = torch.vmap(
                    torch.func.jacrev(residual_fn, argnums=0),
                    in_dims=(0, 0),
                )(y_next, y_batch)
                J = J + torch.eye(J.shape[-1], device=self.device, dtype=self.dtype) * self.damping
                delta = torch.linalg.solve(J, -R)
            except (torch.linalg.LinAlgError, RuntimeError):
                return y_next, False

            # Damped update
            alpha = 1.0
            for _ in range(5):
                y_test = y_next + alpha * delta
                if torch.any(y_test[:, self.nn_mask] < 0):
                    alpha *= 0.5
                    continue
                dy_test = self._batched_rhs(y_test, params, y_batch, dt)
                R_test = y_test - y_batch - dt * dy_test
                if torch.norm(R_test * inv_S) < torch.norm(R * inv_S):
                    break
                alpha *= 0.5

            y_next = y_next + alpha * delta

        return y_next, False

    def _batched_rhs(self, y_batch, params, y_old_batch, dt):
        """Override point for batched RHS computation."""
        return self.rhs_fn(y_batch, params, y_old_batch, dt)

    def _single_rhs(self, y, params, y_old, dt):
        """Override point for single-instance RHS (used by vmap)."""
        return self.rhs_fn(y, params, y_old, dt)
