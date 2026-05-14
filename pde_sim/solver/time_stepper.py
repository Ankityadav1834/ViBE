"""
Adaptive time stepping — PID-controlled step-size selection.

Supports:
- Step doubling (Richardson extrapolation) for LTE estimation
- PID/I-controller for smooth step-size adaptation
- Safety factor and min/max clamping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class TimeStepConfig:
    """
    Configuration for adaptive time stepping.

    Parameters
    ----------
    dt_init : float
        Initial time step.
    dt_min : float
        Minimum allowed time step.
    dt_max : float
        Maximum allowed time step.
    safety : float
        Safety factor (< 1) to keep below the stability boundary.
    abstol : float
        Absolute error tolerance for LTE.
    reltol : float
        Relative error tolerance for LTE.
    growth_max : float
        Maximum step-size growth factor per step.
    shrink_min : float
        Minimum step-size shrink factor per step.
    """
    dt_init: float = 1.0
    dt_min: float = 1e-6
    dt_max: float = 100.0
    safety: float = 0.9
    abstol: float = 1e-5
    reltol: float = 1e-3
    growth_max: float = 2.0
    shrink_min: float = 0.2


class AdaptiveTimeStepper:
    """
    Adaptive time stepper with LTE-based step-size control.

    Uses step doubling: compares a full step with two half-steps
    to estimate the local truncation error. Adjusts dt using an
    I-controller.

    Parameters
    ----------
    solver : ImplicitSolver
        The nonlinear solver to use for each step.
    config : TimeStepConfig
        Step-size control parameters.
    """

    def __init__(self, solver, config: Optional[TimeStepConfig] = None):
        self.solver = solver
        self.config = config or TimeStepConfig()

    def advance(
        self,
        y: torch.Tensor,
        t: float,
        dt: float,
        params=None,
        scale: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Attempt one adaptive step from (t, y).

        Returns
        -------
        y_new : Tensor
            Accepted state (from the two half-steps, if accepted).
        t_new : float
            New time.
        dt_new : float
            Suggested next time step.
        accepted : bool
            Whether the step was accepted.
        error : float
            Normalized error estimate.
        """
        cfg = self.config

        # Full step
        y_full, ok_full, _ = self.solver.step(y, dt, params)
        if not ok_full:
            dt_new = max(dt * cfg.shrink_min, cfg.dt_min)
            return y, t, dt_new, False, float('inf')

        # Two half-steps
        dt_half = dt / 2.0
        y_half1, ok_h1, _ = self.solver.step(y, dt_half, params)
        if not ok_h1:
            dt_new = max(dt * cfg.shrink_min, cfg.dt_min)
            return y, t, dt_new, False, float('inf')

        y_half2, ok_h2, _ = self.solver.step(y_half1, dt_half, params)
        if not ok_h2:
            dt_new = max(dt * cfg.shrink_min, cfg.dt_min)
            return y, t, dt_new, False, float('inf')

        # Error estimation
        lte = torch.abs(y_half2 - y_full)
        if scale is None:
            scale = self.solver.scale
        weight = cfg.abstol * scale + cfg.reltol * torch.abs(y_half2)
        error_norm = torch.max(lte / weight).item()

        if error_norm <= 1.0:
            # Accept
            dt_factor = cfg.safety * (1.0 / (error_norm + 1e-10)) ** 0.5
            dt_factor = min(max(dt_factor, cfg.shrink_min), cfg.growth_max)
            dt_new = min(dt * dt_factor, cfg.dt_max)
            return y_half2, t + dt, dt_new, True, error_norm
        else:
            # Reject
            dt_factor = cfg.safety * (1.0 / error_norm) ** 0.5
            dt_factor = max(dt_factor, cfg.shrink_min)
            dt_new = max(dt * dt_factor, cfg.dt_min)
            return y, t, dt_new, False, error_norm

    def integrate(
        self,
        y0: torch.Tensor,
        t_start: float,
        t_end: float,
        params=None,
        callback: Optional[Callable] = None,
        max_steps: int = 100000,
    ) -> dict:
        """
        Integrate from t_start to t_end.

        Parameters
        ----------
        y0 : Tensor
            Initial state.
        t_start, t_end : float
            Time interval.
        params : dict or None
            Runtime parameters.
        callback : callable or None
            Called as callback(step, t, y, dt, error) after each accepted step.
            Return True to stop early.
        max_steps : int
            Safety limit on total attempts.

        Returns
        -------
        result : dict
            Keys: "times", "states", "errors", "dts", "n_steps", "n_rejected".
        """
        y = y0.clone()
        t = t_start
        dt = self.config.dt_init

        times = [t]
        states = [y.clone().cpu()]
        errors = []
        dts = []
        n_rejected = 0
        step_count = 0

        while t < t_end and step_count < max_steps:
            if t + dt > t_end:
                dt = t_end - t

            y_new, t_new, dt_new, accepted, error = self.advance(
                y, t, dt, params, self.solver.scale,
            )

            if accepted:
                y = y_new
                t = t_new
                times.append(t)
                states.append(y.clone().cpu())
                errors.append(error)
                dts.append(dt)

                if callback is not None:
                    if callback(len(times) - 1, t, y, dt, error):
                        break
            else:
                n_rejected += 1

            dt = dt_new
            step_count += 1

            if dt < self.config.dt_min:
                raise RuntimeError(
                    f"Time step below minimum ({self.config.dt_min}) at t={t:.6g}. "
                    "Physics may be too stiff."
                )

        return {
            "times": times,
            "states": states,
            "errors": errors,
            "dts": dts,
            "n_steps": len(times) - 1,
            "n_rejected": n_rejected,
        }
