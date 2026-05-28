"""
Experiment_AutogradSensitivity.py
==================================
Demonates EXACT sensitivity analysis of the VIBE pack simulation using
PyTorch automatic differentiation -- replacing 2*N_params forward simulations
with a SINGLE backward pass.
"""

import os
import gc
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Add parent folder to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import ImplicitBatterySolver
from controllers import build_controller
from pde_sim.output import OutputSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

PACK_CONFIG = {
    'n_series':   2,
    'n_parallel': 2,
    'topology':   'series_first',
    'device':     'cpu',
    'stress_options': {'enabled': False},
}

DISCRETIZATION = {
    'Nr_n': 10, 'Nr_p': 10,
    'Nx_n':  5, 'Nx_s':  5, 'Nx_p': 5,
    'Nsei': 1,
}

DIFF_PARAMS = [
    'Ds_n',       # Solid diffusivity anode
    'Ds_p',       # Solid diffusivity cathode
    'm_ref_n',    # Exchange current rate constant anode
    'm_ref_p',    # Exchange current rate constant cathode
    'kappa_sei',  # SEI ionic conductivity
    'hA',         # Convective cooling coefficient
    'R_contact',  # Contact resistance
    'R_bus',      # Busbar resistance
    'eps_e_n',    # Electrolyte porosity anode
]

LOSS_FUNCTIONS = {
    'sei_growth_rate':    'rate of SEI thickness growth [m/s], pack average',
    'temp_rise_rate':     'rate of temperature rise [K/s], pack average',
    'capacity_fade_rate': 'rate of capacity loss, proxy via sei_growth_rate',
}

N_WARMUP_STEPS = 50   # ~50s of simulation time
WARMUP_DT      = 1.0  # seconds per step
PACK_CURRENT   = 5.0  # A (1C discharge for a 5 Ah cell)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'simulation_result', 'autograd_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# HELPER: enable grad on specific params
# ============================================================================

def enable_param_grads(physics, param_keys):
    grad_tensors = []
    for k in param_keys:
        t = physics.params[k]
        t.requires_grad_(True)
        if t.grad is not None:
            t.grad.zero_()
        grad_tensors.append(t)
    return grad_tensors


def disable_param_grads(physics):
    for v in physics.params.values():
        v.requires_grad_(False)
        v.grad = None


# ============================================================================
# MODE 1: INSTANTANEOUS SENSITIVITY
# ============================================================================

def run_instantaneous_sensitivity(battery, I_pack=PACK_CURRENT):
    physics = battery.physics
    y = battery.y.detach().clone()   # snapshot -- grad does NOT flow through y

    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(y, I_pack)

    grad_tensors = enable_param_grads(physics, DIFF_PARAMS)
    dY = physics.batched_derivatives(y, I_cells)   # shape: [n_cells, state_size]

    layout = physics.state_layout

    sei_sl = layout.slice('Lsei')
    loss_sei = dY[:, sei_sl].mean()

    T_sl = layout.slice('temperature')
    loss_temp = dY[:, T_sl].mean()

    losses = {
        'sei_growth_rate':    loss_sei,
        'temp_rise_rate':     loss_temp,
        'capacity_fade_rate': loss_sei,
    }

    PRIMARY_LOSS = 'temp_rise_rate'

    results = {}
    for loss_name, loss_val in losses.items():
        grads = torch.autograd.grad(
            loss_val,
            grad_tensors,
            retain_graph=True,   # keep graph for next loss
            create_graph=False,
            allow_unused=True,
        )
        results[loss_name] = {}
        base_params = ImplicitBatterySolver.get_standard_parameters()
        for param_key, grad in zip(DIFF_PARAMS, grads):
            if grad is None:
                S = 0.0
            else:
                g_scalar = grad.mean().item()
                p0 = base_params[param_key]
                y0 = loss_val.item()
                if abs(y0) < 1e-30:
                    S = g_scalar * p0
                else:
                    S = g_scalar * p0 / y0   # dimensionless normalised sensitivity
            results[loss_name][param_key] = S

    disable_param_grads(physics)
    return results, {k: v.item() for k, v in losses.items()}, PRIMARY_LOSS


# ============================================================================
# MODE 2: TRAJECTORY SENSITIVITY
# ============================================================================

def run_trajectory_sensitivity(battery, I_pack=PACK_CURRENT, n_steps=20, dt=1.0):
    physics = battery.physics
    y = battery.y.detach().clone()

    grad_tensors = enable_param_grads(physics, DIFF_PARAMS)

    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(y, I_pack)

    T_sl = physics.state_layout.slice('temperature')
    loss_accum = torch.tensor(0.0, dtype=torch.float64)

    for step in range(n_steps):
        dY = physics.batched_derivatives(y, I_cells)  # [n_cells, state_size]
        loss_accum = loss_accum + dY[:, T_sl].mean() * dt
        with torch.no_grad():
            y = (y + dt * dY).detach()
            I_cells = battery.compute_effective_cell_currents(y, I_pack)

    grads = torch.autograd.grad(
        loss_accum,
        grad_tensors,
        create_graph=False,
        allow_unused=True,
    )

    results = {}
    base_params = ImplicitBatterySolver.get_standard_parameters()
    y0 = loss_accum.item()
    for param_key, grad in zip(DIFF_PARAMS, grads):
        if grad is None:
            results[param_key] = 0.0
        else:
            g_scalar = grad.mean().item()
            p0 = base_params[param_key]
            results[param_key] = g_scalar * p0 / y0 if abs(y0) > 1e-30 else g_scalar * p0

    disable_param_grads(physics)
    return results, y0


# ============================================================================
# FINITE DIFFERENCE COMPARISON
# ============================================================================

def fd_gradient_at_state(battery, param_key, I_pack=PACK_CURRENT, delta=0.01):
    physics = battery.physics
    y = battery.y.detach().clone()

    with torch.no_grad():
        I_cells = battery.compute_effective_cell_currents(y, I_pack)

    sei_sl = physics.state_layout.slice('Lsei')
    p_tensor = physics.params[param_key]
    p0_val = p_tensor.mean().item()

    def sei_rate(p_val):
        physics.params[param_key] = torch.full_like(p_tensor, p_val)
        dY = physics.batched_derivatives(y, I_cells)
        return dY[:, sei_sl].mean().item()

    f_pos = sei_rate(p0_val * (1 + delta))
    f_neg = sei_rate(p0_val * (1 - delta))
    f_0   = sei_rate(p0_val)

    physics.params[param_key] = p_tensor

    fd_deriv = (f_pos - f_neg) / (2 * delta * p0_val)
    S_fd = fd_deriv * p0_val / abs(f_0) if abs(f_0) > 1e-30 else fd_deriv * p0_val
    return S_fd, f_0


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  VIBE AutogradSensitivity -- Exact gradients via PyTorch autograd")
    print("=" * 70)

    print(f"\n[1/4] Building 2S2P pack and warming up state ({N_WARMUP_STEPS} steps) ...")
    battery = ImplicitBatterySolver(
        PACK_CONFIG, DISCRETIZATION, {},
        initial_state_mode='fully_charged',
    )

    with torch.no_grad():
        for step in range(N_WARMUP_STEPS):
            I_cells = battery.compute_effective_cell_currents(battery.y, PACK_CURRENT)
            y_new, ok = battery.basic_solver.newton_step(
                battery.y, WARMUP_DT, I_cells
            )
            if ok:
                battery.y = y_new

    T_now = battery.y[:, -1].mean().item() - 273.15
    print(f"    Warm-up done. Mean cell temperature: {T_now:.2f} degC")

    print(f"\n[2/4] Computing INSTANTANEOUS autograd sensitivity ...")
    t0 = time.perf_counter()
    ag_results, loss_vals, primary_loss = run_instantaneous_sensitivity(battery, PACK_CURRENT)
    t_ag = time.perf_counter() - t0
    print(f"    Done in {t_ag:.3f}s  (1 forward + 3 backward passes)")
    for k, v in loss_vals.items():
        print(f"    {k}: {v:.4e}")

    print(f"\n[3/4] Computing TRAJECTORY autograd sensitivity (20 steps) ...")
    t0 = time.perf_counter()
    traj_results, traj_loss = run_trajectory_sensitivity(battery, PACK_CURRENT, n_steps=20)
    t_traj = time.perf_counter() - t0
    print(f"    Done in {t_traj:.3f}s  cumulative loss = {traj_loss:.4e}")

    print(f"\n[4/4] Finite-difference validation (3 params, same state) ...")
    fd_comparison = {}
    validate_params = DIFF_PARAMS[:3]
    for pk in validate_params:
        S_fd, f0 = fd_gradient_at_state(battery, pk)
        S_ag = ag_results['sei_growth_rate'][pk]
        err  = abs(S_ag - S_fd) / (abs(S_fd) + 1e-12) * 100
        fd_comparison[pk] = {'autograd': S_ag, 'finite_diff': S_fd, 'error_pct': err}
        print(f"    {pk:15s}  autograd={S_ag:+.4f}  FD={S_fd:+.4f}  err={err:.2f}%")

    import csv
    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['param', 'autograd_sei', 'autograd_sei_normalized', 'autograd_temp', 'traj_sei'])
        for pk in DIFF_PARAMS:
            writer.writerow([
                pk,
                ag_results['sei_growth_rate'][pk],
                round(ag_results['sei_growth_rate'][pk], 5),
                round(ag_results['temp_rise_rate'][pk], 5),
                round(traj_results.get(pk, 0.0), 5),
            ])
    print(f"\n  Saved: {csv_path}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'gradient_magnitudes.npz'),
        param_keys     = np.array(DIFF_PARAMS),
        S_sei_instant  = np.array([ag_results['sei_growth_rate'][p] for p in DIFF_PARAMS]),
        S_temp_instant = np.array([ag_results['temp_rise_rate'][p]  for p in DIFF_PARAMS]),
        S_sei_traj     = np.array([traj_results.get(p, 0.0)         for p in DIFF_PARAMS]),
    )

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'Autograd Sensitivity Analysis -- VIBE 2S2P Pack\n'
        '(Exact PyTorch gradients, 1 backward pass per loss)',
        fontsize=12, fontweight='bold'
    )

    x      = np.arange(len(DIFF_PARAMS))
    labels = DIFF_PARAMS
    width  = 0.35

    S_sei  = [ag_results['sei_growth_rate'][p] for p in DIFF_PARAMS]
    S_temp = [ag_results['temp_rise_rate'][p]  for p in DIFF_PARAMS]
    S_traj = [traj_results.get(p, 0.0)         for p in DIFF_PARAMS]

    bars1 = axs[0].bar(x - width/2, S_sei,  width, label='SEI growth rate',  color='steelblue',  alpha=0.8)
    bars2 = axs[0].bar(x + width/2, S_temp, width, label='Temp rise rate',   color='firebrick',  alpha=0.8)
    axs[0].axhline(0, color='black', linewidth=0.8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    axs[0].set_ylabel('Normalised sensitivity  S_i = (dL/L) / (dp/p)')
    axs[0].set_title('Instantaneous sensitivity (1 backward pass each)')
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.4)

    bars3 = axs[1].bar(x - width/2, S_sei,  width, label='Instantaneous (1-step)', color='steelblue', alpha=0.8)
    bars4 = axs[1].bar(x + width/2, S_traj, width, label='Trajectory (20 steps)',  color='darkorange', alpha=0.8)
    axs[1].axhline(0, color='black', linewidth=0.8)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    axs[1].set_ylabel('Normalised sensitivity (SEI growth rate output)')
    axs[1].set_title('Instantaneous vs. Trajectory sensitivity')
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'Sensitivity_Comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    if fd_comparison:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        pkeys = list(fd_comparison.keys())
        xv    = np.arange(len(pkeys))
        ag_v  = [fd_comparison[p]['autograd']   for p in pkeys]
        fd_v  = [fd_comparison[p]['finite_diff'] for p in pkeys]
        ax2.bar(xv - 0.2, ag_v, 0.4, label='PyTorch autograd (exact)',  color='steelblue')
        ax2.bar(xv + 0.2, fd_v, 0.4, label='Finite difference (OAT 1%)', color='darkorange', alpha=0.7)
        ax2.set_xticks(xv)
        ax2.set_xticklabels(pkeys, fontsize=11)
        ax2.set_ylabel('Normalised sensitivity (SEI growth rate)')
        ax2.set_title('Validation: Autograd vs. Finite Difference\n(same state, same loss function)')
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        val_path = os.path.join(RESULTS_DIR, 'Autograd_vs_FD_Validation.png')
        plt.savefig(val_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {val_path}")

    print("\n" + "=" * 70)
    print(f"  AUTOGRAD SENSITIVITY RANKING  (output: {primary_loss})")
    print("=" * 70)
    ranked = sorted(DIFF_PARAMS, key=lambda p: abs(ag_results[primary_loss][p]), reverse=True)
    for rank, pk in enumerate(ranked, 1):
        S = ag_results[primary_loss][pk]
        print(f"  #{rank:2d}  {pk:20s}  S = {S:+.6f}")
    print("="*70)
    print(f"\n  Also showing SEI growth sensitivity (near-zero at fresh state):")
    for pk in ranked[:5]:
        S_sei = ag_results['sei_growth_rate'][pk]
        print(f"       {pk:20s}  S_sei = {S_sei:+.4e}")
    print(f"\n  Autograd time:       {t_ag:.3f}s  (ALL {len(DIFF_PARAMS)} params in 1 backward pass)")
    print(f"  FD equivalent time: ~{t_ag * len(DIFF_PARAMS) * 2:.1f}s")
    print(f"  Speedup:            ~{len(DIFF_PARAMS)*2:.0f}x")
    print(f"\n  Results directory: {os.path.abspath(RESULTS_DIR)}")


if __name__ == '__main__':
    main()
