"""
Smoke test for the pde_sim framework.

Validates:
1. All imports work
2. Symbolic expression construction
3. Equation system creation
4. Mesh generation (all 3 methods)
5. Discretization backends
6. Assembly pipeline evaluation
7. State layout
8. Full simulation pipeline (heat equation)
"""

import sys
import os
# Ensure the workspace root is on sys.path
_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

import torch
torch.set_default_dtype(torch.float64)


def test_imports():
    """Verify all public API imports."""
    from pde_sim import (
        # Symbolic
        Expression, Constant, Field, Param,
        Grad, Div, Laplacian, Dt, SphericalDiv,
        Abs, Sqrt, Exp, Log, Tanh, Sinh, Cosh,
        ensure_expression,
        PDEEquation, AlgebraicEquation, System,
        # Mesh
        Interval, Region, CompositeDomain,
        Mesh1D, CompositeMesh1D,
        # BC
        DirichletBC, NeumannBC, RobinBC, CustomBC, BoundarySet,
        # Discretization
        DiscretizationBackend,
        FiniteVolumeBackend, FiniteDifferenceBackend, ChebyshevBackend,
        # Assembly
        AssemblyPipeline, DiscretizationContext,
        # State
        StateLayout, FieldSpec,
        # Solver
        ImplicitSolver, AdaptiveTimeStepper,
        # Output
        OutputManager, DerivedQuantity,
        # Config
        SimulationConfig, Simulation,
    )
    print("[PASS] All imports successful")


def test_symbolic():
    """Test expression construction and repr."""
    from pde_sim import Field, Param, Grad, Div, Dt, Constant

    c = Field("c", domain="test")
    D = Param("D")

    expr = Div(D * Grad(c))
    assert "Div" in repr(expr)
    assert "Grad" in repr(expr)

    pair = Dt(c) == Div(D * Grad(c))
    assert hasattr(pair, "lhs")
    assert hasattr(pair, "rhs")

    # Arithmetic
    expr2 = c + 1.0
    expr3 = 2.0 * c - D
    expr4 = c ** 2
    expr5 = -c

    print("[PASS] Symbolic expressions work correctly")


def test_system():
    """Test System creation."""
    from pde_sim import Field, Param, Grad, Div, Dt, System

    c = Field("c", domain="test", size=10)
    T = Field("T", domain="test", size=1)
    D = Param("D")
    Q = Param("Q")
    rho_Cp = Param("rho_Cp")

    sys = System({
        "c": Dt(c) == Div(D * Grad(c)),
        "T": Dt(T) == Q / rho_Cp,
    }, metadata={
        "c": {"initial_condition": 1.0, "scale": 1.0},
        "T": {"initial_condition": 300.0, "scale": 300.0},
    })

    assert len(sys) == 2
    assert "c" in sys.pdes
    assert "T" in sys.pdes
    assert sys.names == ["c", "T"]
    print("[PASS] System creation works correctly")


def test_mesh_fvm():
    """Test FVM mesh generation."""
    from pde_sim import CompositeDomain, CompositeMesh1D

    domain = CompositeDomain()
    domain.add_region("left", 0.5, 10)
    domain.add_region("right", 0.5, 10)

    mesh = CompositeMesh1D(domain, method="finite_volume")
    assert mesh.n_nodes == 20
    assert mesh.faces is not None
    assert mesh.n_faces == 21  # N+1 for FVM
    print("[PASS] FVM mesh works correctly")


def test_mesh_fdm():
    """Test FDM mesh generation."""
    from pde_sim import CompositeDomain, CompositeMesh1D

    domain = CompositeDomain()
    domain.add_region("rod", 1.0, 20)

    mesh = CompositeMesh1D(domain, method="finite_difference")
    assert mesh.n_nodes == 20
    assert mesh.faces is None
    print("[PASS] FDM mesh works correctly")


def test_mesh_chebyshev():
    """Test Chebyshev mesh generation."""
    from pde_sim import CompositeDomain, CompositeMesh1D

    domain = CompositeDomain()
    domain.add_region("seg1", 0.3, 8)
    domain.add_region("seg2", 0.4, 8)
    domain.add_region("seg3", 0.3, 8)

    mesh = CompositeMesh1D(domain, method="chebyshev")
    assert mesh.n_nodes == 24
    assert len(mesh.region_names) == 3
    print("[PASS] Chebyshev mesh works correctly")


def test_backends():
    """Test all three discretization backends."""
    from pde_sim import (
        CompositeDomain, CompositeMesh1D,
        FiniteVolumeBackend, FiniteDifferenceBackend, ChebyshevBackend,
    )

    # FVM
    d = CompositeDomain()
    d.add_region("r", 1.0, 20)
    mesh_fvm = CompositeMesh1D(d, method="finite_volume")
    fvm = FiniteVolumeBackend(mesh_fvm)
    vals = torch.linspace(0, 1, 20)
    grad = fvm.gradient(vals)
    div = fvm.divergence(grad)
    assert grad.shape[0] == 21  # N+1 faces
    assert div.shape[0] == 20

    # FDM
    d2 = CompositeDomain()
    d2.add_region("r", 1.0, 20)
    mesh_fdm = CompositeMesh1D(d2, method="finite_difference")
    fdm = FiniteDifferenceBackend(mesh_fdm)
    grad2 = fdm.gradient(vals)
    assert grad2.shape[0] == 20

    # Chebyshev
    d3 = CompositeDomain()
    d3.add_region("r", 1.0, 20)
    mesh_cheb = CompositeMesh1D(d3, method="chebyshev")
    cheb = ChebyshevBackend(mesh_cheb)
    grad3 = cheb.gradient(vals)
    assert grad3.shape[0] == 20

    print("[PASS] All discretization backends work correctly")


def test_assembly():
    """Test the assembly pipeline evaluation."""
    from pde_sim import (
        Field, Param, Grad, Div, Dt, System,
        CompositeDomain, CompositeMesh1D,
        FiniteVolumeBackend,
        AssemblyPipeline, DiscretizationContext,
    )

    c = Field("c", domain="test", size=10)
    D = Param("D")
    equations = System({"c": Dt(c) == Div(D * Grad(c))})

    domain = CompositeDomain()
    domain.add_region("test", 1.0, 10)
    mesh = CompositeMesh1D(domain, method="finite_volume")
    backend = FiniteVolumeBackend(mesh)

    pipeline = AssemblyPipeline(
        equations,
        meshes={"test": mesh},
        backends={"test": backend},
    )

    fields = {"c": torch.ones(10) * 300.0}
    params = {"D": 0.01}

    rhs = pipeline.evaluate_rhs("c", fields, params)
    assert rhs.shape == (10,)
    print("[PASS] Assembly pipeline works correctly")


def test_state_layout():
    """Test state layout pack/unpack."""
    from pde_sim import StateLayout

    layout = StateLayout()
    layout.register("c", 10, initial=1.0, scale=1.0, nonnegative=True)
    layout.register("T", 1, initial=300.0, scale=300.0)

    assert layout.total_size == 11
    assert "c" in layout
    assert "T" in layout

    y = layout.initial_state()
    assert y.shape == (11,)
    assert torch.allclose(layout.get(y, "c"), torch.ones(10))
    assert torch.allclose(layout.get(y, "T"), torch.tensor([300.0]))

    fields = layout.unpack(y)
    assert "c" in fields
    assert "T" in fields

    y_packed = layout.pack(fields)
    assert torch.allclose(y, y_packed)

    print("[PASS] State layout works correctly")


def test_output_manager():
    """Test the output manager."""
    from pde_sim import StateLayout, OutputManager, DerivedQuantity

    layout = StateLayout()
    layout.register("c", 5, initial=1.0)

    out = OutputManager(layout)
    out.register(DerivedQuantity(
        name="c_mean",
        fn=lambda fields, params, t: fields["c"].mean(),
        requires=("c",),
    ))

    state = layout.initial_state()
    derived = out.evaluate_derived(state)
    assert "c_mean" in derived
    assert abs(float(derived["c_mean"]) - 1.0) < 1e-10

    print("[PASS] Output manager works correctly")


def test_full_simulation():
    """Test a complete simulation run (heat equation)."""
    from pde_sim import (
        Field, Param, Grad, Div, Dt, System,
        SimulationConfig, DomainConfig, Simulation,
        DirichletBC, NeumannBC, BoundarySet,
    )
    from pde_sim.solver.time_stepper import TimeStepConfig

    T = Field("T", domain="rod", size=20)
    alpha = Param("alpha")

    equations = System({
        "T": Dt(T) == Div(alpha * Grad(T)),
    }, metadata={
        "T": {"initial_condition": 300.0, "scale": 300.0, "nonnegative": True},
    })

    config = SimulationConfig(
        domains=[DomainConfig("rod", [{"name": "rod", "length": 1.0, "n_nodes": 20}], "finite_volume")],
        parameters={"alpha": 0.01},
        initial_conditions={"T": 300.0},
        boundary_conditions={},
        time=TimeStepConfig(dt_init=0.1, dt_min=1e-6, dt_max=5.0),
        solver={"tol": 1e-6, "max_iter": 15},
        output={"filename": "test_results.csv"},
    )

    sim = Simulation(equations, config)
    result = sim.run(t_end=1.0, print_interval=50)

    assert len(result["times"]) > 1
    assert result["n_steps"] > 0

    # Cleanup
    if os.path.exists("test_results.csv"):
        os.remove("test_results.csv")

    print("[PASS] Full simulation pipeline works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("pde_sim Framework Smoke Tests")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_symbolic,
        test_system,
        test_mesh_fvm,
        test_mesh_fdm,
        test_mesh_chebyshev,
        test_backends,
        test_assembly,
        test_state_layout,
        test_output_manager,
        test_full_simulation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
