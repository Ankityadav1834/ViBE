# General PDE Plug-In Pipeline

This document explains how equations are now meant to be added to this
codebase.

The goal is:

```text
model_registry.py = equation book
main.py           = physics context + solver wiring
pde_framework.py  = reusable operators, meshes, and PDE evaluators
```

A researcher should not need to edit the Newton solver to add a new equation.
They should usually add a state and an operator equation in `model_registry.py`,
then enable it from `run.py`.

## 1. The Big Picture

Every solved quantity is a state:

```text
cs_n
cs_p
electrolyte
stress
Lsei
temperature
```

The solver only understands one thing:

```text
dy/dt for the flat state vector
```

The PDE pipeline converts this:

```text
du/dt = -Div(flux) + source
```

or this:

```text
dc_s/dt = D_s * (d2c_s/dr2 + 2/r * dc_s/dr)
```

into a tensor derivative with the same size as the state.

## 2. Where Equations Are Defined

Executable equation definitions live in:

```text
model_registry.py
```

Look for:

```python
STATE_EQUATIONS = {
    ...
}
```

Each state is a `StateEquationSpec`. Spatial PDE states carry an
`operator=OperatorEquationSpec(...)`.

Example:

```python
"stress": StateEquationSpec(
    size=physics_attr("Nel"),
    operator=through_cell_flux_operator(
        "stress",
        parameters={
            "flux_coefficient": stress_flux_coefficient,
            "source": stress_source,
        },
    ),
)
```

The optional `equation`, `notes`, `discretization`, and old-style
`boundary_conditions` fields are for humans. They are not the executable source
of truth.

The executable source of truth is:

```python
operator=OperatorEquationSpec(...)
```

## 3. What Is Plain English And What Is Code

Plain English metadata:

```python
equation="dstress/dt = ..."
notes="Example add-on PDE..."
discretization={"domain": "same through-cell mesh as electrolyte"}
```

Useful for documentation, not used to discretize.

Coded equation:

```python
operator=OperatorEquationSpec(
    state_name="stress",
    variable_name="stress",
    domain="through_cell",
    evaluator="conservative",
    rhs=-Div(Parameter("flux_coefficient") * Grad(Variable("stress"))) + Parameter("source"),
    flux=Parameter("flux_coefficient") * Grad(Variable("stress")),
    source=Parameter("source"),
    parameters={...},
    boundary_conditions={...},
)
```

The helper `through_cell_flux_operator(...)` builds this common pattern for
you.

## 4. Available Equation Families

The code cannot magically solve every PDE in the universe. It can solve many
useful battery PDEs if they fit one of these families.

### 4.1 Conservative Through-Cell PDE

Use for most 1D transport/reaction equations:

```text
du/dt = -Div(flux) + source
```

Examples:

```text
electrolyte concentration
stress smoothing field
lithium plating amount
species balance in x direction
advection-diffusion if written as a flux
migration-diffusion if written as a flux
```

Flux can be almost anything expressible from variables and parameters:

```text
flux = -D Grad(u)
flux = -D Grad(u) + v u
flux = -D Grad(c) - kappa Grad(phi)
```

In code, use:

```python
through_cell_flux_operator(...)
```

### 4.2 Spherical Particle PDE

Use for solid particle diffusion:

```text
dc_s/dt = D_s * (d2c_s/dr2 + 2/r * dc_s/dr)
```

Current states using this:

```text
cs_n
cs_p
```

### 4.3 General Operator Expression

Use when the RHS can be directly written with available operators:

```text
du/dt = expression(u, Grad(u), Div(...), parameters)
```

This is flexible, but for finite volume you should prefer conservative flux
form when boundary fluxes matter.

### 4.4 Plain RHS Callback

Use for ODEs or algebraic-style states with no spatial operator:

```text
dz/dt = f(z, other states, current, temperature)
```

Add `rhs=my_rhs_function` in `StateEquationSpec`.

### 4.5 Custom Evaluator

Use when your equation has a new structure not covered above.

You add a new model class in `pde_framework.py`, then map its evaluator string
inside `BatteryPhysics._build_operator_pipeline()` in `main.py`.

## 5. Operator Vocabulary

From `pde_framework.py`:

```python
Variable("u")       # solved or auxiliary field
Parameter("D")      # runtime coefficient/source supplied by callback
Grad(expr)          # spatial gradient
Div(expr)           # spatial divergence
Laplacian(expr)     # Div(Grad(expr))
SphericalDiv(a, r)  # spherical radial divergence
```

Arithmetic works:

```python
-Div(Parameter("D") * Grad(Variable("u"))) + Parameter("source")
```

For complex nonlinear physics, do not force everything into symbolic nodes.
Put nonlinear battery kinetics in a callback and expose the result as a
`Parameter`.

## 6. What Source Means

`source` means local generation or consumption at each mesh point.

It is not a special hard-coded battery term. It is just:

```text
local rate added to du/dt
```

For lithium plating:

```text
source = plating_rate - stripping_rate
```

For electrolyte:

```text
source = reaction source in negative electrode
       + zero in separator
       + reaction source in positive electrode
```

The source can be nonlinear and coupled:

```python
def my_source(physics, y_flat, i_app, p, context):
    u = physics.state(y_flat, "my_state")
    ce = context["ce_state"]
    T = physics.state(y_flat, "temperature")
    return torch.exp(-1000.0 / T) * ce - 1e-4 * u
```

## 7. What Flux/Diffusion Means

Diffusion is one possible flux:

```text
flux = -D Grad(u)
du/dt = -Div(flux) + source
```

But the pipeline is not limited to pure diffusion. The conservative evaluator
only asks for `flux`.

Examples:

```text
diffusion:          flux = -D Grad(u)
advection:          flux = v u
advection-diff:     flux = -D Grad(u) + v u
migration-diff:     flux = -D Grad(c) - mobility*c*Grad(phi)
```

For finite volume, coefficients in a flux often need to live on faces. The
helper:

```python
physics.through_cell_flux_coefficient(D)
```

returns `-D` in the correct shape for the active through-cell discretization.

## 8. Meshes And Regions

Executable domain names are coded strings:

```text
through_cell
negative_particle
positive_particle
```

They are mapped to actual meshes in `main.py`.

Through-cell region order is:

```text
[negative electrode][separator][positive electrode]
```

Use these slices in callbacks:

```python
neg = slice(0, physics.Nx_n)
sep = slice(physics.Nx_n, physics.Nx_n + physics.Nx_s)
pos = slice(physics.Nx_n + physics.Nx_s, physics.Nel)
```

Example:

```python
source = torch.zeros_like(u)
source[neg] = negative_electrode_source
source[sep] = 0.0
source[pos] = positive_electrode_source
```

That is how region-specific physics is currently made executable.

## 9. Runtime Callbacks

Most real battery equations are nonlinear and coupled. Therefore, coefficients
and sources are usually callbacks.

Callback signature:

```python
def callback(physics, y_flat, i_app, p, context):
    ...
```

Inputs:

```text
physics
    BatteryPhysics object. Use physics.state(...) and mesh sizes.

y_flat
    Current Newton iterate for one cell.

i_app
    Cell current.

p
    Parameter dictionary for one cell.

context
    Shared intermediate quantities computed in main.py.
```

Read states:

```python
cs_n = physics.state(y_flat, "cs_n")
ce = context["ce_state"]
T = physics.state(y_flat, "temperature")
```

Return a tensor with the correct size.

## 10. Current Equations In The Pipeline

Current operator states:

```text
cs_n
    domain="negative_particle"
    evaluator="spherical_particle"

cs_p
    domain="positive_particle"
    evaluator="spherical_particle"

electrolyte
    domain="through_cell"
    evaluator="conservative"
    flux = flux_coefficient * Grad(ce)
    source = electrolyte_source

stress
    domain="through_cell"
    evaluator="conservative"
    flux = flux_coefficient * Grad(stress)
    source = coupling*(ce - ce_0) - relaxation*stress

sei
    rhs callback returns zero

Lsei
    rhs callback returns SEI growth rate computed in shared context

temperature
    rhs callback returns heat-generation temperature rate computed in shared context
```

Important: `main.py` still computes shared electrochemical quantities such as
reaction current, SEI side current, electrolyte diffusivity, and heat
generation. But the PDE evaluation for the current PDE states is routed through
the operator specs in `model_registry.py`, and scalar ODE-like states use RHS
callbacks registered in `model_registry.py`.

## 11. Adding A New Plug-In PDE

Suppose we want:

```text
du/dt = -Div(-D(u, T) Grad(u)) + S(u, ce, cs_n, I, T)
```

### Step 1: Pick A State Name

```text
my_state
```

### Step 2: Pick Domain And Size

For through-cell:

```python
size=physics_attr("Nel")
```

For negative particle:

```python
size=physics_attr("Nr_n")
```

For positive particle:

```python
size=physics_attr("Nr_p")
```

### Step 3: Write Flux Callback

```python
def my_flux_coefficient(physics, y_flat, _i_app, _p, _context):
    options = physics.state_equation_options.get("my_state", {})
    u = physics.state(y_flat, "my_state")

    D = torch.ones_like(u) * float(options.get("diffusivity", 1e-12))
    return physics.through_cell_flux_coefficient(D)
```

### Step 4: Write Source Callback

```python
def my_source(physics, y_flat, i_app, p, context):
    u = physics.state(y_flat, "my_state")
    ce = context["ce_state"]
    T = physics.state(y_flat, "temperature")

    source = torch.zeros_like(u)
    neg = slice(0, physics.Nx_n)

    source[neg] = 1e-8 * torch.abs(i_app) * ce[neg] - 1e-4 * u[neg]
    return source
```

### Step 5: Register State

Add to `STATE_EQUATIONS`:

```python
"my_state": StateEquationSpec(
    order=60,
    size=physics_attr("Nel"),
    initial=0.0,
    scale=1.0,
    nonnegative=True,
    enabled=lambda config: bool(config.get("my_state_options", {}).get("enabled", False)),
    options_key="my_state_options",
    dependencies=("electrolyte", "temperature"),
    operator=through_cell_flux_operator(
        "my_state",
        parameters={
            "flux_coefficient": my_flux_coefficient,
            "source": my_source,
        },
    ),
    equation="dmy_state/dt = -Div(-D Grad(my_state)) + source",
    notes="Human-readable explanation only.",
),
```

### Step 6: Enable In `run.py`

```python
config = {
    "my_state_options": {
        "enabled": True,
        "diffusivity": 1e-12,
    },
}
```

### Step 7: Add Output If Needed

```python
def _my_state_total(battery, y, _i_pack):
    u = battery.physics.state(y, "my_state")
    return torch.sum(u, dim=1)


DERIVED_OUTPUTS["my_state_total"] = DerivedOutputSpec(
    _my_state_total,
    requires_state=("my_state",),
)
```

## 12. Lithium Plating Example

Continuous form:

```text
dLi_plated/dt = -Div(flux_plating) + plating_source - stripping_source
flux_plating = -D_plating Grad(Li_plated)
```

Callbacks:

```python
def li_plating_flux_coefficient(physics, y_flat, _i_app, _p, _context):
    options = physics.state_equation_options.get("li_plating", {})
    li = physics.state(y_flat, "li_plating")

    D = torch.ones_like(li) * float(options.get("diffusivity", 1e-16))
    D[physics.Nx_n:] = 0.0

    return physics.through_cell_flux_coefficient(D)


def li_plating_source(physics, y_flat, i_app, p, context):
    options = physics.state_equation_options.get("li_plating", {})

    li = physics.state(y_flat, "li_plating")
    ce = context["ce_state"]

    source = torch.zeros_like(li)
    neg = slice(0, physics.Nx_n)

    k_plate = float(options.get("k_plate", 1e-10))
    k_strip = float(options.get("k_strip", 1e-6))

    charge_drive = torch.clamp(-i_app, min=0.0)
    plating = k_plate * charge_drive * ce[neg]
    stripping = k_strip * li[neg]

    source[neg] = plating - stripping
    return source
```

State registration:

```python
"li_plating": StateEquationSpec(
    order=60,
    size=physics_attr("Nel"),
    initial=0.0,
    scale=1.0,
    nonnegative=True,
    enabled=lambda config: bool(config.get("li_plating_options", {}).get("enabled", False)),
    options_key="li_plating_options",
    dependencies=("electrolyte", "cs_n", "temperature"),
    operator=through_cell_flux_operator(
        "li_plating",
        parameters={
            "flux_coefficient": li_plating_flux_coefficient,
            "source": li_plating_source,
        },
    ),
)
```

This is nonlinear/coupled if the callbacks use `ce`, `cs_n`, `T`, `Lsei`,
current, or overpotential.

## 13. Writing Nonlinear Coupled Physics

Put difficult formulas in callbacks.

Example:

```python
def nonlinear_source(physics, y_flat, i_app, p, context):
    u = physics.state(y_flat, "u")
    ce = context["ce_state"]
    T = physics.state(y_flat, "temperature")

    rate = p["k0"] * torch.exp(-p["Ea"] / (p["R_g"] * T))
    return rate * torch.sqrt(torch.clamp(ce, min=1e-12)) - 1e-4 * u
```

The operator expression stays readable:

```text
du/dt = -Div(flux) + source
```

The callback owns the nonlinear battery science.

## 14. Adding A New Category Of PDE

If the equation does not fit the available evaluators, add a new evaluator.

Example category:

```text
fourth-order phase-field equation
```

Implementation path:

1. Add expression helpers if needed in `pde_framework.py`.
2. Add a class such as `PhaseFieldPDEModel`.
3. Give it an `evaluate(...)` method returning one derivative tensor.
4. Add a branch in `BatteryPhysics._build_operator_pipeline()`:

```python
elif operator_spec.evaluator == "phase_field":
    model = PhaseFieldPDEModel(...)
```

5. Register states with:

```python
operator=OperatorEquationSpec(
    evaluator="phase_field",
    domain="through_cell",
    ...
)
```

The Newton solver still does not change.

## 15. Boundary Conditions

Use executable boundary conditions in the operator spec:

```python
boundary_conditions={
    "left": BoundaryCondition("neumann", 0.0),
    "right": BoundaryCondition("neumann", 0.0),
}
```

Kinds:

```text
neumann
    Flux/gradient condition.

dirichlet
    Fixed value condition.

residual
    Advanced: provide algebraic residual directly.
```

Boundary values can also be callbacks:

```python
BoundaryCondition("neumann", my_boundary_callback)
```

For particle states, `cs_n_surface_gradient` and `cs_p_surface_gradient` are
examples already in `model_registry.py`.

## 16. Checklist

1. Write the continuous equation.
2. Decide if it is ODE, conservative through-cell PDE, spherical particle PDE,
   general operator PDE, or needs a new evaluator.
3. Pick state name, size, initial value, scale, and nonnegative flag.
4. Write flux/coefficient callbacks.
5. Write source callbacks.
6. Register `StateEquationSpec` in `model_registry.py`.
7. Put the executable equation in `operator=...`.
8. Enable it in `run.py`.
9. Add derived outputs if useful.
10. Run a tiny one-cell smoke test.
11. Then run real simulations.

## 17. Practical Rules

Keep the operator expression simple.

Put complicated battery formulas in callbacks.

Use conservative flux form when possible:

```text
du/dt = -Div(flux) + source
```

Treat English fields as documentation only.

If you create a new `domain` or `evaluator` string, it must be mapped in
`main.py`.

The returned derivative must have the same size as the registered state.
