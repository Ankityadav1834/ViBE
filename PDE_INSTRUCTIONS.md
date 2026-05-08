# Modular State And Output Registry

The model now has one main file for solved states and equation hooks:

```text
model_registry.py
```

It contains two registries:

- `STATE_EQUATIONS`: things the Newton solver solves as part of the flat state vector.
- `DERIVED_OUTPUTS`: things saved to `all_results.csv` but not solved as states.

The current core states are already declared there:

- `cs_n`
- `cs_p`
- `electrolyte`
- optional `stress`
- optional `sei`
- optional `Lsei`
- `temperature`

`main.py` now builds `StateLayout` from this registry, so initial values,
scales, positivity rules, and slices are derived automatically.

## Adding A New Solved State

Add one entry to `STATE_EQUATIONS` in `model_registry.py`.

Minimal pattern:

```python
def my_state_rhs(physics, y_flat, i_app, p, context):
    my_state = physics.state(y_flat, "my_state")
    electrolyte = context["ce_state"]

    rate = 1e-4
    coupling = 0.1 * (electrolyte - p["ce_0"])
    return coupling - rate * my_state


STATE_EQUATIONS["my_state"] = StateEquationSpec(
    order=50,
    size=physics_attr("Nel"),
    initial=0.0,
    scale=1.0,
    nonnegative=False,
    enabled=lambda config: bool(config.get("my_state_options", {}).get("enabled", False)),
    options_key="my_state_options",
    rhs=my_state_rhs,
    dependencies=("electrolyte",),
    equation="dmy_state/dt = 0.1*(ce - ce_0) - rate*my_state",
    discretization={
        "domain": "same through-cell mesh as electrolyte",
        "method": "config['electrolyte_spatial_method']",
        "points": "Nx_n + Nx_s + Nx_p",
    },
    boundary_conditions={
        "left": {"type": "neumann", "value": 0.0},
        "right": {"type": "neumann", "value": 0.0},
    },
)
```

Then enable it from `run.py`:

```python
config = {
    "n_series": 2,
    "n_parallel": 1,
    "my_state_options": {
        "enabled": True,
    },
}
```

After that, the state is automatically included in:

- flat state vector layout
- initial state construction
- Newton scaling
- nonnegative masking
- derivative packing
- `all_results.csv` state rows

## Discretization Choices

For states that use the existing through-cell mesh, set:

```python
size=physics_attr("Nel")
```

and choose the method in `run.py`:

```python
config = {
    "electrolyte_spatial_method": "finite_volume",
}
```

Supported methods are:

```python
"finite_volume"
"finite_difference"
"chebyshev"
```

The number of through-cell points is still controlled by:

```python
discretization = {
    "Nx_n": 10,
    "Nx_s": 10,
    "Nx_p": 10,
}
```

For scalar ODE-like states, use:

```python
size=1
```

For a custom point count controlled from options, define a small resolver:

```python
@context_value
def my_state_size(_physics, options):
    return int(options.get("points", 20))
```

Then use:

```python
size=my_state_size
```

If the state uses a different mesh from electrolyte, its RHS function should
build or call the discretization it needs. The registry keeps the size,
method, point count, and boundary-condition metadata together so the equation
definition remains self-contained.

## Interconnected States

Use `dependencies` for documentation and use `context`/`physics.state(...)` in
the RHS for the actual coupling.

Useful RHS inputs:

```python
physics.state(y_flat, "state_name")  # any solved state
context["ce_state"]                  # real electrolyte concentration
context["ce_real"]                   # same electrolyte data before concatenation
context["dcs_n"]                     # core cs_n derivative
context["dcs_p"]                     # core cs_p derivative
context["dce"]                       # core electrolyte derivative
context["dT_dt"]                     # core temperature derivative
```

## Diffusion-Source PDE Helper

For an add-on through-cell PDE in flux form:

```text
dstate/dt = -Div(J) + source
J = -D Grad(state)
```

use:

```python
def my_diffusion_rhs(physics, y_flat, i_app, p, context):
    state = physics.state(y_flat, "my_diffusion_state")
    diffusivity = torch.ones_like(state) * 1e-12
    source = 0.1 * (context["ce_state"] - p["ce_0"])
    return physics.evaluate_diffusion_source_rhs(state, diffusivity, source)
```

The existing optional `stress` state is implemented this way in
`model_registry.py`.

## Adding A Derived Output

Derived outputs are values you want in the CSV but do not want Newton to solve
as independent states.

Add one entry to `DERIVED_OUTPUTS` in `model_registry.py`:

```python
def my_metric(battery, y, i_pack):
    my_state = battery.physics.state(y, "my_state")
    return torch.mean(my_state, dim=1)


DERIVED_OUTPUTS["my_metric"] = DerivedOutputSpec(
    my_metric,
    requires_state=("my_state",),
)
```

It will be written to `all_results.csv` as:

```text
kind = output
name = my_metric
```

The raw state itself is also saved automatically with:

```text
kind = state
name = my_state
```

## Minimal Checklist

1. Add a `StateEquationSpec` in `model_registry.py`.
2. Put its RHS function in the same file if it has dynamics.
3. Add an options block in `run.py` if the state should be configurable.
4. Add a `DerivedOutputSpec` for any non-state quantity you want exported.
5. Run a small smoke test before a long simulation.
