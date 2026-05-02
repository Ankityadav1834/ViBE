# Adding a New PDE

This project now has two pieces that make new PDEs easier to plug in:

- `StateLayout` in `pde_framework.py`: owns flat state-vector slices, initialization, scaling, and positivity rules.
- Operator models in `pde_framework.py`: let PDEs use `Grad`, `Div`, finite-volume operators, or Chebyshev operators without hand-writing every index.

## Selecting Discretization

Yes, you can select the spatial discretization method from `run.py`.

Add or change this key inside `config`:

```python
config = {
    'n_series': 4,
    'n_parallel': 4,
    'electrolyte_spatial_method': 'finite_volume',
}
```

Supported values are:

```python
'finite_volume'
'chebyshev'
```

If you do not provide the option, the default is:

```python
'finite_volume'
```

At the moment, the example `stress` PDE uses the same through-cell mesh and operator as electrolyte, so it follows the same `electrolyte_spatial_method` setting.

## Current Example: Stress PDE

The stress PDE is optional and configured in `run.py`:

```python
'stress_options': {
    'enabled': False,
    'initial': 0.0,
    'scale': 1e6,
    'diffusivity': 1e-12,
    'relaxation': 1e-4,
    'coupling': 1.0
}
```

Turn it on with:

```python
'enabled': True
```

The example equation is:

```text
dstress/dt = diffusion(stress) + coupling * (ce - ce_0) - relaxation * stress
```

This is mainly a framework example. You can replace the physics with your own stress model later.

## Files You Usually Touch

For a new PDE, you usually edit:

- `run.py`: expose configuration options.
- `main.py`: register the new state field and compute its derivative.
- `pde_framework.py`: only if you need a new reusable PDE class or new operator behavior.
- `solver.py`: usually no change, unless the new state has special constraints.

## Step 1: Add Configuration in `run.py`

Example:

```python
config = {
    'n_series': 4,
    'n_parallel': 4,
    'my_pde_options': {
        'enabled': True,
        'initial': 0.0,
        'scale': 1.0,
        'diffusivity': 1e-12,
        'source_gain': 1.0
    }
}
```

Use this config block to control whether the PDE is active and to store constants used by the equation.

## Step 2: Read the Options in `BatteryPhysics.__init__`

In `main.py`, inside `BatteryPhysics.__init__`, add:

```python
self.my_pde_options = config.get('my_pde_options', {})
self.my_pde_enabled = bool(self.my_pde_options.get('enabled', False))
self.my_pde_model = ElectrolytePDEModel(
    self.electrolyte_mesh,
    self.electrolyte_operators
) if self.my_pde_enabled else None
```

This example reuses the electrolyte mesh. If your PDE needs a different domain, create a new mesh and operator pair.

## Step 3: Register the State Field

In `BatteryPhysics._build_state_layout()`, register the new PDE field:

```python
if self.my_pde_enabled:
    layout.register(
        'my_pde',
        self.Nel,
        initial=self.my_pde_options.get('initial', 0.0),
        scale=self.my_pde_options.get('scale', 1.0),
        nonnegative=False,
    )
```

Important options:

- `name`: how you access the field later.
- `size`: number of state variables.
- `initial`: scalar, tensor, parameter name, or callable.
- `scale`: Newton/Jacobian scaling.
- `nonnegative`: `True` for concentration-like states, `False` for signed states like stress or potential.

After this, the field is automatically included in:

- state size
- initial state
- solver scale vector
- positivity mask
- derivative packing

## Step 4: Access the Field in the Physics Function

Inside `compute_derivatives_functional()`, access your field by name:

```python
my_pde = self.state(y_flat, 'my_pde')
```

Do not manually calculate slices unless absolutely necessary.

## Step 5: Compute the PDE RHS

A simple diffusion-source PDE can reuse `ElectrolytePDEModel`.

Example:

```python
def _evaluate_my_pde_rhs(self, my_pde, ce_state, p):
    diffusivity = torch.ones_like(my_pde) * float(
        self.my_pde_options.get('diffusivity', 1e-12)
    )
    source_gain = float(self.my_pde_options.get('source_gain', 1.0))
    source = source_gain * (ce_state - p['ce_0'])

    flux_coefficient = -diffusivity
    if self.discretization_methods['electrolyte'] == 'finite_volume':
        flux_coefficient = lambda ctx, coeff=diffusivity: -ctx.operators.face_coefficients(coeff)

    return self.my_pde_model.evaluate(
        my_pde,
        flux_coefficient=flux_coefficient,
        source=source,
    )
```

Why negative diffusivity here?

`ElectrolytePDEModel` is written in flux form:

```text
dstate/dt = -Div(J) + source
J = flux_coefficient * Grad(state)
```

For normal diffusion:

```text
J = -D * Grad(state)
```

So the coefficient passed to the model is `-D`.

## Step 6: Add the Derivative to the Dictionary

At the end of `compute_derivatives_functional()`, add your PDE derivative:

```python
derivatives = {
    'cs_n': dcs_n_out,
    'cs_p': dcs_p_out,
    'electrolyte': dce_out,
    'sei': torch.zeros(self.Nsei, device=self.device, dtype=y_flat.dtype),
    'Lsei': dLsei_dt,
    'temperature': dT_dt,
}

if self.my_pde_enabled:
    derivatives['my_pde'] = self._evaluate_my_pde_rhs(my_pde, ce_state, p)

return self.state_layout.pack(derivatives, device=self.device, dtype=y_flat.dtype)
```

If a registered field is not included in `derivatives`, it automatically receives a zero derivative.

## Step 7: Add Parameters if Needed

If your PDE needs a parameter from `p`, make sure it is included in the `keys` list inside `BatteryPhysics.__init__`.

Example:

```python
keys = [
    ...,
    'ce_0',
    'my_new_parameter',
]
```

The parameter must also exist in `get_standard_parameters()` or be provided through `overrides`.

## Step 8: Plot or Export the New Field

The solver now has a central output manager in `output_manager.py`.

Every registered state is saved automatically to:

```text
all_results.csv
```

The file uses long format:

```text
time, cell, kind, name, local_index, value
```

For states:

```text
kind = state
name = cs_n, cs_p, electrolyte, stress, ...
```

For derived outputs:

```text
kind = output
name = terminal_voltage, pack_voltage, force_from_stress, ...
```

You can access outputs in Python through:

```python
outputs = battery_solver.output_manager.evaluate(battery_solver.y, pack_current)
terminal_voltage = outputs['terminal_voltage']
```

If you want to add a derived output for a new PDE, register it with the output manager.

Example:

```python
def my_force_output(y, i_pack):
    stress = battery_solver.physics.state(y, 'stress')
    return torch.mean(stress, dim=1) * 0.1027

battery_solver.output_manager.register_output('my_force', my_force_output)
```

Now `my_force` will be included in `all_results.csv` whenever results are saved.

The built-in stress example already registers:

```text
stress_mean
stress_min
stress_max
force_from_stress
```

If you want to visualize a new output in plots, update `solver.py` plotting code. Saving to CSV does not require changing `solver.py`.

### Direct State Access

Example access:

```python
stress = self.battery.physics.state(y_t, 'stress')
```

For a new field:

```python
my_pde = self.battery.physics.state(y_t, 'my_pde')
```

Then add it to plots or CSV output.

## Minimal Checklist

1. Add config block in `run.py`.
2. Add `self.my_pde_options`, `self.my_pde_enabled`, and model setup in `BatteryPhysics.__init__`.
3. Register the state in `_build_state_layout()`.
4. Access it using `self.state(y_flat, 'my_pde')`.
5. Compute its derivative.
6. Add it to `derivatives`.
7. Add any needed parameters to the `keys` list and `get_standard_parameters()`.
8. Optionally add plotting/export.

## Quick Example Config

```python
config = {
    'n_series': 4,
    'n_parallel': 4,
    'electrolyte_spatial_method': 'finite_volume',
    'stress_options': {
        'enabled': True,
        'initial': 0.0,
        'scale': 1e6,
        'diffusivity': 1e-12,
        'relaxation': 1e-4,
        'coupling': 1.0
    }
}
```
