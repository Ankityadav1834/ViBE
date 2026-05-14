from dataclasses import dataclass, field

import torch

from pde_framework import (
    BoundaryCondition,
    Div,
    Grad,
    OperatorEquationSpec,
    Parameter,
    Variable,
    spherical_diffusion_expression,
)


def context_value(fn):
    """
    Marks a value as needing the live BatteryPhysics object before it can be
    registered with StateLayout.
    """
    fn._uses_model_context = True
    return fn


def option_value(name, default):
    @context_value
    def resolver(_physics, options):
        return options.get(name, default)

    return resolver


def physics_attr(name):
    @context_value
    def resolver(physics, _options):
        return getattr(physics, name)

    return resolver


@context_value
def electrolyte_initial(physics, _options):
    def initial(params, device, dtype):
        return physics.eps_vec.to(device=device, dtype=dtype) * params["ce_0"]

    return initial


def stress_enabled(config):
    return bool(config.get("stress_options", {}).get("enabled", True))


def sei_enabled(config):
    return bool(config.get("sei_options", {}).get("enabled", True))


def context_item(name):
    def resolver(_physics, _y_flat, _i_app, _p, context):
        return context[name]

    return resolver


def parameter_item(name):
    def resolver(_physics, _y_flat, _i_app, p, _context):
        return p[name]

    return resolver


def state_item(name):
    def resolver(physics, y_flat, _i_app, _p, _context):
        return physics.state(y_flat, name)

    return resolver


@dataclass(frozen=True)
class StateEquationSpec:
    """
    One solved state in the model.

    Add a new entry to STATE_EQUATIONS to make it part of the flat solver state.
    If rhs is provided, BatteryPhysics will call it automatically after the core
    coupled electrochemical equations are evaluated.
    """

    order: int
    size: object
    initial: object = 0.0
    scale: object = 1.0
    nonnegative: bool = True
    enabled: object = True
    options_key: str = None
    rhs: object = None
    dependencies: tuple = ()
    operator: object = None
    discretization: dict = field(default_factory=dict)
    boundary_conditions: dict = field(default_factory=dict)
    equation: str = ""
    notes: str = ""

    def options(self, config):
        return config.get(self.options_key, {}) if self.options_key else {}

    def is_enabled(self, config):
        return bool(self.enabled(config)) if callable(self.enabled) else bool(self.enabled)

    def resolve(self, value, physics, config):
        if callable(value) and getattr(value, "_uses_model_context", False):
            return value(physics, self.options(config))
        return value

    def resolved_size(self, physics, config):
        return int(self.resolve(self.size, physics, config))

    def resolved_initial(self, physics, config):
        return self.resolve(self.initial, physics, config)

    def resolved_scale(self, physics, config):
        return self.resolve(self.scale, physics, config)


@dataclass(frozen=True)
class DerivedOutputSpec:
    """
    A reported quantity that is not solved as an independent state.
    These values are written to all_results.csv with kind='output'.
    """

    fn: object
    enabled: object = True
    requires_state: tuple = ()
    notes: str = ""

    def is_enabled(self, battery):
        if callable(self.enabled):
            return bool(self.enabled(battery))
        return bool(self.enabled)


def stress_flux_coefficient(physics, y_flat, _i_app, _p, _context):
    options = physics.state_equation_options.get("stress", {})
    stress = physics.state(y_flat, "stress")
    diffusivity = torch.ones_like(stress) * float(options.get("diffusivity", 1e-12))
    return physics.through_cell_flux_coefficient(diffusivity)


def stress_source(physics, y_flat, _i_app, p, context):
    options = physics.state_equation_options.get("stress", {})
    ce_state = context["ce_state"]
    relaxation = float(options.get("relaxation", 1e-4))
    coupling = float(options.get("coupling", 1.0))
    stress = physics.state(y_flat, "stress")
    return coupling * (ce_state - p["ce_0"]) - relaxation * stress


def cs_n_surface_gradient(_physics, _y_flat, _i_app, p, context):
    return context["flux_n"] / p["Ds_n"]


def cs_p_surface_gradient(_physics, _y_flat, _i_app, p, context):
    return context["flux_p"] / p["Ds_p"]


def sei_rhs(physics, y_flat, _i_app, _p, _context):
    return torch.zeros(physics.Nsei, device=physics.device, dtype=y_flat.dtype)


def lsei_rhs(_physics, _y_flat, _i_app, _p, context):
    return context["dLsei_dt"]


def temperature_rhs(_physics, _y_flat, _i_app, _p, context):
    return context["dT_dt"]


def spherical_particle_operator(state_name, domain):
    if state_name == "cs_n":
        diffusivity = parameter_item("Ds_n")
        radius = parameter_item("Rs_n")
        surface_value = cs_n_surface_gradient
    elif state_name == "cs_p":
        diffusivity = parameter_item("Ds_p")
        radius = parameter_item("Rs_p")
        surface_value = cs_p_surface_gradient
    else:
        diffusivity = parameter_item("diffusivity")
        radius = parameter_item("radius")
        surface_value = 0.0

    return OperatorEquationSpec(
        state_name=state_name,
        variable_name=state_name,
        domain=domain,
        evaluator="spherical_particle",
        method="chebyshev",
        rhs=spherical_diffusion_expression(state_name),
        parameters={
            "diffusivity": diffusivity,
            "particle_radius": radius,
        },
        boundary_conditions={
            "center": BoundaryCondition("neumann", 0.0),
            "surface": BoundaryCondition("neumann", surface_value),
        },
    )


def through_cell_flux_operator(
    state_name,
    variable_name=None,
    parameters=None,
    values=None,
    time_values=None,
    flux=None,
    source=None,
):
    variable_name = variable_name or state_name
    flux = flux or (Parameter("flux_coefficient") * Grad(Variable(variable_name)))
    source = source or Parameter("source")
    return OperatorEquationSpec(
        state_name=state_name,
        variable_name=variable_name,
        domain="through_cell",
        evaluator="conservative",
        method="config['electrolyte_spatial_method']",
        rhs=-Div(flux) + source,
        values=values,
        time_values=time_values,
        parameters=parameters or {},
        flux=flux,
        source=source,
        boundary_conditions={
            "left": BoundaryCondition("neumann", 0.0),
            "right": BoundaryCondition("neumann", 0.0),
        },
    )


def through_cell_diffusion_operator(state_name, variable_name=None, parameters=None, values=None, time_values=None):
    return through_cell_flux_operator(
        state_name,
        variable_name=variable_name,
        parameters=parameters,
        values=values,
        time_values=time_values,
    )


STATE_EQUATIONS = {
    "cs_n": StateEquationSpec(
        order=10,
        size=physics_attr("Nr_n"),
        initial=29866.0,
        scale=30000.0,
        nonnegative=True,
        operator=spherical_particle_operator("cs_n", "negative_particle"),
        equation="dcs_n/dt = Ds_n * (d2cs_n/dr2 + 2/r * dcs_n/dr)",
        notes="Negative-electrode solid concentration. Core RHS is computed in BatteryPhysics.",
        discretization={
            "domain": "negative particle radius",
            "method": "chebyshev",
            "points": "discretization['Nr_n']",
        },
        boundary_conditions={
            "center": "symmetry",
            "surface": "reaction flux including SEI side current",
        },
    ),
    "cs_p": StateEquationSpec(
        order=20,
        size=physics_attr("Nr_p"),
        initial=17038.0,
        scale=30000.0,
        nonnegative=True,
        operator=spherical_particle_operator("cs_p", "positive_particle"),
        equation="dcs_p/dt = Ds_p * (d2cs_p/dr2 + 2/r * dcs_p/dr)",
        notes="Positive-electrode solid concentration. Core RHS is computed in BatteryPhysics.",
        discretization={
            "domain": "positive particle radius",
            "method": "chebyshev",
            "points": "discretization['Nr_p']",
        },
        boundary_conditions={
            "center": "symmetry",
            "surface": "reaction flux",
        },
    ),
    "electrolyte": StateEquationSpec(
        order=30,
        size=physics_attr("Nel"),
        initial=electrolyte_initial,
        scale=1000.0,
        nonnegative=True,
        dependencies=("cs_n", "cs_p"),
        operator=through_cell_diffusion_operator(
            "electrolyte",
            variable_name="ce",
            values=context_item("ce_state"),
            time_values=state_item("electrolyte"),
            parameters={
                "flux_coefficient": context_item("electrolyte_flux_coefficient"),
                "source": context_item("electrolyte_source"),
            },
        ),
        equation="delectrolyte/dt = -Div(-Deff * Grad(ce)) + reaction_source",
        notes="Through-cell electrolyte concentration times porosity. Core RHS is computed in BatteryPhysics.",
        discretization={
            "domain": "negative electrode + separator + positive electrode",
            "method": "config['electrolyte_spatial_method']",
            "points": "Nx_n + Nx_s + Nx_p",
        },
        boundary_conditions={
            "left": {"type": "neumann", "value": 0.0},
            "right": {"type": "neumann", "value": 0.0},
            "interfaces": "continuity",
        },
    ),
    "stress": StateEquationSpec(
        order=40,
        size=physics_attr("Nel"),
        initial=option_value("initial", 0.0),
        scale=option_value("scale", 1e6),
        nonnegative=False,
        enabled=stress_enabled,
        options_key="stress_options",
        operator=through_cell_diffusion_operator(
            "stress",
            parameters={
                "flux_coefficient": stress_flux_coefficient,
                "source": stress_source,
            },
        ),
        dependencies=("electrolyte",),
        equation="dstress/dt = -Div(-D_stress * Grad(stress)) + coupling*(ce - ce_0) - relaxation*stress",
        notes="Example add-on PDE. Add another entry like this for a new coupled state.",
        discretization={
            "domain": "same through-cell mesh as electrolyte",
            "method": "config['electrolyte_spatial_method']",
            "points": "Nx_n + Nx_s + Nx_p",
        },
        boundary_conditions={
            "left": {"type": "neumann", "value": 0.0},
            "right": {"type": "neumann", "value": 0.0},
        },
    ),
    "sei": StateEquationSpec(
        order=80,
        size=physics_attr("Nsei"),
        initial=0.0,
        scale=1.0,
        nonnegative=True,
        enabled=sei_enabled,
        options_key="sei_options",
        rhs=sei_rhs,
        equation="dsei/dt = 0",
        notes="Reserved SEI state vector. Current model keeps it algebraic/zero RHS.",
    ),
    "Lsei": StateEquationSpec(
        order=90,
        size=1,
        initial="Lsei_0",
        scale=1e-8,
        nonnegative=True,
        enabled=sei_enabled,
        options_key="sei_options",
        dependencies=("cs_n",),
        rhs=lsei_rhs,
        equation="dLsei/dt = i_side * Msei / (2 * F * rho_sei)",
        notes="SEI thickness evolved by the core side-reaction model.",
    ),
    "temperature": StateEquationSpec(
        order=1000,
        size=1,
        initial="T_amb",
        scale=300.0,
        nonnegative=True,
        rhs=temperature_rhs,
        equation="dT/dt = heat_generation / (rho_Cp * Vol_cell) plus pack thermal coupling",
        notes="Cell temperature. Kept last because the solver adds pack thermal coupling to the last state.",
    ),
}


def enabled_state_equations(physics, config):
    return {
        name: spec
        for name, spec in sorted(STATE_EQUATIONS.items(), key=lambda item: item[1].order)
        if spec.is_enabled(config)
    }


def evaluate_registered_rhs(physics, y_flat, i_app, p, context, derivatives):
    for name, spec in physics.state_equation_specs.items():
        if spec.rhs is None or name in derivatives:
            continue
        derivatives[name] = spec.rhs(physics, y_flat, i_app, p, context)
    return derivatives


def _cell_current(battery, y, i_pack):
    if torch.is_tensor(i_pack):
        return i_pack.view(battery.n_cells)
    return battery.compute_effective_cell_currents(y, float(i_pack)).flatten()


def _terminal_voltage(battery, y, i_pack):
    current = _cell_current(battery, y, i_pack).view(battery.n_cells, 1)
    return battery.get_exact_terminal_voltages(y, current)


def _pack_voltage(battery, y, i_pack):
    cell_voltages = _terminal_voltage(battery, y, i_pack).view(
        battery.n_series,
        battery.n_parallel,
    )
    return torch.sum(torch.mean(cell_voltages, dim=1)).reshape(1)


def _soc(battery, y, _i_pack):
    cs_n = battery.physics.state(y, "cs_n")
    cs_max = battery.physics.params["cs_max_n"]
    return ((cs_n[:, -1:] / cs_max - 0.01) / 0.94).flatten()


def _capacity_fade_pct(battery, y, _i_pack):
    Lsei = battery.physics.state(y, "Lsei")
    p = battery.physics.params
    delta_Lsei = Lsei - p["Lsei_0"]

    fade = 2.0 * (p["rho_sei"] / p["Msei"]) * delta_Lsei * (3.0 / p["Rs_n"]) / p["cs_max_n"]
    return (fade * 100.0).flatten()


def _get_c_bar_data(battery, y):
    cs_n = battery.physics.state(y, "cs_n")
    r_ref = battery.physics.r_n_ref.to(battery.device).unsqueeze(0)

    r_faces = torch.zeros((1, r_ref.shape[1] + 1), device=r_ref.device)
    r_faces[:, 1:-1] = 0.5 * (r_ref[:, :-1] + r_ref[:, 1:])
    r_faces[:, -1] = 1.0

    v = r_faces[:, 1:] ** 3 - r_faces[:, :-1] ** 3
    cv = torch.cumsum(v, dim=1)
    cv2 = torch.cumsum(cs_n * v, dim=1)

    c_bar_r = cv2 / cv
    c_bar = c_bar_r[:, -1:]
    return cs_n, c_bar_r, c_bar


def _dis_stress_vm_peak(battery, y, _i_pack):
    cs_n, c_bar_r, c_bar = _get_c_bar_data(battery, y)

    E = 15e9
    nu = 0.3
    Omega = 3.17e-6
    pf = E * Omega / (3.0 * (1.0 - nu))

    sr = 2.0 * pf * (c_bar - c_bar_r)
    sth = pf * (2.0 * c_bar + c_bar_r - 3.0 * cs_n)
    svm = torch.abs(sr - sth)
    return torch.max(svm, dim=1).values.flatten()


def _dis_stress_th_surf(battery, y, _i_pack):
    cs_n, _c_bar_r, c_bar = _get_c_bar_data(battery, y)

    E = 15e9
    nu = 0.3
    Omega = 3.17e-6
    pf = E * Omega / (3.0 * (1.0 - nu))

    c_surf = cs_n[:, -1]
    c_bar_surf = c_bar.squeeze(1)

    sth_surf = pf * (2.0 * c_bar_surf + c_bar_surf - 3.0 * c_surf)
    return sth_surf.flatten()


def _sei_mismatch_stress(battery, y, _i_pack):
    cs_n = battery.physics.state(y, "cs_n")
    c_surf = cs_n[:, -1]
    c_max = battery.physics.params["cs_max_n"].flatten()
    c_surf_ref = 0.8 * c_max

    E_sei = 10e9
    nu_sei = 0.25
    E_sei_b = E_sei / (1.0 - nu_sei)
    Omega = 3.17e-6
    sigma_intr = -0.5e9

    delta_c = c_surf - c_surf_ref
    sigma_mism = E_sei_b * (Omega / 3.0) * delta_c + sigma_intr
    return sigma_mism.flatten()


def _total_surf_stress(battery, y, i_pack):
    return _dis_stress_th_surf(battery, y, i_pack) + _sei_mismatch_stress(battery, y, i_pack)


def _stress_mean(battery, y, _i_pack):
    return torch.mean(battery.physics.state(y, "stress"), dim=1)


def _stress_min(battery, y, _i_pack):
    return torch.min(battery.physics.state(y, "stress"), dim=1).values


def _stress_max(battery, y, _i_pack):
    return torch.max(battery.physics.state(y, "stress"), dim=1).values


def _force_from_stress(battery, y, _i_pack):
    stress = battery.physics.state(y, "stress")
    area = float(battery.config.get("stress_options", {}).get("force_area", battery.raw_params[0]["A"]))
    return torch.mean(stress, dim=1) * area


DERIVED_OUTPUTS = {
    "terminal_voltage": DerivedOutputSpec(_terminal_voltage),
    "cell_current": DerivedOutputSpec(_cell_current),
    "pack_voltage": DerivedOutputSpec(_pack_voltage),
    "temperature": DerivedOutputSpec(
        lambda battery, y, _i_pack: battery.physics.state(y, "temperature").flatten(),
        requires_state=("temperature",),
    ),
    "soc": DerivedOutputSpec(_soc, requires_state=("cs_n",)),
    "sei_thickness_nm": DerivedOutputSpec(
        lambda battery, y, _i_pack: battery.physics.state(y, "Lsei").flatten() * 1e9,
        requires_state=("Lsei",),
    ),
    "capacity_fade_pct": DerivedOutputSpec(_capacity_fade_pct, requires_state=("Lsei",)),
    "dis_stress_vm_peak": DerivedOutputSpec(_dis_stress_vm_peak, requires_state=("cs_n",)),
    "dis_stress_th_surf": DerivedOutputSpec(_dis_stress_th_surf, requires_state=("cs_n",)),
    "sei_mismatch_stress": DerivedOutputSpec(_sei_mismatch_stress, requires_state=("cs_n",)),
    "total_surf_stress": DerivedOutputSpec(_total_surf_stress, requires_state=("cs_n",)),
    "stress_mean": DerivedOutputSpec(_stress_mean, requires_state=("stress",)),
    "stress_min": DerivedOutputSpec(_stress_min, requires_state=("stress",)),
    "stress_max": DerivedOutputSpec(_stress_max, requires_state=("stress",)),
    "force_from_stress": DerivedOutputSpec(_force_from_stress, requires_state=("stress",)),
}
