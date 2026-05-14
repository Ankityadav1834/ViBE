from .cs_n import spec as cs_n_spec
from .cs_p import spec as cs_p_spec
from .electrolyte import spec as electrolyte_spec
from .stress import spec as stress_spec
from .sei import spec_sei, spec_lsei
from .temperature import spec as temperature_spec
from .lithium_plating import spec as lithium_plating_spec

STATE_EQUATIONS = {
    "cs_n": cs_n_spec,
    "cs_p": cs_p_spec,
    "electrolyte": electrolyte_spec,
    "stress": stress_spec,
    "sei": spec_sei,
    "Lsei": spec_lsei,
    "temperature": temperature_spec,
    "lithium_plating": lithium_plating_spec,
}

def enabled_state_equations(physics, config):
    return {
        name: spec 
        for name, spec in sorted(STATE_EQUATIONS.items(), key=lambda item: item[1].order) 
        if spec.is_enabled(config)
    }

def evaluate_registered_rhs(physics, y_flat, i_app, p, context, derivatives):
    for name, spec in physics.state_equation_specs.items():
        if spec.rhs is None or name in derivatives: continue
        derivatives[name] = spec.rhs(physics, y_flat, i_app, p, context)
    return derivatives
