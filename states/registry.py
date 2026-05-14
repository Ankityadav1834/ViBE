from dataclasses import dataclass, field
import torch

def context_value(fn):
    fn._uses_model_context = True
    return fn

def option_value(name, default):
    @context_value
    def resolver(_physics, options): return options.get(name, default)
    return resolver

def physics_attr(name):
    @context_value
    def resolver(physics, _options): return getattr(physics, name)
    return resolver

@context_value
def electrolyte_initial(physics, _options):
    def initial(params, device, dtype):
        return physics.eps_vec.to(device=device, dtype=dtype) * params["ce_0"]
    return initial

def context_item(name):
    def resolver(_physics, _y_flat, _i_app, _p, context): return context[name]
    return resolver

def parameter_item(name):
    def resolver(_physics, _y_flat, _i_app, p, _context): return p[name]
    return resolver

def state_item(name):
    def resolver(physics, y_flat, _i_app, _p, _context): return physics.state(y_flat, name)
    return resolver

@dataclass(frozen=True)
class StateEquationSpec:
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

    def options(self, config): return config.get(self.options_key, {}) if self.options_key else {}
    def is_enabled(self, config): return bool(self.enabled(config)) if callable(self.enabled) else bool(self.enabled)
    def resolve(self, value, physics, config):
        if callable(value) and getattr(value, "_uses_model_context", False): return value(physics, self.options(config))
        return value
    def resolved_size(self, physics, config): return int(self.resolve(self.size, physics, config))
    def resolved_initial(self, physics, config): return self.resolve(self.initial, physics, config)
    def resolved_scale(self, physics, config): return self.resolve(self.scale, physics, config)


