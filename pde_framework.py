from dataclasses import dataclass, field

import torch


class Expression:
    def __add__(self, other):
        return Add(self, ensure_expression(other))

    def __radd__(self, other):
        return Add(ensure_expression(other), self)

    def __sub__(self, other):
        return Subtract(self, ensure_expression(other))

    def __rsub__(self, other):
        return Subtract(ensure_expression(other), self)

    def __mul__(self, other):
        return Multiply(self, ensure_expression(other))

    def __rmul__(self, other):
        return Multiply(ensure_expression(other), self)

    def __truediv__(self, other):
        return Divide(self, ensure_expression(other))

    def __rtruediv__(self, other):
        return Divide(ensure_expression(other), self)

    def __pow__(self, other):
        return Power(self, ensure_expression(other))

    def __neg__(self):
        return Negate(self)


@dataclass(frozen=True)
class Constant(Expression):
    value: float


@dataclass(frozen=True)
class Variable(Expression):
    name: str


@dataclass(frozen=True)
class Parameter(Expression):
    name: str


@dataclass(frozen=True)
class Grad(Expression):
    child: Expression


@dataclass(frozen=True)
class Div(Expression):
    child: Expression


@dataclass(frozen=True)
class SphericalDiv(Expression):
    child: Expression
    radius: Expression


@dataclass(frozen=True)
class Laplacian(Expression):
    child: Expression


@dataclass(frozen=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Subtract(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Divide(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Power(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Negate(Expression):
    child: Expression


def ensure_expression(value):
    if isinstance(value, Expression):
        return value
    return Constant(float(value))


def diffusion_source_expression(variable_name):
    """
    Conservative flux-source PDE:

        d(state)/dt = -Div(flux_coefficient * Grad(state)) + source

    A normal diffusion equation uses flux_coefficient = -D.
    """
    variable = Variable(variable_name)
    return -Div(Parameter("flux_coefficient") * Grad(variable)) + Parameter("source")


def conservative_flux_expression(flux_name="flux", source_name="source"):
    return -Div(Parameter(flux_name)) + Parameter(source_name)


def spherical_diffusion_expression(variable_name):
    return SphericalDiv(
        Parameter("diffusivity") * Grad(Variable(variable_name)),
        Parameter("radius"),
    )


Field = Variable
Param = Parameter


def grad(expr):
    return Grad(ensure_expression(expr))


def div(expr):
    return Div(ensure_expression(expr))


def laplacian(expr):
    return Laplacian(ensure_expression(expr))


def spherical_div(expr, radius):
    return SphericalDiv(ensure_expression(expr), ensure_expression(radius))


@dataclass(frozen=True)
class BoundaryCondition:
    """
    Boundary condition metadata used by operator PDE models.

    kind:
        "neumann" means the boundary flux/normal-gradient residual is enforced.
        "dirichlet" means the state value itself is constrained.
        "residual" means value is already an algebraic residual.
    value:
        scalar/tensor/callable. Callable values receive a DiscretizationContext
        for flux-source PDEs and a small dict for spherical particle PDEs.
    """

    kind: str
    value: object = 0.0


@dataclass(frozen=True)
class DomainSpec:
    name: str
    regions: tuple = ()
    method: object = None
    notes: str = ""


@dataclass(frozen=True)
class OperatorEquationSpec:
    """
    PyBaMM-inspired symbolic PDE declaration.

    State registry entries point at one of these specs. The BatteryPhysics layer
    supplies nonlinear coefficients/sources at runtime; this spec defines the
    operator form, region/domain, evaluator, and default boundary conditions.
    """

    state_name: str
    variable_name: str
    domain: str
    rhs: Expression
    evaluator: str = "general"
    method: object = None
    values: object = None
    time_values: object = None
    variables: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    flux: Expression = None
    source: Expression = None
    boundary_conditions: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class OperatorEvaluation:
    rhs: torch.Tensor
    flux: torch.Tensor = None
    context: object = None


@dataclass(frozen=True)
class StateFieldSpec:
    name: str
    size: int
    initial: object = 0.0
    scale: object = 1.0
    nonnegative: bool = True


class StateLayout:
    """
    Owns the flat state-vector layout so new PDE/state fields only need to be
    registered once. Slices, initial state, scaling, and derivative packing are
    then derived from the registry.
    """

    def __init__(self):
        self.fields = []
        self.slices = {}
        self.total_size = 0

    def register(self, name, size, initial=0.0, scale=1.0, nonnegative=True):
        if name in self.slices:
            raise ValueError(f"State field {name!r} is already registered.")
        size = int(size)
        if size <= 0:
            raise ValueError(f"State field {name!r} must have positive size.")
        start = self.total_size
        stop = start + size
        self.fields.append(StateFieldSpec(name, size, initial, scale, bool(nonnegative)))
        self.slices[name] = slice(start, stop)
        self.total_size = stop
        return self.slices[name]

    def __contains__(self, name):
        return name in self.slices

    def names(self):
        return [field.name for field in self.fields]

    def slice(self, name):
        return self.slices[name]

    def get(self, state, name):
        return state[..., self.slices[name]]

    def pack(self, values, *, device=None, dtype=None):
        chunks = []
        for field in self.fields:
            if field.name in values:
                value = values[field.name]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=device, dtype=dtype)
                else:
                    value = value.to(device=device, dtype=dtype)
                chunks.append(value.reshape(field.size))
            else:
                chunks.append(torch.zeros(field.size, device=device, dtype=dtype))
        return torch.cat(chunks, dim=0)

    def _resolve_value(self, value, params, device, dtype):
        if callable(value):
            value = value(params, device, dtype)
        elif isinstance(value, str):
            value = params[value]

        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype).reshape(-1)
        return torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)

    def _expand_field_value(self, field, value):
        if value.numel() == 1:
            return value.expand(field.size)
        if value.numel() != field.size:
            raise ValueError(
                f"State field {field.name!r} expected {field.size} values, "
                f"got {value.numel()}."
            )
        return value

    def initial_state(self, params_list, device, dtype):
        y = torch.zeros((len(params_list), self.total_size), device=device, dtype=dtype)
        for cell_idx, params in enumerate(params_list):
            for field in self.fields:
                value = self._resolve_value(field.initial, params, device, dtype)
                y[cell_idx, self.slices[field.name]] = self._expand_field_value(field, value)
        return y

    def scale_vector(self, device, dtype):
        scale = torch.ones(self.total_size, device=device, dtype=dtype)
        for field in self.fields:
            value = self._resolve_value(field.scale, {}, device, dtype)
            scale[self.slices[field.name]] = self._expand_field_value(field, value)
        return scale

    def nonnegative_mask(self, device):
        mask = torch.zeros(self.total_size, device=device, dtype=torch.bool)
        for field in self.fields:
            if field.nonnegative:
                mask[self.slices[field.name]] = True
        return mask


class CompositeMesh:
    """
    Stitched 1D mesh over multiple regions.

    Supports:
    - finite-volume style cell-centered uniform/non-uniform meshes
    - region-wise Chebyshev collocation nodes mapped to physical domains
    """

    def __init__(self, region_spec, method="finite_volume", device=None, dtype=torch.float64):
        self.region_spec = region_spec
        self.method = method
        self.device = device
        self.dtype = dtype

        self.region_names = list(region_spec.keys())
        self.region_slices = {}
        self.region_node_coordinates = {}
        self.region_dx = {}
        self.region_lengths = {}

        node_chunks = []
        dx_chunks = []
        labels = []
        face_positions = []

        cursor = 0
        position = 0.0
        for name, spec in region_spec.items():
            length = float(spec["length"])
            n_nodes = int(spec["N"])
            self.region_lengths[name] = length

            if "nodes" in spec:
                local_nodes = torch.as_tensor(spec["nodes"], device=device, dtype=dtype)
                local_nodes = torch.sort(local_nodes).values
            elif method == "chebyshev":
                idx = torch.arange(n_nodes, device=device, dtype=dtype)
                local_nodes = 0.5 * length * (1.0 - torch.cos(torch.pi * idx / (n_nodes - 1)))
            elif method == "finite_difference":
                local_nodes = torch.linspace(0.0, length, n_nodes, device=device, dtype=dtype)
            else:
                edges = torch.linspace(0.0, length, n_nodes + 1, device=device, dtype=dtype)
                local_nodes = 0.5 * (edges[:-1] + edges[1:])

            if "dx" in spec:
                local_dx = torch.as_tensor(spec["dx"], device=device, dtype=dtype)
            elif method == "chebyshev":
                dx_left = torch.empty_like(local_nodes)
                dx_left[0] = local_nodes[1] - local_nodes[0]
                dx_left[1:] = local_nodes[1:] - local_nodes[:-1]
                dx_right = torch.empty_like(local_nodes)
                dx_right[:-1] = local_nodes[1:] - local_nodes[:-1]
                dx_right[-1] = local_nodes[-1] - local_nodes[-2]
                local_dx = 0.5 * (dx_left + dx_right)
            elif method == "finite_difference":
                local_dx = torch.empty_like(local_nodes)
                local_dx[:] = length / (n_nodes - 1) if n_nodes > 1 else length
            else:
                local_dx = torch.empty_like(local_nodes)
                local_dx[:] = length / n_nodes

            start = cursor
            stop = cursor + n_nodes
            self.region_slices[name] = slice(start, stop)
            self.region_node_coordinates[name] = local_nodes + position
            self.region_dx[name] = local_dx

            node_chunks.append(local_nodes + position)
            dx_chunks.append(local_dx)
            labels.extend([name] * n_nodes)

            if method == "finite_volume":
                if "face_positions" in spec:
                    region_faces = torch.as_tensor(spec["face_positions"], device=device, dtype=dtype) + position
                else:
                    region_faces = torch.linspace(0.0, length, n_nodes + 1, device=device, dtype=dtype) + position
                if len(face_positions) > 0:
                    region_faces = region_faces[1:]
                face_positions.append(region_faces)

            cursor = stop
            position += length

        self.nodes = torch.cat(node_chunks)
        self.dx = torch.cat(dx_chunks)
        self.region_labels = labels
        self.n_nodes = self.nodes.numel()
        self.total_length = position

        if method == "finite_volume":
            self.face_positions = torch.cat(face_positions)
            self.n_faces = self.face_positions.numel()
        else:
            self.face_positions = None
            self.n_faces = self.n_nodes


class FiniteDifferenceOperators:
    """
    Builds FDM operators for a stitched 1D node-centered mesh.
    """

    def __init__(self, mesh):
        if mesh.method != "finite_difference":
            raise ValueError("FiniteDifferenceOperators require a finite_difference mesh.")
        self.mesh = mesh
        self.device = mesh.nodes.device
        self.dtype = mesh.nodes.dtype
        self.gradient_matrix = self._build_gradient_matrix()
        self.divergence_matrix = self._build_divergence_matrix()

    def _build_gradient_matrix(self):
        n_nodes = self.mesh.n_nodes
        G = torch.zeros((n_nodes, n_nodes), device=self.device, dtype=self.dtype)
        x = self.mesh.nodes
        if n_nodes > 1:
            for i in range(1, n_nodes - 1):
                G[i, i-1] = -0.5 / (x[i] - x[i-1])
                G[i, i+1] = 0.5 / (x[i+1] - x[i])
            G[0, 0] = -1.0 / (x[1] - x[0])
            G[0, 1] = 1.0 / (x[1] - x[0])
            G[-1, -2] = -1.0 / (x[-1] - x[-2])
            G[-1, -1] = 1.0 / (x[-1] - x[-2])
        return G

    def _build_divergence_matrix(self):
        # In 1D FDM, simple divergence is same as gradient matrix
        return self._build_gradient_matrix()

    def gradient(self, values):
        return self.gradient_matrix @ values

    def divergence(self, values):
        return self.divergence_matrix @ values

    def face_coefficients(self, coefficients):
        return coefficients



class FiniteVolumeOperators:
    """
    Builds generic gradient/divergence matrices for a stitched 1D cell-centered mesh.
    """

    def __init__(self, mesh):
        if mesh.method != "finite_volume":
            raise ValueError("FiniteVolumeOperators require a finite_volume mesh.")
        self.mesh = mesh
        self.device = mesh.nodes.device
        self.dtype = mesh.nodes.dtype
        self.gradient_matrix = self._build_gradient_matrix()
        self.divergence_matrix = self._build_divergence_matrix()

    def _build_gradient_matrix(self):
        n_faces = self.mesh.n_faces
        n_nodes = self.mesh.n_nodes
        if n_faces != n_nodes + 1:
            raise ValueError("Finite-volume meshes must provide N + 1 faces for N cell centers.")
        G = torch.zeros((n_faces, n_nodes), device=self.device, dtype=self.dtype)
        x = self.mesh.nodes

        # Boundary rows stay zero, giving the same no-flux padding used by the
        # original electrolyte finite-volume discretization.
        for face_idx in range(1, n_faces - 1):
            left = face_idx - 1
            right = face_idx
            spacing = x[right] - x[left]
            G[face_idx, left] = -1.0 / spacing
            G[face_idx, right] = 1.0 / spacing
        return G

    def _build_divergence_matrix(self):
        n_faces = self.mesh.n_faces
        n_nodes = self.mesh.n_nodes
        if n_faces != n_nodes + 1:
            raise ValueError("Finite-volume meshes must provide N + 1 faces for N cell centers.")
        D = torch.zeros((n_nodes, n_faces), device=self.device, dtype=self.dtype)
        dx = self.mesh.dx
        for node_idx in range(n_nodes):
            D[node_idx, node_idx] = -1.0 / dx[node_idx]
            D[node_idx, node_idx + 1] = 1.0 / dx[node_idx]
        return D

    @staticmethod
    def harmonic_mean(coefficients):
        return 2.0 * coefficients[:-1] * coefficients[1:] / (coefficients[:-1] + coefficients[1:] + 1e-20)

    def face_coefficients(self, coefficients):
        return torch.cat([
            coefficients[:1],
            self.harmonic_mean(coefficients),
            coefficients[-1:],
        ], dim=0)

    def laplacian_matrix(self, coefficients):
        coeff_faces = self.face_coefficients(coefficients)
        return self.divergence_matrix @ (coeff_faces.unsqueeze(1) * self.gradient_matrix)

    def gradient(self, values):
        return self.gradient_matrix @ values

    def divergence(self, values):
        return self.divergence_matrix @ values


class ChebyshevRegionOperators:
    """
    Region-wise Chebyshev collocation operators, preserving the current electrolyte PDE
    implementation structure while making it reusable.
    """

    def __init__(self, mesh):
        if mesh.method != "chebyshev":
            raise ValueError("ChebyshevRegionOperators require a chebyshev mesh.")
        self.mesh = mesh
        self.device = mesh.nodes.device
        self.dtype = mesh.nodes.dtype
        self.region_gradient = {}
        self.region_second = {}
        self.block_gradient = []
        self._build_region_matrices()

    def _build_cheb_matrix(self, n):
        x_ref = 0.5 * (1.0 - torch.cos(torch.pi * torch.arange(n, device=self.device, dtype=self.dtype) / (n - 1)))
        D = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        w = torch.ones(n, device=self.device, dtype=self.dtype)
        for i in range(n):
            diff = x_ref[i] - x_ref
            diff[i] = 1.0
            w[i] = 1.0 / torch.prod(diff)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x_ref[i] - x_ref[j])
        for i in range(n):
            D[i, i] = -torch.sum(D[i, :]) + D[i, i]
        return x_ref, D, D @ D

    def _build_region_matrices(self):
        for name in self.mesh.region_names:
            region_slice = self.mesh.region_slices[name]
            n = region_slice.stop - region_slice.start
            _, D_ref, D2_ref = self._build_cheb_matrix(n)
            length = self.mesh.region_lengths[name]
            D = D_ref / length
            D2 = D2_ref / (length ** 2)
            self.region_gradient[name] = D
            self.region_second[name] = D2
            self.block_gradient.append(D)

    def split_by_region(self, values):
        return {name: values[self.mesh.region_slices[name]] for name in self.mesh.region_names}

    def concatenate_by_region(self, region_values):
        return torch.cat([region_values[name] for name in self.mesh.region_names], dim=0)

    def gradient(self, values):
        region_values = self.split_by_region(values)
        return self.concatenate_by_region(
            {name: self.region_gradient[name] @ region_values[name] for name in self.mesh.region_names}
        )

    def divergence(self, values):
        region_values = self.split_by_region(values)
        return self.concatenate_by_region(
            {name: self.region_gradient[name] @ region_values[name] for name in self.mesh.region_names}
        )


class DiscretizationContext:
    def __init__(self, mesh, operators, variables, parameters, state=None, boundary_conditions=None):
        self.mesh = mesh
        self.operators = operators
        self.variables = variables
        self.parameters = parameters
        self.state = state or {}
        self.boundary_conditions = boundary_conditions or {}


def discretize(expr, context):
    if isinstance(expr, Constant):
        return torch.tensor(expr.value, device=context.mesh.nodes.device, dtype=context.mesh.nodes.dtype)

    if isinstance(expr, Variable):
        value = context.variables[expr.name]
        return value() if callable(value) else value

    if isinstance(expr, Parameter):
        value = context.parameters[expr.name]
        if callable(value):
            value = value(context)
        return value

    if isinstance(expr, Add):
        return discretize(expr.left, context) + discretize(expr.right, context)

    if isinstance(expr, Subtract):
        return discretize(expr.left, context) - discretize(expr.right, context)

    if isinstance(expr, Multiply):
        return discretize(expr.left, context) * discretize(expr.right, context)

    if isinstance(expr, Divide):
        return discretize(expr.left, context) / discretize(expr.right, context)

    if isinstance(expr, Power):
        return discretize(expr.left, context) ** discretize(expr.right, context)

    if isinstance(expr, Negate):
        return -discretize(expr.child, context)

    if isinstance(expr, Grad):
        return context.operators.gradient(discretize(expr.child, context))

    if isinstance(expr, Div):
        return context.operators.divergence(discretize(expr.child, context))

    if isinstance(expr, Laplacian):
        return context.operators.divergence(context.operators.gradient(discretize(expr.child, context)))

    if isinstance(expr, SphericalDiv):
        r = discretize(expr.radius, context)
        child_val = discretize(expr.child, context)
        # Spherical Div: 1/r^2 d/dr (r^2 J)
        # However, at r=0 it's usually handled by L'hopital.
        # But we can just use 1/r^2 * Div(r^2 * child) or chain rule.
        r_safe = r.clone()
        r_safe[r_safe == 0] = 1.0
        r_sq = r_safe ** 2
        r_sq_child = r_sq * child_val
        div_r_sq_child = context.operators.divergence(r_sq_child)
        result = div_r_sq_child / r_sq
        # L'hopital for r=0: Div(J) at r=0 is 3 * dJ/dr (for J ~ r)
        if hasattr(context.operators, 'mesh') and context.operators.mesh.nodes[0] == 0:
            result[0] = 3.0 * context.operators.gradient(child_val)[0]
        return result

    raise TypeError(f"Unsupported expression type: {type(expr)!r}")


def _as_boundary_condition(value):
    if isinstance(value, BoundaryCondition):
        return value
    if isinstance(value, dict):
        return BoundaryCondition(
            kind=value.get("kind", value.get("type", "neumann")),
            value=value.get("value", 0.0),
        )
    return BoundaryCondition(kind="neumann", value=value)


def _resolve_context_value(value, context):
    return value(context) if callable(value) else value


class OperatorPDEPipeline:
    """
    Registry of discretized PDE models.

    BatteryPhysics builds one of these once, then derivative evaluation asks the
    pipeline for each state's RHS. Adding a PDE becomes:

    1. Add an OperatorEquationSpec to the state registry.
    2. Provide runtime coefficients/sources in BatteryPhysics.
    3. Let the pipeline select the domain operators and boundary handling.
    """

    def __init__(self):
        self.models = {}

    def register(self, state_name, model):
        if state_name in self.models:
            raise ValueError(f"Operator PDE {state_name!r} is already registered.")
        self.models[state_name] = model

    def __contains__(self, state_name):
        return state_name in self.models

    def model(self, state_name):
        return self.models[state_name]

    def evaluate_state(self, state_name, **kwargs):
        return self.models[state_name].evaluate(**kwargs)


class GeneralOperatorPDEModel:
    """
    Generic expression evaluator.

    This is the most open-ended path: the RHS is any expression built from the
    supported expression nodes and runtime parameters. Use this for local ODEs,
    reaction terms, advection-like terms, or custom operator expressions that do
    not need finite-volume boundary-flux injection.
    """

    def __init__(self, mesh, operators, spec):
        self.mesh = mesh
        self.operators = operators
        self.spec = spec
        self.variable_name = spec.variable_name
        self.equation = spec.rhs

    def _merge_boundary_conditions(self, boundary_conditions):
        merged = dict(self.spec.boundary_conditions)
        if boundary_conditions:
            merged.update(boundary_conditions)
        return merged

    def _build_context(self, values, parameters, variables=None, boundary_conditions=None):
        variable_values = dict(variables or {})
        variable_values.setdefault(self.variable_name, values)
        return DiscretizationContext(
            mesh=self.mesh,
            operators=self.operators,
            variables=variable_values,
            parameters=dict(parameters or {}),
            boundary_conditions=self._merge_boundary_conditions(boundary_conditions),
        )

    def _evaluate_raw(self, values, context):
        rhs = discretize(self.equation, context)
        flux = discretize(self.spec.flux, context) if self.spec.flux is not None else None
        return OperatorEvaluation(rhs=rhs, flux=flux, context=context)

    def evaluate(
        self,
        values,
        time_values=None,
        parameters=None,
        variables=None,
        boundary_conditions=None,
        old_values=None,
        dt=None,
        return_details=False,
    ):
        if time_values is None:
            time_values = values

        context = self._build_context(values, parameters, variables, boundary_conditions)
        evaluation = self._evaluate_raw(values, context)
        rhs = evaluation.rhs

        if old_values is not None and dt is not None and evaluation.flux is not None:
            if isinstance(self.operators, ChebyshevRegionOperators):
                rhs = self.apply_chebyshev_boundary_constraints(
                    values=values,
                    time_values=time_values,
                    rhs=rhs,
                    flux=evaluation.flux,
                    old_values=old_values,
                    dt=dt,
                    boundary_conditions=context.boundary_conditions,
                )
            elif isinstance(self.operators, FiniteDifferenceOperators):
                rhs = self.apply_endpoint_boundary_constraints(
                    values=values,
                    time_values=time_values,
                    rhs=rhs,
                    flux=evaluation.flux,
                    old_values=old_values,
                    dt=dt,
                    boundary_conditions=context.boundary_conditions,
                    context=context,
                )

        if return_details:
            return OperatorEvaluation(rhs=rhs, flux=evaluation.flux, context=context)
        return rhs

    def apply_endpoint_boundary_constraints(
        self,
        values,
        time_values,
        rhs,
        flux,
        old_values,
        dt,
        boundary_conditions,
        context,
    ):
        constrained = rhs.clone()
        for location, index in (("left", 0), ("right", -1)):
            if location not in boundary_conditions:
                continue
            bc = _as_boundary_condition(boundary_conditions[location])
            target = _resolve_context_value(bc.value, context)
            if bc.kind == "dirichlet":
                residual = values[index] - target
            elif bc.kind == "residual":
                residual = target
            else:
                residual = flux[index] - target
            constrained[index] = (time_values[index] - old_values[index]) / dt - residual / dt
        return constrained

    def apply_chebyshev_boundary_constraints(
        self,
        values,
        time_values,
        rhs,
        flux,
        old_values,
        dt,
        boundary_conditions,
    ):
        if not isinstance(self.operators, ChebyshevRegionOperators):
            return rhs

        values_regions = self.operators.split_by_region(values)
        time_regions = self.operators.split_by_region(time_values)
        rhs_regions = {
            name: chunk.clone()
            for name, chunk in self.operators.split_by_region(rhs).items()
        }
        flux_regions = self.operators.split_by_region(flux)
        old_regions = self.operators.split_by_region(old_values)

        names = self.mesh.region_names
        bc_context = DiscretizationContext(
            self.mesh,
            self.operators,
            {self.variable_name: values},
            {},
            boundary_conditions=boundary_conditions,
        )

        first_name = names[0]
        left_bc = _as_boundary_condition(
            boundary_conditions.get("left", BoundaryCondition("neumann", 0.0))
        )
        left_value = _resolve_context_value(left_bc.value, bc_context)
        if left_bc.kind == "dirichlet":
            left_residual = values_regions[first_name][0] - left_value
        elif left_bc.kind == "residual":
            left_residual = left_value
        else:
            left_residual = flux_regions[first_name][0] - left_value
        rhs_regions[first_name][0] = (
            (time_regions[first_name][0] - old_regions[first_name][0]) / dt
            - left_residual / dt
        )

        last_name = names[-1]
        right_bc = _as_boundary_condition(
            boundary_conditions.get("right", BoundaryCondition("neumann", 0.0))
        )
        right_value = _resolve_context_value(right_bc.value, bc_context)
        if right_bc.kind == "dirichlet":
            right_residual = values_regions[last_name][-1] - right_value
        elif right_bc.kind == "residual":
            right_residual = right_value
        else:
            right_residual = flux_regions[last_name][-1] - right_value
        rhs_regions[last_name][-1] = (
            (time_regions[last_name][-1] - old_regions[last_name][-1]) / dt
            - right_residual / dt
        )

        for left_name, right_name in zip(names[:-1], names[1:]):
            rhs_regions[left_name][-1] = (
                (time_regions[left_name][-1] - old_regions[left_name][-1]) / dt
                - (values_regions[left_name][-1] - values_regions[right_name][0]) / dt
            )
            rhs_regions[right_name][0] = (
                (time_regions[right_name][0] - old_regions[right_name][0]) / dt
                - (flux_regions[left_name][-1] - flux_regions[right_name][0]) / dt
            )

        return self.operators.concatenate_by_region(rhs_regions)


class ConservativeFluxPDEModel(GeneralOperatorPDEModel):
    """
    Conservative 1D PDE evaluator:

        du/dt = -Div(flux) + source

    The flux expression can represent diffusion, migration, advection, or any
    combination that can be discretized on the selected 1D operators.
    """

    def __init__(self, mesh, operators, spec):
        super().__init__(mesh, operators, spec)
        if spec.flux is None:
            raise ValueError(f"Conservative PDE {spec.state_name!r} requires spec.flux.")

    def _with_finite_volume_boundary_flux(self, flux, context):
        if not isinstance(self.operators, FiniteVolumeOperators):
            return flux

        adjusted = flux.clone()
        for location, index in (("left", 0), ("right", -1)):
            if location not in context.boundary_conditions:
                continue
            bc = _as_boundary_condition(context.boundary_conditions[location])
            if bc.kind != "neumann":
                continue
            adjusted[index] = _resolve_context_value(bc.value, context)
        return adjusted

    def _evaluate_raw(self, values, context):
        flux = discretize(self.spec.flux, context)
        flux = self._with_finite_volume_boundary_flux(flux, context)
        source_expr = self.spec.source or Parameter("source")
        source = discretize(source_expr, context)
        rhs = -self.operators.divergence(flux) + source
        return OperatorEvaluation(rhs=rhs, flux=flux, context=context)


class DiffusionSourcePDEModel(ConservativeFluxPDEModel):
    """
    Generic conservative PDE model:

        du/dt = -Div(flux_coefficient * Grad(u)) + source

    For diffusion, pass flux_coefficient = -D. The class supports the existing
    finite-volume, finite-difference, and region-wise Chebyshev operators.
    """

    def __init__(self, mesh, operators, spec=None, variable_name="state"):
        self.mesh = mesh
        self.operators = operators
        if spec is None:
            spec = OperatorEquationSpec(
                state_name=variable_name,
                variable_name=variable_name,
                domain="through_cell",
                rhs=diffusion_source_expression(variable_name),
                evaluator="diffusion_source",
                flux=Parameter("flux_coefficient") * Grad(Variable(variable_name)),
                source=Parameter("source"),
            )
        self.spec = spec
        self.variable_name = spec.variable_name
        self.equation = spec.rhs

    def _merge_boundary_conditions(self, boundary_conditions):
        merged = dict(self.spec.boundary_conditions)
        if boundary_conditions:
            merged.update(boundary_conditions)
        return merged

    def _build_context(self, values, parameters, variables=None, boundary_conditions=None):
        variable_values = dict(variables or {})
        variable_values.setdefault(self.variable_name, values)
        return DiscretizationContext(
            mesh=self.mesh,
            operators=self.operators,
            variables=variable_values,
            parameters=dict(parameters or {}),
            boundary_conditions=self._merge_boundary_conditions(boundary_conditions),
        )

    def _flux(self, values, context):
        coefficient = _resolve_context_value(context.parameters["flux_coefficient"], context)
        return coefficient * self.operators.gradient(values)

    def _source(self, context):
        return _resolve_context_value(context.parameters["source"], context)

    def _with_finite_volume_boundary_flux(self, flux, context):
        if not isinstance(self.operators, FiniteVolumeOperators):
            return flux

        adjusted = flux.clone()
        for location, index in (("left", 0), ("right", -1)):
            if location not in context.boundary_conditions:
                continue
            bc = _as_boundary_condition(context.boundary_conditions[location])
            if bc.kind != "neumann":
                continue
            adjusted[index] = _resolve_context_value(bc.value, context)
        return adjusted

    def _evaluate_raw(self, values, context):
        flux = self._with_finite_volume_boundary_flux(self._flux(values, context), context)
        if isinstance(self.operators, FiniteVolumeOperators):
            rhs = -self.operators.divergence(flux) + self._source(context)
        else:
            rhs = discretize(self.equation, context)
        return OperatorEvaluation(rhs=rhs, flux=flux, context=context)

    def evaluate(
        self,
        values=None,
        time_values=None,
        parameters=None,
        variables=None,
        boundary_conditions=None,
        old_values=None,
        dt=None,
        return_details=False,
        **legacy_kwargs,
    ):
        if values is None:
            values = legacy_kwargs.pop(self.variable_name, None)
        if values is None and "ce" in legacy_kwargs:
            values = legacy_kwargs.pop("ce")
        if parameters is None:
            parameters = {}
        parameters = dict(parameters)
        if "flux_coefficient" in legacy_kwargs:
            parameters["flux_coefficient"] = legacy_kwargs.pop("flux_coefficient")
        if "source" in legacy_kwargs:
            parameters["source"] = legacy_kwargs.pop("source")
        if boundary_conditions is None:
            boundary_conditions = legacy_kwargs.pop("boundary_data", None)
        if time_values is None:
            time_values = values

        context = self._build_context(values, parameters, variables, boundary_conditions)
        evaluation = self._evaluate_raw(values, context)
        rhs = evaluation.rhs

        if old_values is not None and dt is not None:
            if isinstance(self.operators, ChebyshevRegionOperators):
                rhs = self.apply_chebyshev_boundary_constraints(
                    values=values,
                    time_values=time_values,
                    rhs=rhs,
                    flux=evaluation.flux,
                    old_values=old_values,
                    dt=dt,
                    boundary_conditions=context.boundary_conditions,
                )
            elif isinstance(self.operators, FiniteDifferenceOperators):
                rhs = self.apply_endpoint_boundary_constraints(
                    values=values,
                    time_values=time_values,
                    rhs=rhs,
                    flux=evaluation.flux,
                    old_values=old_values,
                    dt=dt,
                    boundary_conditions=context.boundary_conditions,
                    context=context,
                )

        if return_details:
            return OperatorEvaluation(rhs=rhs, flux=evaluation.flux, context=context)
        return rhs

    def apply_endpoint_boundary_constraints(
        self,
        values,
        time_values,
        rhs,
        flux,
        old_values,
        dt,
        boundary_conditions,
        context,
    ):
        constrained = rhs.clone()
        for location, index in (("left", 0), ("right", -1)):
            if location not in boundary_conditions:
                continue
            bc = _as_boundary_condition(boundary_conditions[location])
            target = _resolve_context_value(bc.value, context)
            if bc.kind == "dirichlet":
                residual = values[index] - target
            elif bc.kind == "residual":
                residual = target
            else:
                residual = flux[index] - target
            constrained[index] = (time_values[index] - old_values[index]) / dt - residual / dt
        return constrained

    def apply_chebyshev_boundary_constraints(
        self,
        values,
        time_values,
        rhs,
        flux,
        old_values,
        dt,
        boundary_conditions,
    ):
        if not isinstance(self.operators, ChebyshevRegionOperators):
            return rhs

        values_regions = self.operators.split_by_region(values)
        time_regions = self.operators.split_by_region(time_values)
        rhs_regions = {
            name: chunk.clone()
            for name, chunk in self.operators.split_by_region(rhs).items()
        }
        flux_regions = self.operators.split_by_region(flux)
        old_regions = self.operators.split_by_region(old_values)

        names = self.mesh.region_names

        first_name = names[0]
        left_bc = _as_boundary_condition(
            boundary_conditions.get("left", BoundaryCondition("neumann", 0.0))
        )
        left_value = _resolve_context_value(left_bc.value, DiscretizationContext(
            self.mesh,
            self.operators,
            {self.variable_name: values},
            {},
            boundary_conditions=boundary_conditions,
        ))
        if left_bc.kind == "dirichlet":
            left_residual = values_regions[first_name][0] - left_value
        elif left_bc.kind == "residual":
            left_residual = left_value
        else:
            left_residual = flux_regions[first_name][0] - left_value
        rhs_regions[first_name][0] = (
            (time_regions[first_name][0] - old_regions[first_name][0]) / dt
            - left_residual / dt
        )

        last_name = names[-1]
        right_bc = _as_boundary_condition(
            boundary_conditions.get("right", BoundaryCondition("neumann", 0.0))
        )
        right_value = _resolve_context_value(right_bc.value, DiscretizationContext(
            self.mesh,
            self.operators,
            {self.variable_name: values},
            {},
            boundary_conditions=boundary_conditions,
        ))
        if right_bc.kind == "dirichlet":
            right_residual = values_regions[last_name][-1] - right_value
        elif right_bc.kind == "residual":
            right_residual = right_value
        else:
            right_residual = flux_regions[last_name][-1] - right_value
        rhs_regions[last_name][-1] = (
            (time_regions[last_name][-1] - old_regions[last_name][-1]) / dt
            - right_residual / dt
        )

        for left_name, right_name in zip(names[:-1], names[1:]):
            rhs_regions[left_name][-1] = (
                (time_regions[left_name][-1] - old_regions[left_name][-1]) / dt
                - (values_regions[left_name][-1] - values_regions[right_name][0]) / dt
            )
            rhs_regions[right_name][0] = (
                (time_regions[right_name][0] - old_regions[right_name][0]) / dt
                - (flux_regions[left_name][-1] - flux_regions[right_name][0]) / dt
            )

        return self.operators.concatenate_by_region(rhs_regions)


class ElectrolytePDEModel(DiffusionSourcePDEModel):
    """
    Backward-compatible name for the through-cell flux-source PDE model.
    """

    def __init__(self, mesh, operators):
        spec = OperatorEquationSpec(
            state_name="electrolyte",
            variable_name="ce",
            domain="through_cell",
            rhs=diffusion_source_expression("ce"),
            evaluator="diffusion_source",
            flux=Parameter("flux_coefficient") * Grad(Variable("ce")),
            source=Parameter("source"),
        )
        super().__init__(mesh, operators, spec=spec)

    def apply_chebyshev_boundary_constraints(
        self,
        ce=None,
        dce=None,
        flux=None,
        ce_old=None,
        dt=None,
        boundary_conditions=None,
        **kwargs,
    ):
        values = kwargs.pop("values", ce)
        time_values = kwargs.pop("time_values", values)
        rhs = kwargs.pop("rhs", dce)
        old_values = kwargs.pop("old_values", ce_old)
        return super().apply_chebyshev_boundary_constraints(
            values=values,
            time_values=time_values,
            rhs=rhs,
            flux=flux,
            old_values=old_values,
            dt=dt,
            boundary_conditions=boundary_conditions or {},
        )


class SphericalParticlePDEModel:
    """
    Spherical diffusion on a Chebyshev particle-radius domain:

        dc/dt = D * (d2c/dr2 + 2/r * dc/dr)

    Boundary conditions are embedded as algebraic residuals during implicit
    Newton steps in the same DAE style used by the original implementation.
    """

    def __init__(self, radius_nodes, first_matrix, second_matrix, spec=None, variable_name="c"):
        self.radius_nodes = radius_nodes
        self.first_matrix = first_matrix
        self.second_matrix = second_matrix
        if spec is None:
            spec = OperatorEquationSpec(
                state_name=variable_name,
                variable_name=variable_name,
                domain="particle_radius",
                rhs=spherical_diffusion_expression(variable_name),
                evaluator="spherical_particle",
            )
        self.spec = spec
        self.variable_name = spec.variable_name

    def _merge_boundary_conditions(self, boundary_conditions):
        merged = dict(self.spec.boundary_conditions)
        if boundary_conditions:
            merged.update(boundary_conditions)
        return merged

    def _physical_operators(self, radius):
        first = self.first_matrix / radius
        second = self.second_matrix / (radius ** 2)
        r = self.radius_nodes * radius
        return r, first, second

    def gradient(self, values, radius):
        _r, first, _second = self._physical_operators(radius)
        return torch.mv(first, values)

    def evaluate(
        self,
        values,
        time_values=None,
        parameters=None,
        variables=None,
        boundary_conditions=None,
        old_values=None,
        dt=None,
        return_details=False,
    ):
        parameters = dict(parameters or {})
        diffusivity = parameters["diffusivity"]
        radius = parameters.get("particle_radius", parameters.get("radius"))
        if radius is None:
            raise ValueError("Spherical particle PDE requires 'particle_radius' or 'radius'.")

        r, first, second = self._physical_operators(radius)
        r_safe = r.clone()
        r_safe[0] = 1.0

        second_term = torch.mv(second, values)
        first_term = (2.0 / r_safe) * torch.mv(first, values)
        rhs = diffusivity * (second_term + first_term)
        rhs[0] = 3.0 * diffusivity * second_term[0]

        boundary_conditions = self._merge_boundary_conditions(boundary_conditions)
        if old_values is not None and dt is not None:
            rhs = self.apply_boundary_constraints(
                values=values,
                rhs=rhs,
                old_values=old_values,
                dt=dt,
                diffusivity=diffusivity,
                radius=radius,
                boundary_conditions=boundary_conditions,
            )

        if return_details:
            return OperatorEvaluation(
                rhs=rhs,
                flux=-diffusivity * self.gradient(values, radius),
                context={
                    "values": values,
                    "parameters": parameters,
                    "boundary_conditions": boundary_conditions,
                },
            )
        return rhs

    def _boundary_residual(self, location, index, bc, values, diffusivity, radius):
        ctx = {
            "location": location,
            "index": index,
            "values": values,
            "diffusivity": diffusivity,
            "radius": radius,
            "gradient": self.gradient(values, radius),
        }
        target = bc.value(ctx) if callable(bc.value) else bc.value
        if bc.kind == "dirichlet":
            return values[index] - target
        if bc.kind == "residual":
            return target
        return ctx["gradient"][index] - target

    def apply_boundary_constraints(
        self,
        values,
        rhs,
        old_values,
        dt,
        diffusivity,
        radius,
        boundary_conditions,
    ):
        constrained = rhs.clone()
        for location, index in (("center", 0), ("left", 0), ("surface", -1), ("right", -1)):
            if location not in boundary_conditions:
                continue
            bc = _as_boundary_condition(boundary_conditions[location])
            residual = self._boundary_residual(location, index, bc, values, diffusivity, radius)
            constrained[index] = (values[index] - old_values[index]) / dt - residual / dt
        return constrained
