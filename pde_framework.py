from dataclasses import dataclass

import torch


class Expression:
    def __add__(self, other):
        return Add(self, ensure_expression(other))

    def __mul__(self, other):
        return Multiply(self, ensure_expression(other))

    def __rmul__(self, other):
        return Multiply(ensure_expression(other), self)


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
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    left: Expression
    right: Expression


def ensure_expression(value):
    if isinstance(value, Expression):
        return value
    return Constant(float(value))


@dataclass(frozen=True)
class StateFieldSpec:
    name: str
    size: int
    initial: object = 0.0
    scale: object = 1.0


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

    def register(self, name, size, initial=0.0, scale=1.0):
        if name in self.slices:
            raise ValueError(f"State field {name!r} is already registered.")
        size = int(size)
        if size <= 0:
            raise ValueError(f"State field {name!r} must have positive size.")
        start = self.total_size
        stop = start + size
        self.fields.append(StateFieldSpec(name, size, initial, scale))
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

    if isinstance(expr, Multiply):
        return discretize(expr.left, context) * discretize(expr.right, context)

    if isinstance(expr, Grad):
        return context.operators.gradient(discretize(expr.child, context))

    if isinstance(expr, Div):
        return context.operators.divergence(discretize(expr.child, context))

    raise TypeError(f"Unsupported expression type: {type(expr)!r}")


class ElectrolytePDEModel:
    """
    Symbolic electrolyte diffusion equation plus generic boundary-condition embedding.
    This replaces only the discretization layer, not the Newton solver or state layout.
    """

    def __init__(self, mesh, operators):
        self.mesh = mesh
        self.operators = operators
        self.ce = Variable("ce")
        self.equation = Add(
            Multiply(Constant(-1.0), Div(Multiply(Parameter("flux_coefficient"), Grad(self.ce)))),
            Parameter("source")
        )

    def evaluate(self, ce, flux_coefficient, source, boundary_data=None):
        context = DiscretizationContext(
            mesh=self.mesh,
            operators=self.operators,
            variables={"ce": ce},
            parameters={"flux_coefficient": flux_coefficient, "source": source},
            boundary_conditions=boundary_data or {}
        )
        return discretize(self.equation, context)

    def apply_chebyshev_boundary_constraints(
        self,
        ce,
        dce,
        flux,
        ce_old,
        dt,
        boundary_conditions,
    ):
        if not isinstance(self.operators, ChebyshevRegionOperators):
            return dce

        ce_regions = self.operators.split_by_region(ce)
        dce_regions = self.operators.split_by_region(dce)
        flux_regions = self.operators.split_by_region(flux)
        ce_old_regions = self.operators.split_by_region(ce_old)

        names = self.mesh.region_names

        left_bc = boundary_conditions.get("left", {"type": "neumann", "value": 0.0})
        right_bc = boundary_conditions.get("right", {"type": "neumann", "value": 0.0})

        first_name = names[0]
        if left_bc["type"] == "neumann":
            dce_regions[first_name][0] = (
                (ce_regions[first_name][0] - ce_old_regions[first_name][0]) / dt
                - (flux_regions[first_name][0] - left_bc["value"]) / dt
            )
        elif left_bc["type"] == "dirichlet":
            dce_regions[first_name][0] = (
                (ce_regions[first_name][0] - ce_old_regions[first_name][0]) / dt
                - (ce_regions[first_name][0] - left_bc["value"]) / dt
            )

        last_name = names[-1]
        if right_bc["type"] == "neumann":
            dce_regions[last_name][-1] = (
                (ce_regions[last_name][-1] - ce_old_regions[last_name][-1]) / dt
                - (flux_regions[last_name][-1] - right_bc["value"]) / dt
            )
        elif right_bc["type"] == "dirichlet":
            dce_regions[last_name][-1] = (
                (ce_regions[last_name][-1] - ce_old_regions[last_name][-1]) / dt
                - (ce_regions[last_name][-1] - right_bc["value"]) / dt
            )

        for left_name, right_name in zip(names[:-1], names[1:]):
            dce_regions[left_name][-1] = (
                (ce_regions[left_name][-1] - ce_old_regions[left_name][-1]) / dt
                - (ce_regions[left_name][-1] - ce_regions[right_name][0]) / dt
            )
            dce_regions[right_name][0] = (
                (ce_regions[right_name][0] - ce_old_regions[right_name][0]) / dt
                - (flux_regions[left_name][-1] - flux_regions[right_name][0]) / dt
            )

        return self.operators.concatenate_by_region(dce_regions)
