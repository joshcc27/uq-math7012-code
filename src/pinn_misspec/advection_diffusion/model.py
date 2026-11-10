"""Advection--diffusion with an omitted sink, and one inverse-PINN run.

    generating:  u_t + a u_x = kappa u_xx - mu u   on the periodic unit interval
    imposed:     u_t + a u_x = kappa u_xx

Severity is ``s = mu T``. With ``fit_sink`` the sink is trained as a third
coefficient, which is the relaxed check. The trial state is periodic and imposes the
initial profile exactly.

Run ids hash the whole configuration, so it keeps the archive's key names and a few
entries that nothing reads any more; ``collocation_pool`` appears only when set.
"""

import time
from functools import lru_cache, partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pinn_misspec.training import (
    apply_overrides,
    build_network,
    cosine_adam,
    finite,
    integral,
    resume,
)

PROTOCOL_VERSION = "v2-sink-severity"

VELOCITY_TRUE = 1.0
DIFFUSIVITY_TRUE = 0.01
T_END = 0.60
INITIAL_CENTRE = 0.30
INITIAL_WIDTH = 0.08
FOURIER_MODES = 48

VELOCITY_INIT = 0.70
DIFFUSIVITY_INIT = 0.02
VELOCITY_MIN, VELOCITY_MAX = 0.50, 1.50
DIFFUSIVITY_MIN, DIFFUSIVITY_MAX = 0.0025, 0.05
# The relaxed check trains the sink directly, not as a logarithm, since the
# control's true sink is zero; the range includes a small source so zero is interior.
SINK_MIN, SINK_MAX = -1.0, 4.0

SIGMA = 0.02
N_OBS = 600  # 504 noisy interior points and the 96 noise-free initial values
N_INITIAL = 96
N_COL = 600
NETWORK_WIDTH = 30
NETWORK_DEPTH = 3
NETWORK_LEARNING_RATE = 1e-2
THETA_LEARNING_RATE = 3e-3
PHYSICS_STEPS = 5000

# Not read any more, but part of every archived configuration, so they stay
# for the run ids.
INERT = {
    "method": "joint",
    "verdict_window": 0.10,
    "verdict_tolerance_velocity": 0.02,
    "verdict_tolerance_diffusivity": 0.10,
    "verdict_progress": 0.80,
}

# ``collocation_pool`` N trains on the first ``n_col`` of one N-point draw, so
# designs of several densities share their points (the nested test).
OPTIONAL = {"collocation_pool": None}


def default_config(**overrides):
    """Return a validated configuration; an unknown key raises ``KeyError``."""
    config = {
        "protocol": PROTOCOL_VERSION,
        "velocity_true": VELOCITY_TRUE,
        "diffusivity_true": DIFFUSIVITY_TRUE,
        "t_end": T_END,
        "initial_centre": INITIAL_CENTRE,
        "initial_width": INITIAL_WIDTH,
        "fourier_modes": FOURIER_MODES,
        "severity": 0.0,
        "sigma": SIGMA,
        "n_obs": N_OBS,
        "n_initial": N_INITIAL,
        "n_col": N_COL,
        "noise_seed": 0,
        "velocity_init": VELOCITY_INIT,
        "diffusivity_init": DIFFUSIVITY_INIT,
        "velocity_min": VELOCITY_MIN,
        "velocity_max": VELOCITY_MAX,
        "diffusivity_min": DIFFUSIVITY_MIN,
        "diffusivity_max": DIFFUSIVITY_MAX,
        "fit_sink": False,
        "sink_init": 0.0,
        "sink_min": SINK_MIN,
        "sink_max": SINK_MAX,
        "physics_weight": 0.01,
        "network_seed": 0,
        "network_width": NETWORK_WIDTH,
        "network_depth": NETWORK_DEPTH,
        "network_learning_rate": NETWORK_LEARNING_RATE,
        "theta_learning_rate": THETA_LEARNING_RATE,
        "physics_steps": PHYSICS_STEPS,
        **INERT,
    }
    config = apply_overrides(config, overrides, OPTIONAL)
    validate_config(config)
    return config


def validate_config(config):
    if config["method"] != "joint":
        raise ValueError(f"method must be 'joint', got {config['method']!r}")
    if "collocation_pool" in config and integral(
        "collocation_pool", config["collocation_pool"]
    ) < int(config["n_col"]):
        raise ValueError("collocation_pool must be at least n_col")
    positive = (
        "velocity_true",
        "diffusivity_true",
        "t_end",
        "initial_width",
        "velocity_init",
        "diffusivity_init",
        "network_learning_rate",
        "theta_learning_rate",
    )
    for name in positive:
        if not finite(name, config[name]) > 0.0:
            raise ValueError(f"{name} must be positive, got {config[name]}")
    for name in ("severity", "sigma", "physics_weight"):
        if not finite(name, config[name]) >= 0.0:
            raise ValueError(f"{name} must be non-negative, got {config[name]}")
    counts = (
        "n_obs",
        "n_initial",
        "n_col",
        "fourier_modes",
        "network_width",
        "network_depth",
        "physics_steps",
        "noise_seed",
        "network_seed",
    )
    for name in counts:
        value = integral(name, config[name])
        floor = 0 if name.endswith("seed") else 1
        if value < floor:
            raise ValueError(f"{name} must be at least {floor}, got {config[name]}")
    if int(config["n_initial"]) >= int(config["n_obs"]):
        raise ValueError("n_initial must be smaller than n_obs")
    if (
        not 0.0
        < finite("velocity_min", config["velocity_min"])
        < finite("velocity_max", config["velocity_max"])
    ):
        raise ValueError("need 0 < velocity_min < velocity_max")
    if (
        not 0.0
        < finite("diffusivity_min", config["diffusivity_min"])
        < finite("diffusivity_max", config["diffusivity_max"])
    ):
        raise ValueError("need 0 < diffusivity_min < diffusivity_max")
    if not isinstance(config["fit_sink"], bool | np.bool_):
        raise ValueError(f"fit_sink must be True or False, got {config['fit_sink']!r}")
    finite("sink_init", config["sink_init"])
    if not finite("sink_min", config["sink_min"]) < finite("sink_max", config["sink_max"]):
        raise ValueError("need sink_min < sink_max")
    return config


def sink_rate(config):
    return float(config["severity"]) / float(config["t_end"])


def raw_physics_weight(scaled, config):
    """Convert the reported weight ``Lambda = lambda / T^2`` to the loss weight ``lambda``."""
    scaled, t_end = float(scaled), float(config["t_end"])
    if not np.isfinite(scaled) or scaled < 0.0:
        raise ValueError("scaled weight must be finite and non-negative")
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive")
    return scaled * t_end**2


def scaled_physics_weight(config):
    return float(config["physics_weight"]) / float(config["t_end"]) ** 2


def initial_profile_numpy(x, centre=INITIAL_CENTRE, width=INITIAL_WIDTH):
    """Return the wrapped Gaussian initial profile at ``x``."""
    x = np.asarray(x)
    shifts = np.arange(-3, 4, dtype=float)
    return np.sum(np.exp(-0.5 * ((x[..., None] - centre + shifts) / width) ** 2), axis=-1)


def initial_profile_jax(x, centre, width):
    shifts = jnp.arange(-3.0, 4.0)
    return jnp.sum(jnp.exp(-0.5 * ((x - centre + shifts) / width) ** 2))


@lru_cache(maxsize=64)
def _fourier_data(centre, width, modes_count):
    """Return the modes and Fourier coefficients of the initial profile."""
    grid = np.linspace(0.0, 1.0, 4096, endpoint=False)
    modes = np.arange(-int(modes_count), int(modes_count) + 1)
    initial = initial_profile_numpy(grid, float(centre), float(width))
    coefficients = np.mean(initial[:, None] * np.exp(-2j * np.pi * grid[:, None] * modes), axis=0)
    return modes, coefficients


def exact_solution(points, velocity, diffusivity, sink, config):
    """Return the truncated Fourier solution at ``[x, t]`` points.

    A uniform sink multiplies the conservative solution by ``exp(-mu t)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    velocity, diffusivity, sink = float(velocity), float(diffusivity), float(sink)
    if not np.isfinite([velocity, diffusivity, sink]).all() or diffusivity <= 0.0:
        raise ValueError("velocity and sink must be finite and diffusivity positive")
    modes, coefficients = _fourier_data(
        float(config["initial_centre"]),
        float(config["initial_width"]),
        int(config["fourier_modes"]),
    )
    x, t = points[:, 0], points[:, 1]
    symbol = -2j * np.pi * modes * velocity - diffusivity * (2.0 * np.pi * modes) ** 2 - sink
    values = (np.exp(2j * np.pi * x[:, None] * modes) * np.exp(t[:, None] * symbol)) @ coefficients
    return np.real(values)


def exact_derivatives(points, velocity, diffusivity, sink, config):
    """Return the Fourier solution and its derivatives at ``[x, t]`` points.

    Each mode ``c_k exp(2 pi i k x + lambda_k t)`` is differentiated term by
    term, so the derivatives are exact for the truncated series.

    Returns:
        Dictionary of ``u``, ``u_t``, ``u_x`` and ``u_xx`` arrays.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    modes, coefficients = _fourier_data(
        float(config["initial_centre"]),
        float(config["initial_width"]),
        int(config["fourier_modes"]),
    )
    wavenumber = 2.0 * np.pi * modes
    symbol = -1j * wavenumber * float(velocity) - float(diffusivity) * wavenumber**2 - float(sink)
    x, t = points[:, 0], points[:, 1]
    terms = np.exp(1j * x[:, None] * wavenumber) * np.exp(t[:, None] * symbol)
    return {
        "u": np.real(terms @ coefficients),
        "u_t": np.real(terms @ (symbol * coefficients)),
        "u_x": np.real(terms @ (1j * wavenumber * coefficients)),
        "u_xx": np.real(terms @ (-(wavenumber**2) * coefficients)),
    }


def generating_solution(points, config):
    return exact_solution(
        points, config["velocity_true"], config["diffusivity_true"], sink_rate(config), config
    )


def imposed_solution(points, velocity, diffusivity, config):
    return exact_solution(points, velocity, diffusivity, 0.0, config)


def make_observations(config):
    """Return the observation points, the observations and the collocation points.

    The noise seed draws the interior observation locations, then their noise,
    then the collocation points, so the collocation density never changes the
    observations. The initial profile is observed without noise.
    """
    rng = np.random.default_rng(int(config["noise_seed"]))
    interior_count = int(config["n_obs"]) - int(config["n_initial"])
    interior = np.column_stack(
        [
            rng.uniform(0.0, 1.0, interior_count),
            rng.uniform(0.0, config["t_end"], interior_count),
        ]
    )
    initial = np.column_stack(
        [
            np.linspace(0.0, 1.0, int(config["n_initial"]), endpoint=False),
            np.zeros(int(config["n_initial"])),
        ]
    )
    points = np.concatenate([interior, initial])
    observations = generating_solution(points, config)
    observations[:interior_count] += rng.normal(0.0, config["sigma"], interior_count)
    collocation = np.column_stack(
        [
            rng.uniform(0.0, 1.0, int(config["n_col"])),
            rng.uniform(0.0, config["t_end"], int(config["n_col"])),
        ]
    )
    return tuple(map(jnp.asarray, (points, observations, collocation)))


def collocation_points(config):
    """Return the training collocation points of a configuration.

    With ``collocation_pool`` N set, they are the first ``n_col`` points of one
    N-point draw, so smaller designs are subsets of larger ones.
    """
    if "collocation_pool" in config:
        pool = {**config, "n_col": int(config["collocation_pool"])}
        return make_observations(pool)[2][: int(config["n_col"])]
    return make_observations(config)[2]


def trial_spec(config):
    return float(config["initial_centre"]), float(config["initial_width"]), float(config["t_end"])


def state_at(net, point, trial):
    """Return the periodic neural state, with its exact initial profile, at one point."""
    centre, width, t_end = trial
    x, t = point
    features = jnp.array([jnp.sin(2.0 * jnp.pi * x), jnp.cos(2.0 * jnp.pi * x), t / t_end])
    return initial_profile_jax(x, centre, width) + (t / t_end) * net(features)[0]


state_gradient = jax.grad(state_at, argnums=1)


def spatial_derivative_at(net, point, trial):
    return state_gradient(net, point, trial)[0]


spatial_derivative_gradient = jax.grad(spatial_derivative_at, argnums=1)


def coefficients(theta):
    """Decode ``[a, log kappa, ...]`` into ``(a, kappa)``."""
    return theta[0], jnp.exp(theta[1])


def fitted_sink(theta):
    """Return the trained sink rate, or ``None`` when the sink is not fitted.

    The sink is a third entry of ``theta`` only in the relaxed check, so the
    shape of ``theta`` decides it, which is static under ``jit``.
    """
    return theta[2] if theta.shape[0] == 3 else None


def residual_at(net, theta, point, trial):
    """Return ``u_t + a u_x - kappa u_xx``, plus ``mu u`` if the sink is fitted, at one point."""
    velocity, diffusivity = coefficients(theta)
    gradient = state_gradient(net, point, trial)
    value = (
        gradient[1]
        + velocity * gradient[0]
        - diffusivity * spatial_derivative_gradient(net, point, trial)[0]
    )
    # Not added as zero otherwise, so the conservative residual is unchanged.
    sink = fitted_sink(theta)
    if sink is not None:
        value = value + sink * state_at(net, point, trial)
    return value


predict = jax.vmap(state_at, in_axes=(None, 0, None))
residual = jax.vmap(residual_at, in_axes=(None, None, 0, None))


def loss_with_components(
    current_params,
    weight,
    *,
    static_net,
    observation_points,
    observations,
    collocation_points,
    trial,
):
    """Return ``data + weight * residual`` and the pair ``(data, residual)``."""
    trainable_net, theta = current_params
    net = eqx.combine(trainable_net, static_net)
    data_loss = jnp.mean((predict(net, observation_points, trial) - observations) ** 2)
    physics_loss = jnp.mean(residual(net, theta, collocation_points, trial) ** 2)
    return data_loss + weight * physics_loss, (data_loss, physics_loss)


def run_adam(params, optimizer, loss_fn, weight, steps, optimizer_state=None):
    """Run ``steps`` Adam updates and return the trace with the final iterate appended.

    Each trace row is ``[a, kappa, data, residual]``, then the sink if it is
    fitted, for the iterate entering that update.
    """
    if optimizer_state is None:
        optimizer_state = optimizer.init(params)

    def measure(current_params):
        _, losses = loss_fn(current_params, weight)
        velocity, diffusivity = coefficients(current_params[1])
        sink = fitted_sink(current_params[1])
        extra = [] if sink is None else [sink]
        return jnp.array([velocity, diffusivity, *losses, *extra])

    def step(carry, _):
        current_params, current_state = carry
        row = measure(current_params)
        (_, _), gradients = jax.value_and_grad(loss_fn, has_aux=True)(current_params, weight)
        updates, current_state = optimizer.update(gradients, current_state, current_params)
        return (optax.apply_updates(current_params, updates), current_state), row

    (params, optimizer_state), trace = jax.lax.scan(
        step, (params, optimizer_state), None, length=steps
    )
    trace = jnp.concatenate([trace, measure(params)[None, :]], axis=0)
    return params, optimizer_state, trace


def training_setup(config):
    """Return the starting parameters, the loss and the data of one run."""
    trial = trial_spec(config)
    observation_points, observations, _ = make_observations(config)
    collocation = collocation_points(config)
    # The network sees sin 2 pi x, cos 2 pi x and t/T.
    trainable_net, static_net = eqx.partition(
        build_network(config, in_size=3), eqx.is_inexact_array
    )
    theta = jnp.array([config["velocity_init"], jnp.log(config["diffusivity_init"])])
    if config["fit_sink"]:
        theta = jnp.append(theta, config["sink_init"])
    loss = partial(
        loss_with_components,
        static_net=static_net,
        observation_points=observation_points,
        observations=observations,
        collocation_points=collocation,
        trial=trial,
    )
    return {
        "params": (trainable_net, theta),
        "static_net": static_net,
        "loss": loss,
        "labels": (jax.tree.map(lambda _: "net", trainable_net), "theta"),
        "observation_points": observation_points,
        "observations": observations,
        "collocation_points": collocation,
    }


def run_experiment(config, checkpoint=None):
    """Train one PINN and return its network, estimates, losses and trace.

    Args:
        config: A configuration from ``default_config``.
        checkpoint: ``(params, optimizer_state, steps_done)`` from an earlier
            part of the same run, or ``None``.

    Returns:
        Dictionary with the network, parameters, optimizer state, the
        estimates, the final losses, the trace and the data.
    """
    validate_config(config)
    started = time.perf_counter()
    setup = training_setup(config)
    weight = config["physics_weight"]
    steps = int(config["physics_steps"])
    params, optimizer_state, steps_done = resume(checkpoint, steps, setup["params"])
    optimizer = optax.multi_transform(cosine_adam(config, steps), setup["labels"])
    params, optimizer_state, trace = run_adam(
        params, optimizer, setup["loss"], weight, steps - steps_done, optimizer_state
    )
    if steps_done:
        trace = trace[1:]
    velocity, diffusivity = map(float, coefficients(params[1]))
    sink = fitted_sink(params[1])
    _, losses = setup["loss"](params, weight)
    data_loss, physics_loss = map(float, losses)
    jax.block_until_ready(params)
    return {
        "config": config,
        "net": eqx.combine(params[0], setup["static_net"]),
        "params": params,
        "optimizer_state": optimizer_state,
        "velocity": velocity,
        "diffusivity": diffusivity,
        "sink": None if sink is None else float(sink),
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "trace": np.asarray(trace),
        "observation_points": setup["observation_points"],
        "observations": setup["observations"],
        "collocation_points": setup["collocation_points"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarise_run(result):
    """Return the configuration plus the run's estimates, errors and losses, as a row."""
    config = result["config"]
    row = dict(config)
    row.update(
        {
            "sink_rate": sink_rate(config),
            "velocity_final": result["velocity"],
            "diffusivity_final": result["diffusivity"],
            "sink_final": result["sink"],
            "velocity_relative_error": abs(result["velocity"] - config["velocity_true"])
            / abs(config["velocity_true"]),
            "diffusivity_relative_error": abs(result["diffusivity"] - config["diffusivity_true"])
            / abs(config["diffusivity_true"]),
            "data_loss": result["data_loss"],
            "physics_loss": result["physics_loss"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    return row


def evaluation_grid(config, n_x=128, n_t=61):
    """Return a dense periodic space--time grid as points, and its two axes."""
    x = np.linspace(0.0, 1.0, n_x, endpoint=False)
    t = np.linspace(0.0, config["t_end"], n_t)
    x_mesh, t_mesh = np.meshgrid(x, t)
    return np.column_stack([x_mesh.ravel(), t_mesh.ravel()]), x, t


def state_errors(result, n_x=128, n_t=61):
    """Return the dense points, the neural state and the generating state."""
    config = result["config"]
    points, _, _ = evaluation_grid(config, n_x, n_t)
    neural = np.asarray(predict(result["net"], jnp.asarray(points), trial_spec(config)))
    return points, neural, generating_solution(points, config)
