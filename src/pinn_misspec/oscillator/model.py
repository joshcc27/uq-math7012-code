"""The damped oscillator and one inverse-PINN run.

    u'' + 2 gamma u' + omega^2 u = 0,   u(0) = u0,   u'(0) = v0
    y_i = u(t_i) + epsilon_i,           epsilon_i ~ N(0, sigma^2)

Severity ``s`` fixes the damping at ``g_true (1 + s)`` and only the frequency is
fitted; ``severity=None`` fits both, which is the relaxed check. The trial state
``u0 + v0 t + phi(t) N(t/T)`` imposes the initial conditions exactly.

Run ids hash the whole configuration, so it keeps the archive's key names (``w0``,
``g``) and a few entries that nothing reads any more.
"""

import time
from functools import partial

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

# Recorded with every run. Bump it when a change makes new runs incomparable
# with the archived ones.
PROTOCOL_VERSION = "v4-severity"

OMEGA_TRUE = 2.0 * np.pi  # rad/s, a period of one time unit
GAMMA_TRUE = 0.30  # 1/s
U0, V0 = 1.0, 0.0
T_END = 3.0

# Admissible frequencies, half to twice the truth. The PINN may leave this
# range; an estimate outside it is flagged, not clipped.
OMEGA_MIN, OMEGA_MAX = np.pi, 4.0 * np.pi
# Admissible damping for the relaxed check.
GAMMA_MIN, GAMMA_MAX = 0.03, 3.0

OMEGA_INIT, GAMMA_INIT = 5.0, 1.0
SIGMA = 0.05
N_OBS = 201
N_COL = 201
PHYSICS_STEPS = 9000
NETWORK_WIDTH = 20
NETWORK_DEPTH = 3
# Width tau of the initial-condition window, as a fraction of T.
TRIAL_WINDOW_FRACTION = 0.1
NETWORK_LEARNING_RATE = 1e-2
THETA_LEARNING_RATE = 5e-4

# Not read any more, but part of every archived configuration, so they stay
# for the run ids.
INERT = {
    "method": "joint",
    "pretrain_steps": 800,
    "direct_steps": 4000,
    "direct_learning_rate": 2e-2,
    "verdict_window": 0.2,
    "verdict_tolerance": 0.02,
    "verdict_progress": 0.9,
}


# With z = (gamma^2 - omega^2) t^2,
#
#     u(t) = e^{-gamma t} [ u0 C(z) + (v0 + gamma u0) t S(z) ],
#     C(z) = cosh(sqrt z) = sum z^k / (2k)!,
#     S(z) = sinh(sqrt z) / sqrt z = sum z^k / (2k+1)!.
#
# C and S are entire, so one expression covers all three regimes without
# dividing by sqrt(omega^2 - gamma^2). Near z = 0 they are evaluated by their
# series, because sqrt has an infinite derivative there. Overdamped cases occur
# because the imposed-solution gap uses the PINN's own frequency.

SERIES_THRESHOLD = 1e-2  # |z| below which the series is used
SERIES_TERMS = 5  # enough for double precision at that threshold


def _cosh_sqrt_series(z):
    total = 1.0
    for k in range(SERIES_TERMS, 0, -1):
        total = 1.0 + z * total / ((2 * k - 1) * (2 * k))
    return total


def _sinhc_sqrt_series(z):
    total = 1.0
    for k in range(SERIES_TERMS, 0, -1):
        total = 1.0 + z * total / ((2 * k) * (2 * k + 1))
    return total


# jnp.where evaluates both arms, so each arm gets an argument that is harmless
# for it. Otherwise the unused arm puts a NaN or an infinity into the gradient.


def _split_branches(z):
    """Return the series mask and safe arguments for the series and closed forms."""
    small = jnp.abs(z) < SERIES_THRESHOLD
    return small, jnp.where(small, z, 0.0), jnp.where(small, 1.0, z)


def _branch_roots(z_closed):
    """Return the sign mask and safe hyperbolic and circular square roots."""
    positive = z_closed > 0.0
    return (
        positive,
        jnp.sqrt(jnp.where(positive, z_closed, 1.0)),
        jnp.sqrt(jnp.where(positive, 1.0, -z_closed)),
    )


def cosh_sqrt(z):
    """``cosh(sqrt z)``, continued to negative ``z`` as ``cos(sqrt(-z))``."""
    small, z_series, z_closed = _split_branches(z)
    positive, hyperbolic, circular = _branch_roots(z_closed)
    closed = jnp.where(positive, jnp.cosh(hyperbolic), jnp.cos(circular))
    return jnp.where(small, _cosh_sqrt_series(z_series), closed)


def sinhc_sqrt(z):
    """``sinh(sqrt z) / sqrt z``, continued to negative ``z``."""
    small, z_series, z_closed = _split_branches(z)
    positive, hyperbolic, circular = _branch_roots(z_closed)
    closed = jnp.where(
        positive,
        jnp.sinh(hyperbolic) / hyperbolic,
        jnp.sin(circular) / circular,
    )
    return jnp.where(small, _sinhc_sqrt_series(z_series), closed)


def exact_solution(t, omega, gamma, u0=U0, v0=V0):
    """Return the displacement at ``t`` for any positive frequency and damping.

    Where ``z`` is large and positive, ``cosh`` of a large argument times a
    vanishing exponential would overflow, so there the solution is written with
    the two characteristic roots instead. The split is for floating-point range
    only; both branches are the same function.
    """
    discriminant = gamma**2 - omega**2
    z = discriminant * t**2
    overdamped = z > SERIES_THRESHOLD

    bounded = jnp.exp(-gamma * t) * (
        u0 * cosh_sqrt(jnp.where(overdamped, 0.0, z))
        + (v0 + gamma * u0) * t * sinhc_sqrt(jnp.where(overdamped, 0.0, z))
    )

    root = jnp.sqrt(jnp.where(overdamped, discriminant, 1.0))
    slow = jnp.exp(-(gamma - root) * t)
    fast = jnp.exp(-(gamma + root) * t)
    strongly_overdamped = 0.5 * (u0 * (slow + fast) + (v0 + gamma * u0) * (slow - fast) / root)
    return jnp.where(overdamped, strongly_overdamped, bounded)


def default_config(**overrides):
    """Return a validated configuration; an unknown key raises ``KeyError``."""
    config = {
        "protocol": PROTOCOL_VERSION,
        "w0_true": OMEGA_TRUE,
        "g_true": GAMMA_TRUE,
        "u0": U0,
        "v0": V0,
        "t_end": T_END,
        "trial_window_fraction": TRIAL_WINDOW_FRACTION,
        "sigma": SIGMA,
        "n_obs": N_OBS,
        "n_col": N_COL,
        "noise_seed": 0,
        "w0_init": OMEGA_INIT,
        "g_init": GAMMA_INIT,
        "severity": None,
        "physics_weight": 1e-3,
        "w0_min": OMEGA_MIN,
        "w0_max": OMEGA_MAX,
        "g_min": GAMMA_MIN,
        "g_max": GAMMA_MAX,
        "network_seed": 0,
        "network_width": NETWORK_WIDTH,
        "network_depth": NETWORK_DEPTH,
        "network_learning_rate": NETWORK_LEARNING_RATE,
        "theta_learning_rate": THETA_LEARNING_RATE,
        "physics_steps": PHYSICS_STEPS,
        **INERT,
    }
    config = apply_overrides(config, overrides)
    validate_config(config)
    return config


def validate_config(config):
    if config["method"] != "joint":
        raise ValueError(f"method must be 'joint', got {config['method']!r}")
    positive = (
        "w0_true",
        "g_true",
        "w0_init",
        "g_init",
        "t_end",
        "trial_window_fraction",
        "network_learning_rate",
        "theta_learning_rate",
    )
    for name in positive:
        if not finite(name, config[name]) > 0.0:
            raise ValueError(f"{name} must be positive, got {config[name]}")
    for name in ("sigma", "physics_weight"):
        if not finite(name, config[name]) >= 0.0:
            raise ValueError(f"{name} must be non-negative, got {config[name]}")
    for name in ("u0", "v0"):
        finite(name, config[name])
    counts = (
        "n_obs",
        "n_col",
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
    # s > -1 keeps the imposed damping positive.
    if config["severity"] is not None and not finite("severity", config["severity"]) > -1.0:
        raise ValueError(f"severity must exceed -1, got {config['severity']}")
    if not 0.0 < finite("w0_min", config["w0_min"]) < finite("w0_max", config["w0_max"]):
        raise ValueError(f"need 0 < w0_min < w0_max, got {config['w0_min']}, {config['w0_max']}")
    if not 0.0 < finite("g_min", config["g_min"]) < finite("g_max", config["g_max"]):
        raise ValueError(f"need 0 < g_min < g_max, got {config['g_min']}, {config['g_max']}")
    return config


def imposed_damping(config):
    """Return ``gamma_s = gamma_true (1 + s)``, or ``None`` when the damping is fitted."""
    if config["severity"] is None:
        return None
    return float(config["g_true"]) * (1.0 + float(config["severity"]))


def raw_physics_weight(scaled, config):
    """Convert the reported weight ``Lambda = lambda omega_true^4`` to the loss weight ``lambda``."""
    return float(scaled) / float(config["w0_true"]) ** 4


def scaled_physics_weight(config):
    return float(config["physics_weight"]) * float(config["w0_true"]) ** 4


def trial_window(t, tau):
    """Return ``phi(t) = t^2 / (t^2 + tau^2)``: a double zero at 0, then about 1.

    The plain ``(t/T)^2`` also has the double zero but never saturates, which
    makes the network cover a range of hundreds; it fitted the data far worse.
    """
    return t**2 / (t**2 + tau**2)


def state_at(net, t, trial):
    """Return the trial state at one time; ``trial`` is ``(u0, v0, t_end, tau)``."""
    u0, v0, t_end, tau = trial
    return u0 + v0 * t + trial_window(t, tau) * net(jnp.array([t / t_end]))[0]


du_dt = jax.grad(state_at, argnums=1)
d2u_dt2 = jax.grad(du_dt, argnums=1)


def coefficients(log_theta, gamma_imposed):
    """Return ``(omega, gamma)``, inserting the fixed damping if there is one."""
    if gamma_imposed is None:
        return tuple(jnp.exp(log_theta))
    return jnp.exp(log_theta[0]), gamma_imposed


def residual_at(net, log_theta, t, gamma_imposed, trial):
    """Return ``u'' + 2 gamma u' + omega^2 u`` of the trial state at one time."""
    omega, gamma = coefficients(log_theta, gamma_imposed)
    return (
        d2u_dt2(net, t, trial)
        + 2.0 * gamma * du_dt(net, t, trial)
        + omega**2 * state_at(net, t, trial)
    )


predict = jax.vmap(state_at, in_axes=(None, 0, None))
residual = jax.vmap(residual_at, in_axes=(None, None, 0, None, None))


def loss_with_components(
    current_params, weight, *, static_net, t_obs, y_obs, t_col, gamma_imposed, trial
):
    """Return ``data + weight * residual`` and the pair ``(data, residual)``."""
    trainable_net, log_theta = current_params
    current_net = eqx.combine(trainable_net, static_net)
    data_loss = jnp.mean((predict(current_net, t_obs, trial) - y_obs) ** 2)
    physics_loss = jnp.mean(residual(current_net, log_theta, t_col, gamma_imposed, trial) ** 2)
    return data_loss + weight * physics_loss, (data_loss, physics_loss)


def run_adam(params, optimizer, loss_fn, weight, steps, gamma_imposed, optimizer_state=None):
    """Run ``steps`` Adam updates under ``lax.scan``.

    Row ``k`` of the trace holds the parameters and losses of the iterate
    entering update ``k``, and a final row holds the iterate the run ends on,
    so the trace has ``steps + 1`` rows ``[omega, gamma, data, residual]``.
    Continuing a run must pass its saved ``optimizer_state``, because the
    schedule is indexed by the step count stored there.
    """
    if optimizer_state is None:
        optimizer_state = optimizer.init(params)

    def trace_row(current_params, losses):
        omega, gamma = coefficients(current_params[1], gamma_imposed)
        return jnp.array([omega, gamma, *losses])

    def step(carry, _):
        current_params, optimizer_state = carry
        (_, losses), gradients = jax.value_and_grad(loss_fn, has_aux=True)(current_params, weight)
        row = trace_row(current_params, losses)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, current_params)
        current_params = optax.apply_updates(current_params, updates)
        return (current_params, optimizer_state), row

    (params, optimizer_state), trace = jax.lax.scan(
        step, (params, optimizer_state), None, length=steps
    )
    _, losses = loss_fn(params, weight)
    trace = jnp.concatenate([trace, trace_row(params, losses)[None, :]], axis=0)
    return params, optimizer_state, trace


def make_observations(config):
    """Return ``(t_obs, y_obs, t_col)``; the noise depends on ``noise_seed`` only.

    The observations do not depend on the severity, so every severity and weight
    of a noise seed sees the same data.
    """
    rng = np.random.default_rng(config["noise_seed"])
    t_end = config["t_end"]
    t_obs = jnp.linspace(0.0, t_end, config["n_obs"])
    clean = exact_solution(t_obs, config["w0_true"], config["g_true"], config["u0"], config["v0"])
    y_obs = clean + jnp.asarray(rng.normal(0.0, config["sigma"], size=config["n_obs"]))
    t_col = jnp.linspace(0.0, t_end, config["n_col"])
    return t_obs, y_obs, t_col


def trial_spec(config):
    t_end = float(config["t_end"])
    return (
        float(config["u0"]),
        float(config["v0"]),
        t_end,
        float(config["trial_window_fraction"]) * t_end,
    )


def training_setup(config):
    """Return the starting parameters, the loss and the data of one run."""
    gamma_imposed = imposed_damping(config)
    trial = trial_spec(config)
    t_obs, y_obs, t_col = make_observations(config)
    trainable_net, static_net = eqx.partition(
        build_network(config, in_size=1), eqx.is_inexact_array
    )
    # With the damping fixed, only log(omega) is trained.
    initial_theta = (
        jnp.array([config["w0_init"], config["g_init"]])
        if gamma_imposed is None
        else jnp.array([config["w0_init"]])
    )
    loss = partial(
        loss_with_components,
        static_net=static_net,
        t_obs=t_obs,
        y_obs=y_obs,
        t_col=t_col,
        gamma_imposed=gamma_imposed,
        trial=trial,
    )
    return {
        "params": (trainable_net, jnp.log(initial_theta)),
        "static_net": static_net,
        "loss": loss,
        "labels": (jax.tree.map(lambda _: "net", trainable_net), "theta"),
        "t_obs": t_obs,
        "y_obs": y_obs,
        "t_col": t_col,
    }


def run_experiment(config, checkpoint=None):
    """Train one PINN and return its network, estimates, losses and trace.

    Args:
        config: A configuration from ``default_config``.
        checkpoint: ``(params, optimizer_state, steps_done)`` from an earlier
            part of the same run, or ``None``. Only the remaining updates run,
            against the full schedule horizon, so a split run reproduces an
            uninterrupted one.

    Returns:
        Dictionary with the network, parameters, optimizer state, the estimates
        ``w0`` and ``g``, the final losses, the trace and the data. Nothing is
        written to disk.
    """
    validate_config(config)
    started = time.perf_counter()
    gamma_imposed = imposed_damping(config)
    setup = training_setup(config)
    weight = config["physics_weight"]
    steps = int(config["physics_steps"])
    params, optimizer_state, steps_done = resume(checkpoint, steps, setup["params"])
    optimizer = optax.multi_transform(cosine_adam(config, steps), setup["labels"])
    params, optimizer_state, trace = run_adam(
        params,
        optimizer,
        setup["loss"],
        weight,
        steps - steps_done,
        gamma_imposed,
        optimizer_state,
    )
    if steps_done:
        # Row zero repeats the checkpoint's iterate, which the earlier part of
        # the run already recorded as its last row.
        trace = trace[1:]

    omega, gamma = map(float, coefficients(params[1], gamma_imposed))
    _, component_losses = setup["loss"](params, weight)
    data_loss, physics_loss = map(float, component_losses)
    jax.block_until_ready(params)
    return {
        "config": config,
        "net": eqx.combine(params[0], setup["static_net"]),
        "params": params,
        "optimizer_state": optimizer_state,
        "w0": omega,
        "g": gamma,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "trace": np.asarray(trace),
        "t_obs": setup["t_obs"],
        "y_obs": setup["y_obs"],
        "t_col": setup["t_col"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarise_run(result):
    """Return the configuration plus the run's estimates, errors and losses, as a row."""
    config = result["config"]
    w0_true, g_true = config["w0_true"], config["g_true"]
    row = dict(config)
    row.update(
        {
            "gamma_imposed": imposed_damping(config),
            "w0_final": result["w0"],
            "g_final": result["g"],
            "w0_absolute_error": abs(result["w0"] - w0_true),
            "g_absolute_error": abs(result["g"] - g_true),
            "w0_relative_error": abs(result["w0"] - w0_true) / abs(w0_true),
            "g_relative_error": abs(result["g"] - g_true) / abs(g_true),
            "data_loss": result["data_loss"],
            "physics_loss": result["physics_loss"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    return row


def state_errors(result, n_dense=1001):
    """Return ``(t_dense, u_net, u_true)`` on a dense grid over ``[0, T]``."""
    config = result["config"]
    t_dense = jnp.linspace(0.0, config["t_end"], n_dense)
    u_net = np.asarray(predict(result["net"], t_dense, trial_spec(config)))
    u_true = np.asarray(
        exact_solution(t_dense, config["w0_true"], config["g_true"], config["u0"], config["v0"])
    )
    return t_dense, u_net, u_true
