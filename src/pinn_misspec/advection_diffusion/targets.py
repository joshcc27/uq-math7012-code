"""The advection--diffusion targets and the metrics of one run.

Solution fit: the conservative ``(a, kappa)`` whose exact solution best fits the data,
from a scan over Theta (uniform in a, geometric in kappa) and bounded L-BFGS-B from
every local minimum; on the noise-free state it is theta_sol(s). Equation fit: the
``(a, kappa)`` minimising the mean squared conservative residual along the generating
state, theta_eq(s), a two-column least-squares problem. The noise seed draws the
points, so every target is computed per noise seed.

Saved tables keep the archive's names: ``target`` is theta_sol(s), ``solution_fit``
the noisy fit and ``residual_target`` theta_eq(s).
"""

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from pinn_misspec.training import relative_error, relative_l2

from .model import (
    collocation_points,
    evaluation_grid,
    exact_derivatives,
    exact_solution,
    generating_solution,
    imposed_solution,
    make_observations,
    predict,
    residual,
    sink_rate,
    state_errors,
    trial_spec,
)

PROFILE_VELOCITY_POINTS = 31
PROFILE_DIFFUSIVITY_POINTS = 25
# The relaxed fit's coarse three-dimensional scan, refined from its best minima.
RELAXED_VELOCITY_POINTS = 11
RELAXED_DIFFUSIVITY_POINTS = 9
RELAXED_SINK_POINTS = 11
RELAXED_REFINEMENTS = 8
DENSE_X = 128
DENSE_T = 61


def objective(velocity, diffusivity, sink, points, values, config):
    prediction = exact_solution(points, velocity, diffusivity, sink, config)
    return float(np.mean((prediction - np.asarray(values)) ** 2))


def fit_data(config, noise_free):
    points, observations, _ = make_observations(config)
    points = np.asarray(points)
    values = generating_solution(points, config) if noise_free else np.asarray(observations)
    return points, values


def solution_fit_profile(
    config,
    noise_free,
    velocity_points=PROFILE_VELOCITY_POINTS,
    diffusivity_points=PROFILE_DIFFUSIVITY_POINTS,
):
    """Return the velocity grid, the diffusivity grid and the objective over them."""
    velocity_grid = np.linspace(config["velocity_min"], config["velocity_max"], velocity_points)
    diffusivity_grid = np.geomspace(
        config["diffusivity_min"], config["diffusivity_max"], diffusivity_points
    )
    points, values = fit_data(config, noise_free)
    profile = np.empty((diffusivity_points, velocity_points), dtype=float)
    for row, diffusivity in enumerate(diffusivity_grid):
        for column, velocity in enumerate(velocity_grid):
            profile[row, column] = objective(velocity, diffusivity, 0.0, points, values, config)
    return velocity_grid, diffusivity_grid, profile


def fit_solution(
    config,
    noise_free,
    velocity_points=PROFILE_VELOCITY_POINTS,
    diffusivity_points=PROFILE_DIFFUSIVITY_POINTS,
):
    """Scan the two-dimensional domain and refine every discrete local minimum.

    Returns:
        Dictionary with ``velocity``, ``diffusivity``, the ``objective`` and
        ``rmse``, ``n_profile_minima``, ``runner_up_gap`` and ``at_boundary``.
    """
    velocity_grid, diffusivity_grid, profile = solution_fit_profile(
        config, noise_free, velocity_points, diffusivity_points
    )
    if not np.isfinite(profile).all():
        raise ValueError("non-finite solution-fit profile")
    points, values = fit_data(config, noise_free)
    log_bounds = (np.log(config["diffusivity_min"]), np.log(config["diffusivity_max"]))

    def loss(raw):
        return objective(raw[0], np.exp(raw[1]), 0.0, points, values, config)

    local_indices = np.argwhere(profile <= minimum_filter(profile, size=3, mode="nearest"))
    best_grid_index = np.unravel_index(int(np.argmin(profile)), profile.shape)
    if not any(np.array_equal(index, best_grid_index) for index in local_indices):
        local_indices = np.vstack([local_indices, best_grid_index])

    candidates = []
    for row, column in local_indices:
        start = np.array([velocity_grid[column], np.log(diffusivity_grid[row])])
        result = minimize(
            loss,
            start,
            method="L-BFGS-B",
            bounds=[(config["velocity_min"], config["velocity_max"]), log_bounds],
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
        )
        grid_candidate = (
            float(velocity_grid[column]),
            float(diffusivity_grid[row]),
            float(profile[row, column]),
        )
        refined = (float(result.x[0]), float(np.exp(result.x[1])), float(result.fun))
        candidates.append(
            refined if result.success and result.fun <= grid_candidate[2] else grid_candidate
        )
    candidates.append(
        (
            float(velocity_grid[best_grid_index[1]]),
            float(diffusivity_grid[best_grid_index[0]]),
            float(profile[best_grid_index]),
        )
    )
    velocity_step = velocity_grid[1] - velocity_grid[0]
    log_diffusivity_step = np.log(diffusivity_grid[1] / diffusivity_grid[0])
    distinct = []
    for candidate in sorted(candidates, key=lambda item: item[2]):
        if all(
            abs(candidate[0] - kept[0]) > velocity_step
            or abs(np.log(candidate[1] / kept[1])) > log_diffusivity_step
            for kept in distinct
        ):
            distinct.append(candidate)
    velocity, diffusivity, value = distinct[0]
    runner_up = distinct[1][2] - value if len(distinct) > 1 else float("inf")
    velocity_at_boundary = np.isclose(
        velocity, [config["velocity_min"], config["velocity_max"]], atol=1e-7
    ).any()
    diffusivity_at_boundary = np.isclose(
        diffusivity, [config["diffusivity_min"], config["diffusivity_max"]], rtol=1e-6
    ).any()
    return {
        "velocity": velocity,
        "diffusivity": diffusivity,
        "objective": value,
        "rmse": float(np.sqrt(value)),
        "n_profile_minima": int(len(local_indices)),
        "runner_up_gap": float(runner_up),
        "at_boundary": bool(velocity_at_boundary or diffusivity_at_boundary),
    }


def equation_fit(u_t, u_x, u_xx):
    """Return the ``(a, kappa)`` minimising the mean squared conservative residual along a state.

    The residual ``u_t + a u_x - kappa u_xx`` is linear in ``(a, kappa)``, so
    the minimiser is a least-squares fit of ``-u_t`` on ``[u_x, -u_xx]``.
    ``diffusivity`` is NaN when the least-squares value is not positive, since
    the PINN's diffusivity is positive by construction; ``diffusivity_ls``
    keeps the raw value.
    """
    u_t, u_x, u_xx = (np.asarray(value, dtype=float) for value in (u_t, u_x, u_xx))
    design = np.column_stack([u_x, -u_xx])
    (velocity, diffusivity), *_ = np.linalg.lstsq(design, -u_t, rcond=None)
    remainder = u_t + velocity * u_x - diffusivity * u_xx
    return {
        "velocity": float(velocity),
        "diffusivity": float(diffusivity) if diffusivity > 0.0 else float("nan"),
        "diffusivity_ls": float(diffusivity),
        "residual_rms": float(np.sqrt(np.mean(remainder**2))),
    }


def equation_fit_target(config):
    """Return theta_eq(s): the equation fit along the generating state at the training collocation points."""
    derivatives = exact_derivatives(
        np.asarray(collocation_points(config)),
        config["velocity_true"],
        config["diffusivity_true"],
        sink_rate(config),
        config,
    )
    return equation_fit(derivatives["u_t"], derivatives["u_x"], derivatives["u_xx"])


def fit_relaxed(config, noise_free):
    """Fit velocity, diffusivity and sink: coarse scan, then refine the best minima.

    The best ``RELAXED_REFINEMENTS`` grid minima are refined with bounded
    L-BFGS-B in ``(a, log kappa, mu)``; the result is never worse than the scan.
    """
    points, values = fit_data(config, noise_free)
    velocity_grid = np.linspace(
        config["velocity_min"], config["velocity_max"], RELAXED_VELOCITY_POINTS
    )
    diffusivity_grid = np.geomspace(
        config["diffusivity_min"], config["diffusivity_max"], RELAXED_DIFFUSIVITY_POINTS
    )
    sink_grid = np.linspace(config["sink_min"], config["sink_max"], RELAXED_SINK_POINTS)
    profile = np.empty((len(velocity_grid), len(diffusivity_grid), len(sink_grid)))
    for i, velocity in enumerate(velocity_grid):
        for j, diffusivity in enumerate(diffusivity_grid):
            for k, sink in enumerate(sink_grid):
                profile[i, j, k] = objective(velocity, diffusivity, sink, points, values, config)
    if not np.isfinite(profile).all():
        raise ValueError("non-finite relaxed solution-fit profile")

    def loss(raw):
        return objective(raw[0], np.exp(raw[1]), raw[2], points, values, config)

    bounds = [
        (config["velocity_min"], config["velocity_max"]),
        (np.log(config["diffusivity_min"]), np.log(config["diffusivity_max"])),
        (config["sink_min"], config["sink_max"]),
    ]
    local = np.argwhere(profile <= minimum_filter(profile, size=3, mode="nearest"))
    order = np.argsort(profile[tuple(local.T)])
    candidates = []
    for i, j, k in local[order[:RELAXED_REFINEMENTS]]:
        grid_point = (
            float(velocity_grid[i]),
            float(diffusivity_grid[j]),
            float(sink_grid[k]),
            float(profile[i, j, k]),
        )
        result = minimize(
            loss,
            np.array([velocity_grid[i], np.log(diffusivity_grid[j]), sink_grid[k]]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 1000},
        )
        refined = (
            float(result.x[0]),
            float(np.exp(result.x[1])),
            float(result.x[2]),
            float(result.fun),
        )
        candidates.append(
            refined if np.isfinite(result.fun) and result.fun <= grid_point[3] else grid_point
        )
    candidates.sort(key=lambda item: item[3])
    velocity, diffusivity, sink, value = candidates[0]
    others = [
        item[3]
        for item in candidates[1:]
        if abs(item[0] - velocity) > 1e-4
        or abs(np.log(item[1] / diffusivity)) > 1e-3
        or abs(item[2] - sink) > 1e-3
    ]
    at_boundary = (
        np.isclose(velocity, [config["velocity_min"], config["velocity_max"]], atol=1e-7).any()
        or np.isclose(
            diffusivity, [config["diffusivity_min"], config["diffusivity_max"]], rtol=1e-6
        ).any()
        or np.isclose(sink, [config["sink_min"], config["sink_max"]], atol=1e-7).any()
    )
    return {
        "velocity": velocity,
        "diffusivity": diffusivity,
        "sink": sink,
        "objective": value,
        "rmse": float(np.sqrt(value)),
        "n_profile_minima": int(len(local)),
        "runner_up_gap": float(min(others) - value) if others else float("inf"),
        "at_boundary": bool(at_boundary),
    }


def mass_ratio(values, n_x):
    """Return the final-to-initial spatial mean of a flattened time-major grid."""
    field = np.asarray(values).reshape(-1, n_x)
    return float(field[-1].mean() / field[0].mean())


def _centre(field, x):
    """Return the centre of each time slice, from the phase of its first Fourier mode."""
    phase = np.angle(field @ np.exp(-2j * np.pi * x))
    return -np.unwrap(phase) / (2.0 * np.pi)


def network_record(result, n_x=DENSE_X, n_t=DENSE_T):
    """Return the residual on and off the collocation points, for saved networks.

    Also tracks the mass and the centre of the network's state, the generating state and
    the imposed solution at the network's coefficients, so their transport speeds can be
    compared.

    Returns:
        The row entries and the arrays saved beside the network.
    """
    config = result["config"]
    net, theta, trial = result["net"], result["params"][1], trial_spec(config)
    collocation = result["collocation_points"]
    r_col = np.asarray(residual(net, theta, collocation, trial))
    points, x, t = evaluation_grid(config, n_x, n_t)
    r_dense = np.asarray(residual(net, theta, jnp.asarray(points), trial))
    shape = (len(t), len(x))
    sink = result["sink"] if result["sink"] is not None else 0.0
    fields = {
        "pinn": np.asarray(predict(net, jnp.asarray(points), trial)).reshape(shape),
        "true": generating_solution(points, config).reshape(shape),
        "imposed": exact_solution(
            points, result["velocity"], result["diffusivity"], sink, config
        ).reshape(shape),
    }
    centres = {name: _centre(field, x) for name, field in fields.items()}
    speeds = {name: float(np.polyfit(t, path, 1)[0]) for name, path in centres.items()}
    metrics = {
        "mse_collocation": float(np.mean(r_col**2)),
        "mse_dense": float(np.mean(r_dense**2)),
        "speed_state": speeds["pinn"],
        "speed_true": speeds["true"],
        "speed_imposed": speeds["imposed"],
    }
    arrays = {
        "x": x,
        "t": t,
        "u_net": fields["pinn"],
        "u_true": fields["true"],
        "u_imposed": fields["imposed"],
        "residual_dense": r_dense.reshape(shape),
        "residual_collocation": r_col,
        "collocation": np.asarray(collocation),
        **{f"centre_{name}": path for name, path in centres.items()},
        **{f"mass_{name}": field.mean(axis=1) for name, field in fields.items()},
    }
    return metrics, arrays


def relaxed_metrics(result, references, n_x=DENSE_X, n_t=DENSE_T):
    """Return the row entries of one relaxed-check run.

    The sink error is given as ``|mu_hat - mu| T``, because the control's true
    sink is zero and a relative error is undefined there.
    """
    config = result["config"]
    if not config["fit_sink"]:
        raise ValueError("The relaxed check fits the sink, so fit_sink must be set.")
    velocity, diffusivity, sink = result["velocity"], result["diffusivity"], result["sink"]
    target, solution = references["target"], references["solution_fit"]
    t_end, sink_true = float(config["t_end"]), sink_rate(config)
    points, neural, truth = state_errors(result, n_x, n_t)
    solution_state = exact_solution(
        points, solution["velocity"], solution["diffusivity"], solution["sink"], config
    )
    exact_at_pinn = exact_solution(points, velocity, diffusivity, sink, config)
    neural_mass_ratio = mass_ratio(neural, n_x)
    true_mass_ratio = float(np.exp(-config["severity"]))

    inside = (
        config["velocity_min"] <= velocity <= config["velocity_max"]
        and config["diffusivity_min"] <= diffusivity <= config["diffusivity_max"]
        and config["sink_min"] <= sink <= config["sink_max"]
    )
    return {
        "velocity_target": target["velocity"],
        "diffusivity_target": target["diffusivity"],
        "sink_target": target["sink"],
        "velocity_sol": solution["velocity"],
        "diffusivity_sol": solution["diffusivity"],
        "sink_sol": solution["sink"],
        "velocity_rel_error_true": relative_error(velocity, config["velocity_true"]),
        "diffusivity_rel_error_true": relative_error(diffusivity, config["diffusivity_true"]),
        "sink_scaled_error_true": abs(sink - sink_true) * t_end,
        "velocity_rel_error_sol": relative_error(velocity, solution["velocity"]),
        "diffusivity_rel_error_sol": relative_error(diffusivity, solution["diffusivity"]),
        "sink_scaled_error_sol": abs(sink - solution["sink"]) * t_end,
        "state_rel_l2_pinn": relative_l2(neural, truth),
        "state_rel_l2_sol": relative_l2(solution_state, truth),
        "imposed_gap_rel_l2": relative_l2(neural, exact_at_pinn),
        "mass_ratio_pinn": neural_mass_ratio,
        "mass_ratio_true": true_mass_ratio,
        "mass_ratio_abs_error_true": abs(neural_mass_ratio - true_mass_ratio),
        "data_loss": result["data_loss"],
        "physics_loss": result["physics_loss"],
        "parameters_outside_domain": not inside,
    }


def study_metrics(result, references, n_x=DENSE_X, n_t=DENSE_T):
    """Return the row entries of one severity-study run.

    Args:
        result: A finished run from ``run_experiment``.
        references: ``target`` (theta_sol), ``solution_fit`` (the noisy fit)
            and ``residual_target`` (theta_eq) for the run's severity and seed.
        n_x: Positions of the state-evaluation grid.
        n_t: Times of the state-evaluation grid.

    Returns:
        Coefficient errors, the state error, the imposed-solution gap, the mass
        ratios, the losses and the out-of-domain flag.
    """
    config = result["config"]
    velocity, diffusivity = result["velocity"], result["diffusivity"]
    target, solution = references["target"], references["solution_fit"]
    equation = references["residual_target"]
    points, neural, truth = state_errors(result, n_x, n_t)
    solution_state = imposed_solution(points, solution["velocity"], solution["diffusivity"], config)
    imposed_at_pinn = imposed_solution(points, velocity, diffusivity, config)
    neural_mass_ratio = mass_ratio(neural, n_x)
    true_mass_ratio = float(np.exp(-config["severity"]))

    return {
        "velocity_target": target["velocity"],
        "diffusivity_target": target["diffusivity"],
        "velocity_sol": solution["velocity"],
        "diffusivity_sol": solution["diffusivity"],
        "velocity_residual_target": equation["velocity"],
        "diffusivity_residual_target": equation["diffusivity"],
        "velocity_rel_error_true": relative_error(velocity, config["velocity_true"]),
        "diffusivity_rel_error_true": relative_error(diffusivity, config["diffusivity_true"]),
        "velocity_rel_error_target": relative_error(velocity, target["velocity"]),
        "diffusivity_rel_error_target": relative_error(diffusivity, target["diffusivity"]),
        "velocity_rel_error_sol": relative_error(velocity, solution["velocity"]),
        "diffusivity_rel_error_sol": relative_error(diffusivity, solution["diffusivity"]),
        "state_rel_l2_pinn": relative_l2(neural, truth),
        "state_rel_l2_sol": relative_l2(solution_state, truth),
        "imposed_gap_rel_l2": relative_l2(neural, imposed_at_pinn),
        "mass_ratio_pinn": neural_mass_ratio,
        "mass_ratio_true": true_mass_ratio,
        "mass_ratio_abs_error_true": abs(neural_mass_ratio - true_mass_ratio),
        "mass_ratio_abs_error_imposed": abs(neural_mass_ratio - 1.0),
        "data_loss": result["data_loss"],
        "physics_loss": result["physics_loss"],
        "parameters_outside_domain": not (
            config["velocity_min"] <= velocity <= config["velocity_max"]
            and config["diffusivity_min"] <= diffusivity <= config["diffusivity_max"]
        ),
    }
