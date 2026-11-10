"""The oscillator's network-free targets and the metrics of one run.

Solution fit: with the damping fixed at gamma_s, the frequency in Theta whose closed
form best fits the data. The objective has competing phase minima, so Theta is
scanned and every interior minimum refined. On the noise-free trajectory this is
theta_sol(s); on the noisy data it is the fit used to accept the controls.

Equation fit: the frequency minimising the mean squared imposed residual along the
generating state, theta_eq(s). The residual is linear in omega^2.

Saved tables keep the archive's names: ``target`` is theta_sol(s), ``solution_fit``
the noisy fit and ``residual_target`` theta_eq(s).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize, minimize_scalar

from pinn_misspec.training import relative_error, relative_l2

from .model import (
    exact_solution,
    imposed_damping,
    make_observations,
    residual,
    state_errors,
    trial_spec,
)

PROFILE_POINTS = 2001  # scan points over Theta
REFINE_XATOL = 1e-10  # frequency tolerance of each refinement
DENSE_POINTS = 1001  # state-evaluation grid
# The relaxed fit scans frequency uniformly and damping geometrically. The
# frequency spacing, about 0.04 rad/s, is far finer than the phase minima.
RELAXED_W0_POINTS = 241
RELAXED_G_POINTS = 49
RELAXED_REFINEMENTS = 20  # best grid minima refined


def solution_fit_objective(omega, gamma, t, y, u0, v0):
    return jnp.mean((exact_solution(t, omega, gamma, u0, v0) - y) ** 2)


_objective = jax.jit(solution_fit_objective)
_profile = jax.jit(jax.vmap(solution_fit_objective, in_axes=(0, None, None, None, None, None)))


def _fixed_damping(config):
    gamma = imposed_damping(config)
    if gamma is None:
        raise ValueError("Solution fitting estimates the frequency only, so severity must be set.")
    return gamma


def fit_data(config, noise_free):
    t_obs, y_obs, _ = make_observations(config)
    if noise_free:
        y_obs = exact_solution(
            t_obs, config["w0_true"], config["g_true"], config["u0"], config["v0"]
        )
    return t_obs, y_obs


def solution_fit_profile(config, noise_free, grid_size=PROFILE_POINTS):
    """Return ``(omega_grid, objective)`` on a uniform grid over Theta."""
    gamma = _fixed_damping(config)
    t_obs, y = fit_data(config, noise_free)
    omega_grid = np.linspace(float(config["w0_min"]), float(config["w0_max"]), grid_size)
    objective = _profile(
        jnp.asarray(omega_grid), gamma, t_obs, y, float(config["u0"]), float(config["v0"])
    )
    return omega_grid, np.asarray(objective)


def fit_solution(config, noise_free, grid_size=PROFILE_POINTS):
    """Select the best frequency in Theta by scanning, then refining each minimum.

    A refinement that comes out worse than its own grid point is replaced by
    that point, and the best grid point and both endpoints stay candidates, so
    the result is never worse than the scan.

    Returns:
        Dictionary with ``w0``, its ``objective`` and ``rmse``,
        ``n_interior_minima`` of the scan, ``runner_up_gap`` to the next
        distinct minimum (infinite if there is none), and ``at_boundary``.
    """
    gamma = _fixed_damping(config)
    omega_grid, profile = solution_fit_profile(config, noise_free, grid_size)
    if not np.all(np.isfinite(profile)):
        raise ValueError("Non-finite solution-fit objective on the profile grid")
    t_obs, y = fit_data(config, noise_free)
    u0, v0 = float(config["u0"]), float(config["v0"])

    def objective(omega):
        return float(_objective(omega, gamma, t_obs, y, u0, v0))

    interior = np.flatnonzero((profile[1:-1] <= profile[:-2]) & (profile[1:-1] <= profile[2:])) + 1
    # Each local minimum as (frequency, objective). An endpoint counts only
    # when the profile rises into the interior from it.
    minima = []
    for index in interior:
        result = minimize_scalar(
            objective,
            bounds=(omega_grid[index - 1], omega_grid[index + 1]),
            method="bounded",
            options={"xatol": REFINE_XATOL},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Refinement of the minimum near w0={omega_grid[index]:.6f} failed")
        if result.fun <= profile[index]:
            minima.append((float(result.x), float(result.fun)))
        else:
            minima.append((float(omega_grid[index]), float(profile[index])))
    if profile[0] < profile[1]:
        minima.append((float(omega_grid[0]), float(profile[0])))
    if profile[-1] < profile[-2]:
        minima.append((float(omega_grid[-1]), float(profile[-1])))
    # Neighbouring grid minima on a flat stretch can refine to the same place;
    # keep the lower of any two closer than one grid step.
    spacing = omega_grid[1] - omega_grid[0]
    distinct = []
    for candidate in sorted(minima, key=lambda pair: pair[1]):
        if all(abs(candidate[0] - kept[0]) > spacing for kept in distinct):
            distinct.append(candidate)

    best_index = int(np.argmin(profile))
    candidates = distinct + [
        (float(omega_grid[best_index]), float(profile[best_index])),
        (float(omega_grid[0]), float(profile[0])),
        (float(omega_grid[-1]), float(profile[-1])),
    ]
    omega, value = min(candidates, key=lambda pair: pair[1])
    others = [pair[1] for pair in distinct if abs(pair[0] - omega) > spacing]
    return {
        "w0": omega,
        "objective": value,
        "rmse": float(np.sqrt(value)),
        "n_interior_minima": int(len(interior)),
        "runner_up_gap": float(min(others) - value) if others else float("inf"),
        "at_boundary": bool(omega == omega_grid[0] or omega == omega_grid[-1]),
    }


_du_exact = jax.vmap(jax.grad(exact_solution, argnums=0), in_axes=(0, None, None, None, None))
_d2u_exact = jax.vmap(
    jax.grad(jax.grad(exact_solution, argnums=0), argnums=0),
    in_axes=(0, None, None, None, None),
)


def equation_fit(u, du, d2u, gamma):
    """Return the frequency minimising the mean squared imposed residual along a state.

    The residual ``u'' + 2 gamma u' + omega^2 u`` is linear in ``omega^2``, so
    the minimiser is ``omega^2 = -sum u (u'' + 2 gamma u') / sum u^2``. The
    frequency is NaN when that is not positive.
    """
    u, du, d2u = (np.asarray(value, dtype=float) for value in (u, du, d2u))
    forcing = d2u + 2.0 * float(gamma) * du
    omega_squared = float(-np.dot(u, forcing) / np.dot(u, u))
    remainder = forcing + omega_squared * u
    return {
        "w0": float(np.sqrt(omega_squared)) if omega_squared > 0.0 else float("nan"),
        "w0_squared": omega_squared,
        "residual_rms": float(np.sqrt(np.mean(remainder**2))),
    }


def exact_state_derivatives(config, t):
    """Return the generating state and its first two derivatives at ``t``.

    The derivatives come from differentiating the closed form, not from the
    equation, so a residual built from them tests the equation.
    """
    t = jnp.asarray(t)
    args = (config["w0_true"], config["g_true"], config["u0"], config["v0"])
    return tuple(
        np.asarray(function(t, *args)) for function in (exact_solution, _du_exact, _d2u_exact)
    )


def equation_fit_target(config):
    """Return theta_eq(s): the equation fit along the generating state at the collocation times."""
    gamma = _fixed_damping(config)
    _, _, t_col = make_observations(config)
    return equation_fit(*exact_state_derivatives(config, t_col), gamma)


_relaxed_profile = jax.jit(
    jax.vmap(
        jax.vmap(solution_fit_objective, in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None),
    )
)


def _relaxed_value_and_grad(t, y, u0, v0):
    """Return the objective in ``(omega, log gamma)`` and its gradient, for SciPy."""
    function = jax.jit(
        jax.value_and_grad(
            lambda raw: solution_fit_objective(raw[0], jnp.exp(raw[1]), t, y, u0, v0)
        )
    )

    def value_and_grad(raw):
        value, gradient = function(jnp.asarray(raw))
        return float(value), np.asarray(gradient, dtype=float)

    return value_and_grad


def relaxed_profile(config, noise_free, w0_points=RELAXED_W0_POINTS, g_points=RELAXED_G_POINTS):
    """Return ``(omega_grid, gamma_grid, objective)``, the objective shaped ``(gamma, omega)``."""
    t_obs, y = fit_data(config, noise_free)
    omega_grid = np.linspace(float(config["w0_min"]), float(config["w0_max"]), w0_points)
    gamma_grid = np.geomspace(float(config["g_min"]), float(config["g_max"]), g_points)
    objective = _relaxed_profile(
        jnp.asarray(omega_grid),
        jnp.asarray(gamma_grid),
        t_obs,
        y,
        float(config["u0"]),
        float(config["v0"]),
    )
    return omega_grid, gamma_grid, np.asarray(objective)


def fit_relaxed(config, noise_free, w0_points=RELAXED_W0_POINTS, g_points=RELAXED_G_POINTS):
    """Fit frequency and damping together: scan, then refine the best minima.

    The best ``RELAXED_REFINEMENTS`` grid minima are refined with bounded
    L-BFGS-B in ``(omega, log gamma)``; a refinement worse than its grid point
    is replaced by that point.
    """
    omega_grid, gamma_grid, profile = relaxed_profile(config, noise_free, w0_points, g_points)
    if not np.all(np.isfinite(profile)):
        raise ValueError("Non-finite relaxed solution-fit objective on the grid")
    t_obs, y = fit_data(config, noise_free)
    value_and_grad = _relaxed_value_and_grad(t_obs, y, float(config["u0"]), float(config["v0"]))
    bounds = [
        (float(config["w0_min"]), float(config["w0_max"])),
        (float(np.log(config["g_min"])), float(np.log(config["g_max"]))),
    ]
    local = np.argwhere(profile <= minimum_filter(profile, size=3, mode="nearest"))
    order = np.argsort(profile[local[:, 0], local[:, 1]])
    candidates = []
    for row, column in local[order[:RELAXED_REFINEMENTS]]:
        grid_point = (
            float(omega_grid[column]),
            float(gamma_grid[row]),
            float(profile[row, column]),
        )
        result = minimize(
            value_and_grad,
            np.array([omega_grid[column], np.log(gamma_grid[row])]),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
        )
        refined = (float(result.x[0]), float(np.exp(result.x[1])), float(result.fun))
        candidates.append(
            refined if np.isfinite(result.fun) and result.fun <= grid_point[2] else grid_point
        )
    omega_step = omega_grid[1] - omega_grid[0]
    log_gamma_step = np.log(gamma_grid[1] / gamma_grid[0])
    distinct = []
    for candidate in sorted(candidates, key=lambda item: item[2]):
        if all(
            abs(candidate[0] - kept[0]) > omega_step
            or abs(np.log(candidate[1] / kept[1])) > log_gamma_step
            for kept in distinct
        ):
            distinct.append(candidate)
    omega, gamma, value = distinct[0]
    at_boundary = (
        np.isclose(omega, [config["w0_min"], config["w0_max"]], rtol=1e-9).any()
        or np.isclose(gamma, [config["g_min"], config["g_max"]], rtol=1e-7).any()
    )
    return {
        "w0": omega,
        "g": gamma,
        "objective": value,
        "rmse": float(np.sqrt(value)),
        "n_profile_minima": int(len(local)),
        "runner_up_gap": float(distinct[1][2] - value) if len(distinct) > 1 else float("inf"),
        "at_boundary": bool(at_boundary),
    }


def network_record(result, n_dense=DENSE_POINTS):
    """Return the residual on and off the collocation times, for saved networks.

    The held-out times are the dense grid without the collocation times.

    Returns:
        The row entries and the arrays saved beside the network.
    """
    config = result["config"]
    net, log_theta = result["net"], result["params"][1]
    gamma, trial = imposed_damping(config), trial_spec(config)
    t_col = np.asarray(result["t_col"])
    r_col = np.asarray(residual(net, log_theta, jnp.asarray(t_col), gamma, trial))
    dense = np.linspace(0.0, config["t_end"], n_dense)
    held_out = ~np.isclose(dense[:, None], t_col[None, :], rtol=0.0, atol=1e-9).any(axis=1)
    r_dense = np.asarray(residual(net, log_theta, jnp.asarray(dense[held_out]), gamma, trial))
    metrics = {
        "mse_collocation": float(np.mean(r_col**2)),
        "mse_dense": float(np.mean(r_dense**2)),
        "held_out_points": int(held_out.sum()),
    }
    arrays = {
        "t_col": t_col,
        "residual_collocation": r_col,
        "t_held_out": dense[held_out],
        "residual_held_out": r_dense,
    }
    return metrics, arrays


def relaxed_metrics(result, references, n_dense=DENSE_POINTS):
    """Return the row entries of one relaxed-check run."""
    config = result["config"]
    if imposed_damping(config) is not None:
        raise ValueError("The relaxed check fits damping, so severity must be unset.")
    u0, v0 = float(config["u0"]), float(config["v0"])
    omega_hat, gamma_hat = result["w0"], result["g"]
    target, solution = references["target"], references["solution_fit"]
    t_dense, u_net, u_true = state_errors(result, n_dense)
    u_sol = exact_solution(t_dense, solution["w0"], solution["g"], u0, v0)
    u_imposed = exact_solution(t_dense, omega_hat, gamma_hat, u0, v0)

    inside = float(config["w0_min"]) <= omega_hat <= float(config["w0_max"]) and float(
        config["g_min"]
    ) <= gamma_hat <= float(config["g_max"])
    return {
        "w0_target": target["w0"],
        "g_target": target["g"],
        "w0_sol": solution["w0"],
        "g_sol": solution["g"],
        "w0_rel_error_true": relative_error(omega_hat, float(config["w0_true"])),
        "g_rel_error_true": relative_error(gamma_hat, float(config["g_true"])),
        "w0_rel_error_sol": relative_error(omega_hat, solution["w0"]),
        "g_rel_error_sol": relative_error(gamma_hat, solution["g"]),
        "state_rel_l2_pinn": relative_l2(u_net, u_true),
        "state_rel_l2_sol": relative_l2(u_sol, u_true),
        "imposed_gap_rel_l2": relative_l2(u_net, u_imposed),
        "data_loss": result["data_loss"],
        "physics_loss": result["physics_loss"],
        "parameters_outside_domain": not inside,
    }


def study_metrics(result, references, n_dense=DENSE_POINTS):
    """Return the row entries of one severity-study run.

    Args:
        result: A finished run from ``run_experiment``.
        references: ``target`` (theta_sol), ``solution_fit`` (the noisy fit)
            and ``residual_target`` (theta_eq) for the run's severity and
            noise seed.
        n_dense: Points of the state-evaluation grid.

    Returns:
        The coefficient errors, the state error, the imposed-solution gap (the
        state against the exact imposed solution at the PINN's own frequency),
        the losses and the out-of-domain flag.
    """
    config = result["config"]
    gamma = _fixed_damping(config)
    u0, v0 = float(config["u0"]), float(config["v0"])
    omega_hat, omega_true = result["w0"], float(config["w0_true"])
    omega_sol_target = references["target"]["w0"]
    omega_sol = references["solution_fit"]["w0"]
    omega_eq = references["residual_target"]["w0"]

    t_dense, u_net, u_true = state_errors(result, n_dense)
    u_sol = exact_solution(t_dense, omega_sol, gamma, u0, v0)
    u_imposed = exact_solution(t_dense, omega_hat, gamma, u0, v0)
    return {
        "w0_target": omega_sol_target,
        "w0_sol": omega_sol,
        "w0_residual_target": omega_eq,
        "w0_rel_error_true": abs(omega_hat - omega_true) / omega_true,
        "w0_rel_error_target": abs(omega_hat - omega_sol_target) / omega_sol_target,
        "w0_rel_error_sol": abs(omega_hat - omega_sol) / omega_sol,
        "state_rel_l2_pinn": relative_l2(u_net, u_true),
        "state_rel_l2_sol": relative_l2(u_sol, u_true),
        "imposed_gap_rel_l2": relative_l2(u_net, u_imposed),
        "data_loss": result["data_loss"],
        "physics_loss": result["physics_loss"],
        # A NaN estimate fails the range test, so it is flagged too.
        "w0_outside_domain": not (float(config["w0_min"]) <= omega_hat <= float(config["w0_max"])),
    }
