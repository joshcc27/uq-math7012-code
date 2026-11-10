"""The oscillator study: designs, control limits, references and hooks.

theta_sol(s) and theta_eq(s) are computed once per severity and the noisy solution
fit once per severity and noise seed; every run of a noise seed sees the same data.

Output of ``run_study``:

    study.csv                one row per PINN run
    references.csv           theta_sol(s) (``target``) and the noisy fits (``solution_fit``)
    residual_references.csv  theta_eq(s) (``residual_target``)
    profiles/*.npz           each solution-fit objective over Theta
    traces/<run-id>.npy      each trace, rows [omega, gamma, data loss, residual loss]
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from pinn_misspec import runner

from . import model
from .model import (
    PROTOCOL_VERSION,
    default_config,
    imposed_damping,
    raw_physics_weight,
    run_experiment,
    summarise_run,
)
from .targets import (
    equation_fit_target,
    fit_relaxed,
    fit_solution,
    network_record,
    relaxed_metrics,
    relaxed_profile,
    solution_fit_profile,
    study_metrics,
)

SEVERITIES = [-0.75, -0.5, -0.25, 0.0, 0.5, 1.0, 2.0]
# The weights at which every control passed in the control-only screen.
FULL_DESIGN = {
    "severities": SEVERITIES,
    "scaled_weights": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "noise_seeds": [0, 1, 2, 3, 4],
    "network_seeds": [0, 1, 2],
}
# The relaxed check frees the damping, so severity only sets its starting value.
RELAXED_DESIGN = {**FULL_DESIGN, "scaled_weights": [1.0, 100.0], "network_seeds": [0]}

# The state and gap limits sit just above the largest values the controls reached.
CONTROL_LIMITS = {
    "w0_rel_error_true": 0.02,
    "w0_rel_error_sol": 0.01,
    "state_rel_l2_pinn": 0.03,
    "imposed_gap_rel_l2": 0.04,
}
DOMAIN_FLAG = "w0_outside_domain"


def study_config(severity, scaled_weight, noise_seed, network_seed, overrides=None):
    """Return the configuration of one cell; ``overrides`` may not set a design axis."""
    return runner.study_config(model, severity, scaled_weight, noise_seed, network_seed, overrides)


def relaxed_config(severity, scaled_weight, noise_seed, network_seed, overrides=None):
    """Return a relaxed-check cell: damping trained, started at ``gamma_true (1 + s)``."""
    overrides = runner.check_overrides(overrides, runner.DESIGN_AXES + ("g_init",))
    base = default_config(**overrides)
    return default_config(
        **overrides,
        severity=None,
        g_init=float(base["g_true"]) * (1.0 + float(severity)),
        physics_weight=raw_physics_weight(scaled_weight, base),
        noise_seed=int(noise_seed),
        network_seed=int(network_seed),
    )


def profile_filename(severity, noise_seed):
    """Name a saved profile; ``noise_seed=None`` names the noise-free one."""
    seed = "noise-free" if noise_seed is None else f"seed-{int(noise_seed)}"
    return f"s-{runner.number_token(severity)}-{seed}.npz"


def compute_references(design, output, overrides=None):
    """Compute and save every reference of a design.

    theta_sol(s) and theta_eq(s) do not depend on the noise seed, so each is
    computed once per severity. Rows from other designs already in the two
    tables are kept.

    Returns:
        Mapping from ``(severity, noise_seed)`` to ``{"target", "solution_fit",
        "residual_target"}``.
    """
    overrides = runner.check_overrides(overrides)
    profiles = Path(output) / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    references, rows, equation_rows = {}, [], []

    def record(kind, severity, noise_seed, config, noise_free):
        fit = fit_solution(config, noise_free)
        omega_grid, objective = solution_fit_profile(config, noise_free)
        filename = profile_filename(severity, None if noise_free else noise_seed)
        np.savez(
            profiles / filename,
            w0_grid=omega_grid,
            objective=objective,
            severity=float(severity),
            gamma_imposed=imposed_damping(config),
            noise_free=noise_free,
            noise_seed=-1 if noise_free else int(noise_seed),
            w0_selected=fit["w0"],
        )
        rows.append(
            {
                "reference": kind,
                "severity": float(severity),
                "gamma_imposed": imposed_damping(config),
                "noise_seed": None if noise_free else int(noise_seed),
                **fit,
                "profile_file": f"profiles/{filename}",
                "protocol": config["protocol"],
            }
        )
        return fit

    for severity in design["severities"]:
        base = {**overrides, "severity": float(severity)}
        base_config = default_config(**base)
        target = record("target", severity, None, base_config, True)
        equation = equation_fit_target(base_config)
        equation_rows.append(
            {
                "reference": "residual_target",
                "severity": float(severity),
                "gamma_imposed": imposed_damping(base_config),
                "noise_seed": None,
                **equation,
                "protocol": base_config["protocol"],
            }
        )
        for noise_seed in design["noise_seeds"]:
            config = default_config(**base, noise_seed=int(noise_seed))
            references[(float(severity), int(noise_seed))] = {
                "target": target,
                "solution_fit": record("solution_fit", severity, noise_seed, config, False),
                "residual_target": equation,
            }

    def key(row):
        return row["reference"], row["severity"], row["noise_seed"]

    runner.merge_rows(Path(output) / "references.csv", rows, key=key)
    runner.merge_rows(Path(output) / "residual_references.csv", equation_rows, key=key)
    return references


def compute_relaxed_references(design, output, overrides=None):
    """Compute the two-coefficient references, once per noise seed.

    Severity only sets the PINN's starting damping, so it does not enter them.
    """
    profiles = Path(output) / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    references, rows = {}, []
    for noise_seed in design["noise_seeds"]:
        config = relaxed_config(0.0, design["scaled_weights"][0], noise_seed, 0, overrides)
        pair = {}
        for kind, noise_free in (("target", True), ("solution_fit", False)):
            fit = fit_relaxed(config, noise_free)
            omega_grid, gamma_grid, objective = relaxed_profile(config, noise_free)
            label = "noise-free" if noise_free else "noisy"
            filename = f"relaxed-seed-{int(noise_seed)}-{label}.npz"
            np.savez(
                profiles / filename,
                w0_grid=omega_grid,
                g_grid=gamma_grid,
                objective=objective,
                noise_free=noise_free,
                noise_seed=int(noise_seed),
                w0_selected=fit["w0"],
                g_selected=fit["g"],
            )
            rows.append(
                {
                    "reference": kind,
                    "noise_seed": int(noise_seed),
                    **fit,
                    "profile_file": f"profiles/{filename}",
                    "protocol": config["protocol"],
                }
            )
            pair[kind] = fit
        references[int(noise_seed)] = pair
    runner.merge_rows(
        Path(output) / "references.csv", rows, key=lambda row: (row["reference"], row["noise_seed"])
    )
    return references


def _describe_run(cell, result, metrics):
    severity, scaled_weight, noise_seed, network_seed = cell
    return (
        f"s={severity:+.2f} Lambda={scaled_weight:<5g} noise={noise_seed} net={network_seed} | "
        f"w0={result['w0']:.5f} (sol {metrics['w0_sol']:.5f}) "
        f"gap={metrics['imposed_gap_rel_l2']:.1e}"
    )


def _describe_relaxed_run(cell, result, metrics):
    severity, scaled_weight, noise_seed, network_seed = cell
    return (
        f"start s={severity:+.2f} Lambda={scaled_weight:<5g} noise={noise_seed} "
        f"net={network_seed} | w0={result['w0']:.5f} g={result['g']:.5f}"
    )


def main_hooks():
    return runner.StudyHooks(
        study_config=study_config,
        compute_references=compute_references,
        run_experiment=run_experiment,
        summarise_run=summarise_run,
        study_metrics=study_metrics,
        network_record=network_record,
        describe_run=_describe_run,
        control_limits=CONTROL_LIMITS,
        domain_flag=DOMAIN_FLAG,
        protocol=PROTOCOL_VERSION,
        source=("pinn_misspec.oscillator.study", "main_hooks"),
    )


def relaxed_hooks():
    """Return the hooks of the relaxed check, which is not gated."""
    return replace(
        main_hooks(),
        study_config=relaxed_config,
        compute_references=compute_relaxed_references,
        study_metrics=relaxed_metrics,
        describe_run=_describe_relaxed_run,
        domain_flag="parameters_outside_domain",
        # References are per noise seed; the row records the starting severity.
        reference_key=lambda cell: int(cell[2]),
        cell_fields=lambda cell: {
            "init_severity": float(cell[0]),
            "scaled_physics_weight": float(cell[1]),
        },
        source=("pinn_misspec.oscillator.study", "relaxed_hooks"),
    )
