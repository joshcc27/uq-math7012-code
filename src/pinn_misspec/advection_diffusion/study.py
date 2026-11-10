"""The advection--diffusion study: designs, control limits, references and hooks.

The noise seed draws the observation and collocation points, so every reference is
computed per severity and noise seed; a solution fit already saved under the same
reference id is loaded rather than refitted.

Output of ``run_study``:

    study.csv                one row per PINN run
    references.csv           theta_sol(s) (``target``) and the noisy fits (``solution_fit``)
    residual_references.csv  theta_eq(s) (``residual_target``)
    profiles/*.npz           each solution-fit objective over Theta
    traces/<run-id>.npy      each trace, rows [a, kappa, data loss, residual loss(, mu)]
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from pinn_misspec import runner

from . import model
from .model import (
    PROTOCOL_VERSION,
    T_END,
    default_config,
    run_experiment,
    sink_rate,
    summarise_run,
)
from .targets import (
    equation_fit_target,
    fit_relaxed,
    fit_solution,
    network_record,
    relaxed_metrics,
    solution_fit_profile,
    study_metrics,
)

SEVERITIES = [0.0, 0.10, 0.25, 0.50, 1.00]
# The weights at which every control passed in the control-only screen.
FULL_DESIGN = {
    "severities": SEVERITIES,
    "scaled_weights": [0.01, 0.03, 0.10, 0.30],
    "noise_seeds": [0, 1, 2, 3, 4],
    "network_seeds": [0, 1, 2],
}
RELAXED_DESIGN = {**FULL_DESIGN, "scaled_weights": [0.01, 0.30], "network_seeds": [0]}

CONTROL_LIMITS = {
    "velocity_rel_error_true": 0.02,
    "diffusivity_rel_error_true": 0.10,
    "velocity_rel_error_sol": 0.02,
    "diffusivity_rel_error_sol": 0.10,
    "state_rel_l2_pinn": 0.04,
    "imposed_gap_rel_l2": 0.04,
    "mass_ratio_abs_error_true": 0.03,
}
DOMAIN_FLAG = "parameters_outside_domain"

# Everything that affects one solution-fit reference.
REFERENCE_KEYS = (
    "protocol",
    "velocity_true",
    "diffusivity_true",
    "t_end",
    "initial_centre",
    "initial_width",
    "fourier_modes",
    "severity",
    "sigma",
    "n_obs",
    "n_initial",
    "noise_seed",
    "velocity_min",
    "velocity_max",
    "diffusivity_min",
    "diffusivity_max",
)
RELAXED_REFERENCE_KEYS = REFERENCE_KEYS + ("sink_min", "sink_max")


def study_config(severity, scaled_weight, noise_seed, network_seed, overrides=None):
    """Return the configuration of one cell; ``overrides`` may not set a design axis."""
    return runner.study_config(model, severity, scaled_weight, noise_seed, network_seed, overrides)


def relaxed_config(severity, scaled_weight, noise_seed, network_seed, overrides=None):
    """Return a relaxed-check cell, with the sink trained."""
    overrides = runner.check_overrides(overrides, runner.DESIGN_AXES + ("fit_sink",))
    return study_config(
        severity, scaled_weight, noise_seed, network_seed, overrides={**overrides, "fit_sink": True}
    )


def reference_id(config, noise_free):
    """Hash everything that affects one solution-fit reference.

    The noise seed is part of it even for the noise-free fit, because it draws
    the observation locations.
    """
    identity = {key: config[key] for key in REFERENCE_KEYS}
    identity["noise_free"] = bool(noise_free)
    return runner.config_hash(identity)


def relaxed_reference_id(config, noise_free):
    """Hash everything that affects one three-coefficient reference."""
    identity = {key: config[key] for key in RELAXED_REFERENCE_KEYS}
    identity.update(noise_free=bool(noise_free), fit_sink=True)
    return runner.config_hash(identity)


def profile_filename(severity, noise_seed, rid, noise_free=False):
    kind = "noise-free" if noise_free else "noisy"
    return f"s-{runner.number_token(severity)}-seed-{int(noise_seed)}-{kind}-{rid}.npz"


def _cached_fit(row, names):
    return {
        **{name: float(row[name]) for name in names},
        "n_profile_minima": int(float(row["n_profile_minima"])),
        "at_boundary": row["at_boundary"].strip().lower() == "true",
    }


def compute_references(design, output, overrides=None):
    """Compute and save every reference of a design, per severity and noise seed.

    Returns:
        Mapping from ``(severity, noise_seed)`` to ``{"target", "solution_fit",
        "residual_target"}``.
    """
    overrides = runner.check_overrides(overrides)
    output = Path(output)
    profiles = output / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    path = output / "references.csv"
    existing = {row["reference_id"]: row for row in runner.read_rows(path)[1]}
    names = ("velocity", "diffusivity", "objective", "rmse", "runner_up_gap")
    references, rows, equation_rows = {}, [], []

    def record(kind, severity, noise_seed, config, noise_free):
        rid = reference_id(config, noise_free)
        previous = existing.get(rid)
        if previous is not None and (output / previous["profile_file"]).exists():
            return _cached_fit(previous, names)
        fit = fit_solution(config, noise_free)
        velocity_grid, diffusivity_grid, objective = solution_fit_profile(config, noise_free)
        filename = profile_filename(severity, noise_seed, rid, noise_free)
        np.savez(
            profiles / filename,
            velocity_grid=velocity_grid,
            diffusivity_grid=diffusivity_grid,
            objective=objective,
            severity=float(severity),
            sink_rate=sink_rate(config),
            noise_free=noise_free,
            noise_seed=int(noise_seed),
            velocity_selected=fit["velocity"],
            diffusivity_selected=fit["diffusivity"],
            reference_id=rid,
        )
        rows.append(
            {
                "reference_id": rid,
                "reference": kind,
                "severity": float(severity),
                "sink_rate": sink_rate(config),
                "noise_seed": int(noise_seed),
                **fit,
                "profile_file": f"profiles/{filename}",
                "protocol": config["protocol"],
            }
        )
        return fit

    for severity in design["severities"]:
        for noise_seed in design["noise_seeds"]:
            config = default_config(
                **overrides, severity=float(severity), noise_seed=int(noise_seed)
            )
            equation = equation_fit_target(config)
            equation_rows.append(
                {
                    "reference": "residual_target",
                    "severity": float(severity),
                    "sink_rate": sink_rate(config),
                    "noise_seed": int(noise_seed),
                    **equation,
                    "protocol": config["protocol"],
                }
            )
            references[(float(severity), int(noise_seed))] = {
                "target": record("target", severity, noise_seed, config, True),
                "solution_fit": record("solution_fit", severity, noise_seed, config, False),
                "residual_target": equation,
            }
    runner.merge_rows(path, rows, key=lambda row: row["reference_id"])
    runner.merge_rows(
        output / "residual_references.csv",
        equation_rows,
        key=lambda row: (row["reference"], row["severity"], row["noise_seed"]),
    )
    return references


def compute_relaxed_references(design, output, overrides=None):
    """Compute the three-coefficient references, loading any already saved."""
    overrides = runner.check_overrides(overrides, runner.DESIGN_AXES + ("fit_sink",))
    path = Path(output) / "references.csv"
    existing = {
        row["reference_id"]: row for row in runner.read_rows(path)[1] if row.get("reference_id")
    }
    names = ("velocity", "diffusivity", "sink", "objective", "rmse", "runner_up_gap")
    references, rows = {}, []
    for severity in design["severities"]:
        for noise_seed in design["noise_seeds"]:
            config = relaxed_config(severity, design["scaled_weights"][0], noise_seed, 0, overrides)
            pair = {}
            for kind, noise_free in (("target", True), ("solution_fit", False)):
                rid = relaxed_reference_id(config, noise_free)
                if rid in existing:
                    pair[kind] = _cached_fit(existing[rid], names)
                    continue
                fit = fit_relaxed(config, noise_free)
                rows.append(
                    {
                        "reference_id": rid,
                        "reference": kind,
                        "severity": float(severity),
                        "sink_rate": sink_rate(config),
                        "noise_seed": int(noise_seed),
                        **fit,
                        "protocol": config["protocol"],
                    }
                )
                pair[kind] = fit
            references[(float(severity), int(noise_seed))] = pair
    if rows:
        runner.merge_rows(path, rows, key=lambda row: row["reference_id"])
    return references


def _describe_run(cell, result, metrics):
    severity, scaled_weight, noise_seed, network_seed = cell
    return (
        f"s={severity:+.2f} Lambda={scaled_weight:<5g} noise={noise_seed} net={network_seed} | "
        f"a={result['velocity']:.5f}, k={result['diffusivity']:.5f} "
        f"gap={metrics['imposed_gap_rel_l2']:.2e}"
    )


def _describe_relaxed_run(cell, result, metrics):
    severity, scaled_weight, noise_seed, network_seed = cell
    return (
        f"s={severity:+.2f} Lambda={scaled_weight:<5g} noise={noise_seed} net={network_seed} | "
        f"a={result['velocity']:.5f}, k={result['diffusivity']:.5f}, "
        f"mu T={result['sink'] * T_END:.4f}"
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
        source=("pinn_misspec.advection_diffusion.study", "main_hooks"),
    )


def relaxed_hooks():
    """Return the hooks of the relaxed check, which is not gated."""
    return replace(
        main_hooks(),
        study_config=relaxed_config,
        compute_references=compute_relaxed_references,
        study_metrics=relaxed_metrics,
        describe_run=_describe_relaxed_run,
        source=("pinn_misspec.advection_diffusion.study", "relaxed_hooks"),
    )
