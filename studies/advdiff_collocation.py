"""The collocation diagnostics: the residual between the points, density, nesting and the full grid.

Every part trains with 20000 updates and saves its networks, under the results root
(``diagnostics`` is ``advection_diffusion/diagnostics``):

    residual    diagnostics/collocation-residual-2026-10-08/<group>
                seven recorded cells rerun with networks saved, including one
                oscillator cell, to compare the residual on and off the points
    density     diagnostics/collocation-density-2026-10-08/n<N>
                600, 2400 and 9600 points, each density drawing its own points
    nested      diagnostics/collocation-nested-2026-10-11/n<N>
                the first 600 and 2400 of one 9600-point draw per noise seed
    full-grid   diagnostics/dense-grid-2026-10-25/n<N>
                the whole design at 600 and at 9600 points; a weight enters the
                comparison only if all its controls pass at both densities

Archived runs keep the formats of the original one-off scripts (``summary.csv`` and
``state-*.npz``, or ``cells/*.json``); reruns write ``study.csv`` with the same run
ids, one folder per density because the equation-fit target depends on the points.

    uv run python studies/advdiff_collocation.py --workers 6 --only nested
"""

from pinn_misspec import runner
from pinn_misspec.advection_diffusion import study
from pinn_misspec.oscillator import study as oscillator_study

LONG = 20000
DENSITIES = [600, 2400, 9600]
DIAGNOSTICS = "advection_diffusion/diagnostics"


def residual(steps, root, workers):
    """Not gated: these cells repeat recorded runs whose controls already passed."""
    folder = root / f"{DIAGNOSTICS}/collocation-residual-2026-10-08"
    one_seed = {"noise_seeds": [0], "network_seeds": [0]}
    groups = [
        ("advdiff-20000", {"severities": [0.0, 0.5, 1.0], "scaled_weights": [1.0, 3.0]}, LONG),
        ("advdiff-5000", {"severities": [1.0], "scaled_weights": [0.01, 0.3]}, 5000),
    ]
    for name, axes, budget in groups:
        runner.run_study(
            {**axes, **one_seed},
            folder / name,
            study.main_hooks(),
            {"physics_steps": steps or budget},
            gate=False,
            workers=workers,
            save_networks=True,
        )
    runner.run_study(
        {"severities": [2.0], "scaled_weights": [100.0], **one_seed},
        folder / "oscillator",
        oscillator_study.main_hooks(),
        {"physics_steps": steps or 9000},
        gate=False,
        workers=workers,
        save_networks=True,
    )


def density(steps, root, workers):
    design = {
        "severities": [0.0, 0.5, 1.0],
        "scaled_weights": [1.0, 3.0],
        "noise_seeds": [0, 1],
        "network_seeds": [0],
    }
    for count in DENSITIES:
        runner.run_study(
            design,
            root / f"{DIAGNOSTICS}/collocation-density-2026-10-08/n{count}",
            study.main_hooks(),
            {"physics_steps": steps or LONG, "n_col": count},
            gate=False,
            workers=workers,
            save_networks=True,
        )


def nested(steps, root, workers):
    design = {
        "severities": [0.0, 0.5, 1.0],
        "scaled_weights": [3.0],
        "noise_seeds": [0, 1],
        "network_seeds": [0, 1, 2],
    }
    for count in DENSITIES:
        runner.run_study(
            design,
            root / f"{DIAGNOSTICS}/collocation-nested-2026-10-11/n{count}",
            study.main_hooks(),
            {"physics_steps": steps or LONG, "n_col": count, "collocation_pool": max(DENSITIES)},
            gate=False,
            workers=workers,
            save_networks=True,
        )


def full_grid(steps, root, workers):
    """Controls at every weight, then the grid over the weights whose controls passed."""
    for count in (600, 9600):
        folder = root / f"{DIAGNOSTICS}/dense-grid-2026-10-25/n{count}"
        overrides = {"physics_steps": steps or LONG, "n_col": count}
        controls = {
            **runner.control_slice(study.FULL_DESIGN),
            "scaled_weights": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        }
        runner.run_study(
            controls, folder, study.main_hooks(), overrides, workers=workers, save_networks=True
        )
        runner.run_study(
            {**study.FULL_DESIGN, "scaled_weights": [0.1, 0.3, 1.0, 3.0]},
            folder,
            study.main_hooks(),
            overrides,
            workers=workers,
            save_networks=True,
        )


if __name__ == "__main__":
    runner.run_parts(
        {"residual": residual, "density": density, "nested": nested, "full-grid": full_grid},
        runner.study_arguments(__doc__),
    )
