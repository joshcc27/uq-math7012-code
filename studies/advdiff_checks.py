"""The advection--diffusion relaxed check and the longer-budget studies.

Each part writes where the archived results are, under the results root
(``diagnostics`` is ``advection_diffusion/diagnostics``):

    relaxed        advection_diffusion/study/relaxed and relaxed/steps-10000
                   the sink fitted alongside the transport coefficients
    controls-20000 diagnostics/weight-screen-20000-2026-10-08
                   controls at Lambda = 1 and 3 with 20000 updates
    supplementary  diagnostics/supplementary-20000-2026-10-08
                   the grid at Lambda = 1 and 3 with 20000 updates, gated
    budget-20000   diagnostics/budget-20000-2026-10-08
                   network seed 0 at Lambda = 0.01 and 0.3 with 20000 updates

    uv run python studies/advdiff_checks.py --workers 6
"""

from pinn_misspec import runner
from pinn_misspec.advection_diffusion import study
from pinn_misspec.advection_diffusion.model import PHYSICS_STEPS

LONG = 20000
DIAGNOSTICS = "advection_diffusion/diagnostics"


def relaxed(steps, root, workers):
    for budget, folder in ((PHYSICS_STEPS, "relaxed"), (2 * PHYSICS_STEPS, "relaxed/steps-10000")):
        runner.run_study(
            study.RELAXED_DESIGN,
            root / "advection_diffusion/study" / folder,
            study.relaxed_hooks(),
            {"physics_steps": steps or budget},
            gate=False,
            workers=workers,
        )


def controls_20000(steps, root, workers):
    design = {**runner.control_slice(study.FULL_DESIGN), "scaled_weights": [1.0, 3.0]}
    runner.run_study(
        design,
        root / f"{DIAGNOSTICS}/weight-screen-20000-2026-10-08",
        study.main_hooks(),
        {"physics_steps": steps or LONG},
        workers=workers,
    )


def supplementary(steps, root, workers):
    """Run only after every control at that weight passed."""
    runner.run_study(
        {**study.FULL_DESIGN, "scaled_weights": [1.0, 3.0]},
        root / f"{DIAGNOSTICS}/supplementary-20000-2026-10-08",
        study.main_hooks(),
        {"physics_steps": steps or LONG},
        workers=workers,
    )


def budget_20000(steps, root, workers):
    design = {
        **study.FULL_DESIGN,
        "severities": [0.0, 0.5, 1.0],
        "scaled_weights": [0.01, 0.3],
        "network_seeds": [0],
    }
    runner.run_study(
        design,
        root / f"{DIAGNOSTICS}/budget-20000-2026-10-08",
        study.main_hooks(),
        {"physics_steps": steps or LONG},
        workers=workers,
    )


if __name__ == "__main__":
    runner.run_parts(
        {
            "relaxed": relaxed,
            "controls-20000": controls_20000,
            "supplementary": supplementary,
            "budget-20000": budget_20000,
        },
        runner.study_arguments(__doc__),
    )
