"""The advection--diffusion main grid, its control screen and its budget check.

Each part writes where the archived results are, under the results root:

    screen   advection_diffusion/protocol/rate-screen-2026-10-06/theta-<rate>
             controls at three coefficient learning rates
    main     advection_diffusion/study                  the main grid, gated
    budget   advection_diffusion/study/steps-10000      network seed 0, twice the budget

The archived rate screen was stopped before every planned run finished, so a full
rerun has more runs in some rows.

    uv run python studies/advdiff_main.py --workers 6
"""

from pinn_misspec import runner
from pinn_misspec.advection_diffusion import study
from pinn_misspec.advection_diffusion.model import PHYSICS_STEPS

SCREEN_RATES = {"0p003": 0.003, "0p001": 0.001, "0p0005": 0.0005}


def screen(steps, root, workers):
    design = {
        **runner.control_slice(study.FULL_DESIGN),
        "scaled_weights": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    }
    for label, rate in SCREEN_RATES.items():
        runner.run_study(
            design,
            root / f"advection_diffusion/protocol/rate-screen-2026-10-06/theta-{label}",
            study.main_hooks(),
            {"physics_steps": steps or PHYSICS_STEPS, "theta_learning_rate": rate},
            workers=workers,
        )


def main(steps, root, workers):
    runner.run_study(
        study.FULL_DESIGN,
        root / "advection_diffusion/study",
        study.main_hooks(),
        {"physics_steps": steps or PHYSICS_STEPS},
        workers=workers,
    )


def budget(steps, root, workers):
    """Network seed 0 of the grid at twice the budget, run fresh."""
    runner.run_study(
        {**study.FULL_DESIGN, "network_seeds": [0]},
        root / f"advection_diffusion/study/steps-{2 * PHYSICS_STEPS}",
        study.main_hooks(),
        {"physics_steps": steps or 2 * PHYSICS_STEPS},
        workers=workers,
    )


if __name__ == "__main__":
    runner.run_parts(
        {"screen": screen, "main": main, "budget": budget}, runner.study_arguments(__doc__)
    )
