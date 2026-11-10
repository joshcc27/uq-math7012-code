"""The oscillator's main grid, its control screen and its budget checks.

Each part writes where the archived results are, under the results root:

    screen      oscillator/protocol/weight-screen-2026-10-06    controls only
    main        oscillator/study                                the main grid, gated
    budget      oscillator/study/steps-18000                    network seed 0, twice the budget
    lambda100   oscillator/diagnostics/lambda100-18000-seeds-1-2-2026-10-11
                                                                Lambda = 100 at 18000 updates

    uv run python studies/oscillator_main.py --workers 6
    uv run python studies/oscillator_main.py --only screen --steps 50 --output scratch
"""

from pinn_misspec import runner
from pinn_misspec.oscillator import study
from pinn_misspec.oscillator.model import PHYSICS_STEPS

LONG = 2 * PHYSICS_STEPS


def screen(steps, root, workers):
    """Controls only, over the screened weights."""
    design = {
        **runner.control_slice(study.FULL_DESIGN),
        "scaled_weights": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000],
    }
    runner.run_study(
        design,
        root / "oscillator/protocol/weight-screen-2026-10-06",
        study.main_hooks(),
        {"physics_steps": steps or PHYSICS_STEPS},
        workers=workers,
    )


def main(steps, root, workers):
    runner.run_study(
        study.FULL_DESIGN,
        root / "oscillator/study",
        study.main_hooks(),
        {"physics_steps": steps or PHYSICS_STEPS},
        workers=workers,
    )


def budget(steps, root, workers):
    """Network seed 0 of the grid at twice the budget, run fresh."""
    runner.run_study(
        {**study.FULL_DESIGN, "network_seeds": [0]},
        root / f"oscillator/study/steps-{LONG}",
        study.main_hooks(),
        {"physics_steps": steps or LONG},
        workers=workers,
    )


def lambda100(steps, root, workers):
    """Network seeds 1 and 2 at Lambda = 100 and twice the budget, where 9000 was too short."""
    runner.run_study(
        {**study.FULL_DESIGN, "scaled_weights": [100.0], "network_seeds": [1, 2]},
        root / "oscillator/diagnostics/lambda100-18000-seeds-1-2-2026-10-11",
        study.main_hooks(),
        {"physics_steps": steps or LONG},
        workers=workers,
    )


if __name__ == "__main__":
    runner.run_parts(
        {"screen": screen, "main": main, "budget": budget, "lambda100": lambda100},
        runner.study_arguments(__doc__),
    )
