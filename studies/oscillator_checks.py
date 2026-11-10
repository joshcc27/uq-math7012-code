"""The oscillator's relaxed check, the damping-noise follow-up and the stronger weights.

Each part writes where the archived results are, under the results root:

    relaxed         oscillator/study/relaxed and relaxed/steps-18000
                    damping fitted alongside the frequency, at 9000 and 18000 updates
    damping-noise   oscillator/diagnostics/relaxed-damping-noise-2026-10-15
                    controls and relaxed fits at three noise seeds whose solution fits err low
    large-weight    oscillator/diagnostics/large-weight-budget-2026-10-25
                    controls at Lambda = 300 and 1000 with four times the budget

    uv run python studies/oscillator_checks.py --workers 6
"""

from pinn_misspec import runner
from pinn_misspec.oscillator import study
from pinn_misspec.oscillator.model import PHYSICS_STEPS

NEGATIVE_SEEDS = [408, 580, 928]


def relaxed(steps, root, workers):
    for budget, folder in ((PHYSICS_STEPS, "relaxed"), (2 * PHYSICS_STEPS, "relaxed/steps-18000")):
        runner.run_study(
            study.RELAXED_DESIGN,
            root / "oscillator/study" / folder,
            study.relaxed_hooks(),
            {"physics_steps": steps or budget},
            gate=False,
            workers=workers,
        )


def damping_noise(steps, root, workers):
    folder = root / "oscillator/diagnostics/relaxed-damping-noise-2026-10-15"
    controls = {
        **runner.control_slice(study.FULL_DESIGN),
        "scaled_weights": [1.0],
        "noise_seeds": NEGATIVE_SEEDS,
    }
    runner.run_study(
        controls,
        folder / "control-negative-seeds",
        study.main_hooks(),
        {"physics_steps": steps or PHYSICS_STEPS},
        workers=workers,
    )
    relaxed_design = {
        **study.RELAXED_DESIGN,
        "severities": [-0.75, 0.0, 2.0],
        "scaled_weights": [1.0],
        "noise_seeds": NEGATIVE_SEEDS,
    }
    runner.run_study(
        relaxed_design,
        folder / "relaxed-negative-seeds",
        study.relaxed_hooks(),
        {"physics_steps": steps or 2 * PHYSICS_STEPS},
        gate=False,
        workers=workers,
    )


def large_weight(steps, root, workers):
    """Controls at Lambda = 300 and 1000 with four times the budget."""
    design = {**runner.control_slice(study.FULL_DESIGN), "scaled_weights": [300.0, 1000.0]}
    runner.run_study(
        design,
        root / "oscillator/diagnostics/large-weight-budget-2026-10-25",
        study.main_hooks(),
        {"physics_steps": steps or 4 * PHYSICS_STEPS},
        workers=workers,
    )


if __name__ == "__main__":
    runner.run_parts(
        {"relaxed": relaxed, "damping-noise": damping_noise, "large-weight": large_weight},
        runner.study_arguments(__doc__),
    )
