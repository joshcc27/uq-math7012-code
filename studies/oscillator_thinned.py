"""The oscillator's thinned-collocation test.

Writes under the results root:

    thinned   oscillator/diagnostics/collocation-thinned-2026-10-11/n<N>
              11, 21, 41 and 201 equally spaced collocation times at Lambda = 30

The archived runs are ``cells/*.json`` from the original one-off script; a rerun
writes ``n<N>/study.csv`` with the same run ids, one folder per density because the
equation-fit target depends on the points.

    uv run python studies/oscillator_thinned.py --workers 6
"""

from pinn_misspec import runner
from pinn_misspec.oscillator import study
from pinn_misspec.oscillator.model import PHYSICS_STEPS

THINNED_COUNTS = [11, 21, 41, 201]


def thinned(steps, root, workers):
    """Not gated: the test asks whether the controls fail where enforcement is lost."""
    design = {**study.FULL_DESIGN, "severities": [0.0, 2.0], "scaled_weights": [30.0]}
    for count in THINNED_COUNTS:
        runner.run_study(
            design,
            root / f"oscillator/diagnostics/collocation-thinned-2026-10-11/n{count}",
            study.main_hooks(),
            {"physics_steps": steps or PHYSICS_STEPS, "n_col": count},
            gate=False,
            workers=workers,
            save_networks=True,
        )


if __name__ == "__main__":
    runner.run_parts({"thinned": thinned}, runner.study_arguments(__doc__))
