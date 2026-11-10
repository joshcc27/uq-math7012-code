"""Regenerate the thesis's tables, figures and diagnostic numbers from the saved results.

    uv run python -m pinn_misspec.report tables        # table rows, as LaTeX
    uv run python -m pinn_misspec.report numbers       # the other quoted values
    uv run python -m pinn_misspec.report intervals     # cluster-bootstrap intervals
    uv run python -m pinn_misspec.report checks        # the budget-check counts
    uv run python -m pinn_misspec.report screens       # the control screens
    uv run python -m pinn_misspec.report figures       # the thesis figures
    uv run python -m pinn_misspec.report diagnostic collocation-nested-2026-10-11
    uv run python -m pinn_misspec.report diagnostic --list

Results are read from ``results/`` or ``PINN_MISSPEC_RESULTS``; figures go to
``results/thesis_figures/`` unless ``--out`` names another folder. Nothing here trains
a network.
"""

import argparse
import sys
from pathlib import Path

from pinn_misspec.paths import THESIS_FIGURES

from . import checks, diagnostics, frame, illustrations, tables


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m pinn_misspec.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tables", "numbers", "intervals", "checks", "screens"):
        commands.add_parser(name)
    figures = commands.add_parser("figures")
    figures.add_argument("--out", type=Path, default=THESIS_FIGURES)
    diagnostic = commands.add_parser("diagnostic")
    diagnostic.add_argument("name", nargs="?", choices=sorted(diagnostics.DIAGNOSTICS))
    diagnostic.add_argument("--list", action="store_true", help="list the diagnostics")
    diagnostic.add_argument(
        "--folder", type=Path, help="analyse this folder instead, such as a rerun"
    )
    diagnostic.add_argument(
        "--monte-carlo",
        type=int,
        metavar="SEEDS",
        help="relaxed-damping-noise only: refit the noisy solution fits of this many seeds first",
    )
    arguments = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if arguments.command == "tables":
        tables.tables_part()
    elif arguments.command == "numbers":
        tables.numbers_part()
    elif arguments.command == "intervals":
        tables.intervals_part()
    elif arguments.command == "checks":
        checks.budget_counts()
    elif arguments.command == "screens":
        checks.screens()
    elif arguments.command == "figures":
        arguments.out.mkdir(parents=True, exist_ok=True)
        written = illustrations.illustrations(arguments.out)
        written += frame.coefficient_figures(arguments.out)
        written += checks.relaxed_figures(arguments.out)
        written += diagnostics.position_figure(arguments.out)
        print(f"wrote {len(written)} figures to {arguments.out}: {', '.join(written)}")
    elif arguments.list or arguments.name is None:
        for name, (_, function) in sorted(diagnostics.DIAGNOSTICS.items()):
            print(f"{name:45s} {function.__doc__.splitlines()[0]}")
    else:
        parent, function = diagnostics.DIAGNOSTICS[arguments.name]
        folder = arguments.folder or parent / arguments.name
        if arguments.monte_carlo:
            diagnostics.solution_fit_monte_carlo(arguments.monte_carlo, folder / "solfit_mc.csv")
        function(folder)


if __name__ == "__main__":
    main()
