"""The relaxed and budget checks: their figures and the budget-check counts.

The relaxed figures take the strong weight from the longer reruns. A budget-check run
is budget-limited when its estimate moves by more than the larger of its cell's
network-seed and noise-seed spreads at the standard budget.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_misspec.paths import ADVECTION_DIFFUSION, OSCILLATOR

from . import frame as rf
from .diagnostics import _control_failures

CHECKS = {
    "oscillator": {
        "study": rf.oscillator_study,
        "prefix": "osc",
        "budget_steps": 18000,
        "standard_steps": 9000,
        "relaxed_long_weights": (100.0,),
        "coefficients": [("w0", "w0_true", r"\omega")],
        "relaxed_extra": [("g", "g_true", r"\gamma")],
        "severity_column": "init_severity",
        "main_label": "fixed",
        "width": 0.8,  # of the text width, as printed
    },
    "advdiff": {
        "study": rf.advdiff_study,
        "prefix": "advdiff",
        "budget_steps": 10000,
        "standard_steps": 5000,
        "relaxed_long_weights": (0.3,),
        "coefficients": [
            ("velocity", "velocity_true", "a"),
            ("diffusivity", "diffusivity_true", r"\kappa"),
        ],
        "relaxed_extra": [],
        "severity_column": "severity",
        "main_label": "omitted",
        "width": 0.96,  # of the text width, as printed
    },
}
CELL = ["severity", "scaled_physics_weight", "noise_seed"]
# The two weights of the relaxed check, told apart by colour and by marker.
WEIGHT_STYLES = [(rf.PINN, "o"), (rf.SECOND_WEIGHT, "s")]
TITLES = {"w0": "frequency", "g": "damping", "velocity": "velocity", "diffusivity": "diffusivity"}


def relaxed_summary(spec, relaxed, main):
    """Return mean errors per ``(severity, weight)``, relaxed beside main (network seed 0)."""
    column = spec["severity_column"]
    main = main[main.network_seed == 0]
    rows = []
    for (severity, weight), cell in relaxed.groupby([column, "scaled_physics_weight"]):
        row = {
            "severity": severity,
            "scaled_physics_weight": weight,
            "runs": len(cell),
            "physics_steps": int(cell.physics_steps.iloc[0]),
        }
        matched = main[(main.severity == severity) & (main.scaled_physics_weight == weight)]
        for name, truth, _ in spec["coefficients"] + spec["relaxed_extra"]:
            error = 100 * (cell[f"{name}_final"] - cell[truth]) / cell[truth]
            row[f"{name}_relaxed_pct"] = error.mean()
            row[f"{name}_relaxed_max_abs_pct"] = error.abs().max()
            if (
                len(matched)
                and f"{name}_final" in matched
                and name in [c[0] for c in spec["coefficients"]]
            ):
                row[f"{name}_main_pct"] = (
                    100 * (matched[f"{name}_final"] - matched[truth]) / matched[truth]
                ).mean()
        if "sink_final" in cell:
            row["sink_scaled_relaxed"] = (cell.sink_final * cell.t_end).mean()
            row["sink_scaled_error_max"] = cell.sink_scaled_error_true.max()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_relaxed(spec, summary, path):
    """Plot each coefficient's error against severity, freed and fixed."""
    coefficients = spec["coefficients"] + spec["relaxed_extra"]
    sink = "sink_scaled_relaxed" in summary
    panels = len(coefficients) + int(sink)
    weights = sorted(summary.scaled_physics_weight.unique())
    with plt.rc_context(rf.style()):
        fig, axes = plt.subplots(
            1, panels, figsize=rf.figure_size(spec["width"], 2.7), squeeze=False
        )
        letters = "abc"

        def tag(weight):
            # Name the budget only where it differs from the standard one.
            steps = int(summary[summary.scaled_physics_weight == weight].physics_steps.iloc[0])
            budget = "" if steps == spec["standard_steps"] else f", {steps} updates"
            return rf"$\Lambda = {weight:g}${budget}"

        for index, (ax, (name, _, symbol)) in enumerate(zip(axes[0], coefficients)):
            ax.set_title(f"({letters[index]}) {TITLES[name]}", loc="left")
            ax.axhline(0.0, color=rf.MUTED, lw=0.8, ls=(0, (1, 2)))
            for (colour, marker), weight in zip(WEIGHT_STYLES, weights):
                part = summary[summary.scaled_physics_weight == weight].sort_values("severity")
                ax.plot(
                    part.severity,
                    part[f"{name}_relaxed_pct"],
                    marker=marker,
                    ms=4,
                    color=colour,
                    lw=1.5,
                    label=f"freed, {tag(weight)}",
                )
                if f"{name}_main_pct" in part:
                    ax.plot(
                        part.severity,
                        part[f"{name}_main_pct"],
                        marker=marker,
                        ms=4,
                        mfc="white",
                        color=colour,
                        lw=1.0,
                        ls="--",
                        label=f"{spec['main_label']}, {tag(weight)}",
                    )
            ax.set_xlabel("severity $s$")
            ax.set_ylabel(f"${symbol}$ error (% of truth)")
            ax.grid(True, alpha=0.9)
            ax.set_axisbelow(True)
        if sink:
            ax = axes[0, -1]
            ax.set_title(f"({letters[panels - 1]}) sink", loc="left")
            severities = np.sort(summary.severity.unique())
            ax.plot(severities, severities, color=rf.MUTED, lw=0.9, ls=(0, (1, 2)), label="truth")
            for (colour, marker), weight in zip(WEIGHT_STYLES, weights):
                part = summary[summary.scaled_physics_weight == weight].sort_values("severity")
                ax.plot(
                    part.severity,
                    part.sink_scaled_relaxed,
                    marker=marker,
                    ms=4,
                    color=colour,
                    lw=1.5,
                )
            ax.set_xlabel("severity $s$")
            ax.set_ylabel(r"fitted sink $\widehat\mu T$")
            ax.grid(True, alpha=0.9)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        # Labels that name a budget are long, so they take two rows.
        two_rows = any("updates" in label for label in labels)
        fig.legend(handles, labels, loc="outside lower center", ncol=2 if two_rows else 4)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")
        plt.close(fig)


def budget_comparison(spec, main, longer):
    """Compare each network-seed-0 run at twice the budget with its standard run."""
    standard = main[main.network_seed == 0].set_index(CELL)
    longer = longer[longer.network_seed == 0].set_index(CELL)
    spread = main.groupby(CELL)
    noise_spread = main[main.network_seed == 0].groupby(["severity", "scaled_physics_weight"])
    common = standard.index.intersection(longer.index)
    table = pd.DataFrame(index=common)
    limited = np.zeros(len(common), dtype=bool)
    for name, truth, _ in spec["coefficients"]:
        truth_values = standard.loc[common, truth]
        change = longer.loc[common, f"{name}_final"] - standard.loc[common, f"{name}_final"]
        table[f"{name}_change"] = change / truth_values
        seeds = spread[f"{name}_final"].agg(lambda v: v.max() - v.min()).loc[common]
        table[f"{name}_network_spread"] = seeds / truth_values
        by_noise = noise_spread[f"{name}_final"].agg(lambda v: v.max() - v.min())
        noise = by_noise.reindex(common.droplevel("noise_seed")).to_numpy()
        table[f"{name}_noise_spread"] = noise / truth_values.to_numpy()
        resolution = np.maximum(table[f"{name}_network_spread"], table[f"{name}_noise_spread"])
        table[f"{name}_budget_limited"] = (table[f"{name}_change"].abs() > resolution).to_numpy()
        limited |= table[f"{name}_budget_limited"].to_numpy()
    table["budget_limited"] = limited
    return table.reset_index()


def relaxed_figures(out):
    """Write the two relaxed-check figures to ``out``; return their file names."""
    names = []
    for spec in CHECKS.values():
        study = spec["study"]()
        main = rf.read_csv(study / "study.csv")
        relaxed = rf.read_csv(study / "relaxed" / "study.csv")
        longer = rf.read_csv(study / "relaxed" / f"steps-{spec['budget_steps']}" / "study.csv")
        budget = rf.read_csv(study / f"steps-{spec['budget_steps']}" / "study.csv")
        # Take the strong weight from the longer relaxed runs and compare them
        # with the budget check's runs, which share their budget and seed.
        strong = list(spec["relaxed_long_weights"])
        relaxed = pd.concat(
            [
                relaxed[~relaxed.scaled_physics_weight.isin(strong)],
                longer[longer.scaled_physics_weight.isin(strong)],
            ]
        )
        matched = pd.concat(
            [
                main[~main.scaled_physics_weight.isin(strong)],
                budget[budget.scaled_physics_weight.isin(strong)],
            ]
        )
        name = f"{spec['prefix']}-relaxed.pdf"
        plot_relaxed(spec, relaxed_summary(spec, relaxed, matched), out / name)
        names.append(name)
    return names


def budget_counts():
    """Print how many runs of each budget check are still moving at twice the budget."""
    for system, spec in CHECKS.items():
        study = spec["study"]()
        comparison = budget_comparison(
            spec,
            rf.read_csv(study / "study.csv"),
            rf.read_csv(study / f"steps-{spec['budget_steps']}" / "study.csv"),
        )
        print(
            f"{system}: budget check, {len(comparison)} cells, "
            f"{int(comparison.budget_limited.sum())} budget-limited"
        )


def screens():
    """Print the control screens: the controls passing their acceptance limits, by weight."""
    frame = rf.read_csv(OSCILLATOR / "protocol" / "weight-screen-2026-10-06" / "study.csv")
    frame["passed"] = [not f for f in _control_failures(frame.astype(str).to_dict("records"))]
    counts = frame.groupby("scaled_physics_weight").passed.agg(["sum", "count"])
    print("Table B.1, oscillator controls passing at 9000 updates, by weight:")
    print("  " + "  ".join(f"{w:g}: {s}/{n}" for w, (s, n) in counts.iterrows()))
    large = frame[frame.scaled_physics_weight >= 300]
    print(
        f"  from Lambda = 300 the controls settle at omega from {large.w0_final.min():.2f} "
        f"to {large.w0_final.max():.2f}"
    )
    print("Table B.2, advection--diffusion controls passing at 5000 updates:")
    for label, rate in (("0p003", 0.003), ("0p001", 0.001), ("0p0005", 0.0005)):
        path = ADVECTION_DIFFUSION / "protocol" / "rate-screen-2026-10-06" / f"theta-{label}"
        frame = rf.read_csv(path / "study.csv")
        frame["passed"] = [not f for f in _control_failures(frame.astype(str).to_dict("records"))]
        weak = frame[frame.scaled_physics_weight <= 0.3]
        strong = frame[frame.scaled_physics_weight >= 1.0]
        print(
            f"  rate {rate:g}: Lambda 0.01 to 0.3 {int(weak.passed.sum())} of {len(weak)}; "
            f"Lambda 1, 3 and 10 {int(strong.passed.sum())} of {len(strong)}"
        )
