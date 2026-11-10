"""Each PINN estimate against its two targets, and the coefficient-against-weight figures.

The position ``p = (theta_hat - theta_eq) / (theta_sol - theta_eq)`` is 0 at the
equation-fit target and 1 at the solution-fit target, for ``s != 0``. The error splits
exactly into three brackets over the truth:

    theta_hat - theta_true = [theta_sol - theta_true]    imposed-model shift
                           + [theta_hat_sol - theta_sol] observation noise
                           + [theta_hat - theta_hat_sol] PINN-specific part

A run is off the range when its error lies more than one range-width outside the span
of the truth and both targets. Position medians leave such runs out; coefficient
medians do not. Saved columns keep the archive's names: ``<c>_target`` is theta_sol and
``<c>_residual_target`` theta_eq.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from pinn_misspec.paths import ADVECTION_DIFFUSION, OSCILLATOR

# Okabe-Ito colours, shared with the illustrations.
PINN, SOLUTION_END, RESIDUAL_END = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
# Second weight of the relaxed checks (Okabe-Ito reddish purple); SOLUTION_END
# is kept for the solution-fit target.
SECOND_WEIGHT = "#CC79A7"
# The thesis's text width (455.24 pt). Figures are drawn at their printed width.
TEXT_WIDTH = 455.24 / 72


def figure_size(fraction, height):
    return (fraction * TEXT_WIDTH, height)


# The coefficients of each study: (column, label, symbol, truth column, off-range floor).
COEFFICIENTS = {
    "oscillator": [("w0", "frequency", r"\omega", "w0_true", 0.01)],
    "advdiff": [
        ("velocity", "velocity", "a", "velocity_true", 0.02),
        ("diffusivity", "diffusivity", r"\kappa", "diffusivity_true", 0.10),
    ],
}
FLOOR = {c[0]: c[4] for coefficients in COEFFICIENTS.values() for c in coefficients}
# Ends closer than this fraction of the truth leave the position undefined.
POSITION_FLOOR = 1e-6
OFF_RANGE_FLOOR = 0.01
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261008


def oscillator_study():
    return OSCILLATOR / "study"


def advdiff_study():
    return ADVECTION_DIFFUSION / "study"


def oscillator_lambda100_sources():
    """The two sources of the oscillator's Lambda = 100 column at 18000 updates.

    Network seed 0 comes from the budget check and seeds 1 and 2 from their own run.
    """
    return (
        OSCILLATOR / "study" / "steps-18000" / "study.csv",
        OSCILLATOR / "diagnostics" / "lambda100-18000-seeds-1-2-2026-10-11" / "study.csv",
    )


def read_csv(path):
    """Read a saved table with exact float round-tripping."""
    return pd.read_csv(path, dtype={"run_id": str}, float_precision="round_trip")


def decompose(frame, coefficient, truth_column, floor=OFF_RANGE_FLOOR):
    """Return a copy of ``frame`` with the brackets, position and off-range flag added.

    New columns, for coefficient ``c`` and as fractions of the truth:
    ``c_total``, ``c_shift``, ``c_noise``, ``c_pinn``, ``c_residual_error``,
    ``c_position``, ``c_range_low``, ``c_range_high`` and ``c_off_range``.
    """
    frame = frame.copy()
    c = coefficient
    truth = frame[truth_column]
    estimate, target = frame[f"{c}_final"], frame[f"{c}_target"]
    solution, residual = frame[f"{c}_sol"], frame[f"{c}_residual_target"]
    frame[f"{c}_total"] = (estimate - truth) / truth
    frame[f"{c}_shift"] = (target - truth) / truth
    frame[f"{c}_noise"] = (solution - target) / truth
    frame[f"{c}_pinn"] = (estimate - solution) / truth
    frame[f"{c}_residual_error"] = (residual - truth) / truth
    span = target - residual
    defined = (frame.severity != 0) & (span.abs() > POSITION_FLOOR * truth.abs())
    frame[f"{c}_position"] = np.where(defined, (estimate - residual) / span, np.nan)
    ends = np.column_stack(
        [np.zeros(len(frame)), frame[f"{c}_shift"], frame[f"{c}_residual_error"]]
    )
    low, high = ends.min(axis=1), ends.max(axis=1)
    width = np.maximum(high - low, floor)
    frame[f"{c}_range_low"], frame[f"{c}_range_high"] = low - width, high + width
    frame[f"{c}_off_range"] = (frame[f"{c}_total"] < frame[f"{c}_range_low"]) | (
        frame[f"{c}_total"] > frame[f"{c}_range_high"]
    )
    return frame


def bootstrap_median(cell, column, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Return the 2.5% and 97.5% points of a cluster bootstrap of the median.

    Noise seeds are resampled, each bringing all its runs, because runs that
    share observations are not independent.
    """
    groups = [group[column].to_numpy(dtype=float) for _, group in cell.groupby("noise_seed")]
    groups = [values[np.isfinite(values)] for values in groups]
    groups = [values for values in groups if len(values)]
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(draws, len(groups)))
    medians = np.array([np.median(np.concatenate([groups[i] for i in row])) for row in picks])
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def reference_errors(frame, coefficient, truth_column):
    """Return each severity's mean equation-fit and solution-fit target error, in percent."""
    c = coefficient
    grouped = frame.groupby("severity")
    residual = 100 * grouped.apply(
        lambda g: ((g[f"{c}_residual_target"] - g[truth_column]) / g[truth_column]).mean(),
        include_groups=False,
    )
    solution = 100 * grouped.apply(
        lambda g: ((g[f"{c}_target"] - g[truth_column]) / g[truth_column]).mean(),
        include_groups=False,
    )
    return {s: {"residual": residual[s], "solution": solution[s]} for s in residual.index}


def style():
    """Return the matplotlib settings of the thesis figures.

    Computer Modern matches the thesis's Latin Modern, with DejaVu Serif for any
    glyph that cmr10 lacks. Every figure is drawn at its printed size, so these
    type sizes are the printed ones.
    """
    return {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        # Named fonts, not the generic "serif", so that matplotlib falls back per glyph.
        "font.family": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "standard",
    }


def _median_line(ax, weights, values, unreliable, **line_style):
    """Draw a median line, dotting each segment that ends at an unreliable weight."""
    weights, values = np.asarray(weights, dtype=float), np.asarray(values, dtype=float)
    bad = np.isin(weights, unreliable)
    ax.plot(weights[~bad], values[~bad], "o-", **line_style)
    for i in np.flatnonzero(bad):
        if i > 0:
            ax.plot(
                weights[i - 1 : i + 1],
                values[i - 1 : i + 1],
                ls=":",
                marker="",
                **{k: v for k, v in line_style.items() if k not in ("ms", "mfc")},
            )
        ax.plot(weights[i], values[i], "o", **line_style)


# Above this many grid weights, only the powers of ten are labelled, so the labels stay apart.
MOST_WEIGHT_LABELS = 4


def plot_coefficient_panels(frame, references, coefficient, symbol, path, unreliable=()):
    """Plot one coefficient's error against the weight, one panel per severity.

    Runs are faint dots, the median over all runs a line, and the band spans the
    network seeds' medians. Runs beyond the axis are drawn as triangles at the
    edge. ``references`` maps each severity to its mean target errors in
    percent; segments into ``unreliable`` weights are dotted. The legend takes the
    first empty panel slot, or sits below the panels when there is none.
    """
    c = coefficient
    severities = sorted(frame.severity.unique())
    columns = 4 if len(severities) > 6 else 3 if len(severities) > 4 else len(severities)
    rows = int(np.ceil(len(severities) / columns))
    # One logarithmic axis in Lambda, with a tick at every grid weight.
    grid_weights = np.sort(frame.scaled_physics_weight.unique())
    labelled = grid_weights
    if len(grid_weights) > MOST_WEIGHT_LABELS:
        labelled = grid_weights[np.isclose(np.log10(grid_weights) % 1.0, 0.0)]
    tick_labels = [f"{w:g}" if w in labelled else "" for w in grid_weights]
    with plt.rc_context(style()):
        fig, axes = plt.subplots(
            rows, columns, figsize=figure_size(0.96, 2.1 * rows), squeeze=False, sharex=True
        )
        spare = list(axes.flat[len(severities) :])
        for ax in spare[1:]:
            ax.set_visible(False)
        is_panel = np.zeros(axes.shape, dtype=bool)
        is_panel.flat[: len(severities)] = True
        for ax, severity in zip(axes.flat, severities):
            everything = frame[frame.severity == severity]
            weights = np.sort(everything.scaled_physics_weight.unique())
            ax.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
            ref = references[severity]
            ax.axhline(ref["residual"], color=RESIDUAL_END, lw=1.4, ls="--", zorder=2)
            ax.axhline(ref["solution"], color=SOLUTION_END, lw=1.4, ls="-.", zorder=2)
            # Medians use every run, as the thesis tables do.
            by_network = (
                everything.groupby(["scaled_physics_weight", "network_seed"])[f"{c}_total"]
                .median()
                .unstack()
                * 100
            ).reindex(weights)
            median = (
                100 * everything.groupby("scaled_physics_weight")[f"{c}_total"].median()
            ).reindex(weights)
            # The axis spans the truth, both targets and every run, capped at the
            # off-range bounds so that distant runs, drawn as edge triangles, cannot
            # flatten the panel. The medians always stay in view.
            errors = 100 * everything[f"{c}_total"]
            anchors = np.concatenate([[0.0, ref["residual"], ref["solution"]], errors.to_numpy()])
            low = max(np.nanmin(anchors), 100 * everything[f"{c}_range_low"].min())
            high = min(np.nanmax(anchors), 100 * everything[f"{c}_range_high"].max())
            low, high = min(low, median.min()), max(high, median.max())
            pad = 0.08 * (high - low)
            low, high = low - pad, high + pad
            ax.set_ylim(low, high)
            cell = everything[(errors >= low) & (errors <= high)]
            outside = everything[(errors < low) | (errors > high)]
            ax.scatter(
                cell.scaled_physics_weight,
                100 * cell[f"{c}_total"],
                s=7,
                color=PINN,
                alpha=0.22,
                lw=0,
                zorder=3,
            )
            ax.fill_between(
                weights,
                by_network.min(axis=1),
                by_network.max(axis=1),
                color=PINN,
                alpha=0.18,
                lw=0,
                zorder=3,
            )
            _median_line(ax, weights, median, unreliable, color=PINN, ms=3.5, lw=1.5, zorder=4)
            if len(outside):
                edge = np.where(outside[f"{c}_total"] > 0, high, low)
                ax.scatter(
                    outside.scaled_physics_weight,
                    edge,
                    marker="v",
                    s=22,
                    facecolors="none",
                    edgecolors=PINN,
                    zorder=5,
                    clip_on=False,
                )
                ax.text(
                    0.02,
                    0.03,
                    f"{len(outside)} off axis",
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color=MUTED,
                )
            ax.set_xscale("log")
            ax.set_xticks(grid_weights, tick_labels)
            ax.minorticks_off()
            ax.set_title(f"$s = {severity:g}$")
            ax.grid(True, alpha=0.9)
            ax.set_axisbelow(True)
        # A panel with no visible panel beneath it carries the shared x axis.
        for row in range(rows):
            for column in range(columns):
                below = row + 1 < rows and is_panel[row + 1, column]
                if is_panel[row, column] and not below:
                    axes[row, column].tick_params(labelbottom=True)
                    axes[row, column].set_xlabel(r"physics weight $\Lambda$")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"${symbol}$ error (% of truth)")
        # In an empty panel slot the legend is narrow, so its longer labels wrap.
        wrap = "\n" if spare else " "
        handles = [
            plt.Line2D([], [], color=PINN, marker="o", ms=3.5, lw=1.5, label="PINN median"),
            Patch(facecolor=PINN, alpha=0.18, edgecolor="none", label=f"network-seed{wrap}medians"),
            plt.Line2D(
                [],
                [],
                color=RESIDUAL_END,
                lw=1.4,
                ls="--",
                label=f"equation-fit{wrap}target " + r"$\theta_{\mathrm{eq}}(s)$",
            ),
            plt.Line2D(
                [],
                [],
                color=SOLUTION_END,
                lw=1.4,
                ls="-.",
                label=f"solution-fit{wrap}target " + r"$\theta_{\mathrm{sol}}(s)$",
            ),
            plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (1, 2)), label="truth"),
        ]
        if spare:
            spare[0].axis("off")
            spare[0].legend(handles=handles, loc="center", handlelength=2.2)
        else:
            fig.legend(handles=handles, loc="outside lower center", ncol=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")
        plt.close(fig)


def coefficient_figures(out):
    """Write the frequency and diffusivity panel figures to ``out``.

    The oscillator's Lambda = 100 column is taken from the 18000-update runs of
    every network seed, because 9000 updates are too short there.
    """
    study = read_csv(oscillator_study() / "study.csv")
    frame = decompose(study, "w0", "w0_true", FLOOR["w0"])
    references = reference_errors(frame, "w0", "w0_true")
    longer = pd.concat(
        [read_csv(path) for path in oscillator_lambda100_sources()], ignore_index=True
    )
    longer = decompose(longer[longer.scaled_physics_weight == 100.0], "w0", "w0_true", FLOOR["w0"])
    reported = pd.concat([frame[frame.scaled_physics_weight != 100.0], longer], ignore_index=True)
    plot_coefficient_panels(reported, references, "w0", r"\omega", out / "osc-w0-vs-weight.pdf")

    study = read_csv(advdiff_study() / "study.csv")
    frame = decompose(study, "diffusivity", "diffusivity_true", FLOOR["diffusivity"])
    references = reference_errors(frame, "diffusivity", "diffusivity_true")
    plot_coefficient_panels(
        frame, references, "diffusivity", r"\kappa", out / "advdiff-diffusivity-vs-weight.pdf"
    )
    return ["osc-w0-vs-weight.pdf", "advdiff-diffusivity-vs-weight.pdf"]
