"""Figures drawn from saved networks: what the neural state does between the data.

    uv run python -m pinn_misspec.report network-figures --out <folder>

These need trained networks, which only two sets of runs saved: the advection--diffusion
full-grid density sweep (``dense-grid-2026-10-25``) and the oscillator residual-structure
diagnostic (``data-residual-structure-2026-10-25``), whose configurations are cells of the
main grid. The published results archive does not include the networks, so these
figures need the full local results. Nothing here trains a network.
"""

import matplotlib

matplotlib.use("Agg")
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from pinn_misspec import runner
from pinn_misspec.oscillator import model as osc_model
from pinn_misspec.oscillator import study as osc_study
from pinn_misspec.paths import ADVECTION_DIFFUSION, OSCILLATOR
from pinn_misspec.training import build_network

from . import frame as rf

DENSE_GRID = ADVECTION_DIFFUSION / "diagnostics" / "dense-grid-2026-10-25"
RESIDUAL_STRUCTURE = OSCILLATOR / "diagnostics" / "data-residual-structure-2026-10-25"

# One run per panel: noise seed 0 and network seed 0, at the strongest weight of the
# sweep, where the two densities differ most.
ADV_WEIGHT, ADV_SEVERITY, NOISE_SEED, NETWORK_SEED = 3.0, 1.0, 0, 0
OSC_SEVERITY, OSC_WEIGHTS = 2.0, (0.3, 30.0)

# Sequential, one hue light to dark, ending in the darkest severity shade.
RESIDUAL_MAP = LinearSegmentedColormap.from_list(
    "residual", ["#ffffff", "#a9cbe8", rf.PINN, "#013b73", "#0b1a33"]
)
RESIDUAL_NORM = LogNorm(1e-3, 10.0)
LIGHT_PINN, DARK_PINN = "#5090f1", "#013b73"


def _sweep_record(n_col, severity):
    """Return the saved arrays of one run of the full-grid sweep, and its row."""
    folder = DENSE_GRID / f"n{n_col}"
    rows = rf.read_csv(folder / "study.csv")
    row = rows[
        (rows.scaled_physics_weight == ADV_WEIGHT)
        & (rows.severity == severity)
        & (rows.noise_seed == NOISE_SEED)
        & (rows.network_seed == NETWORK_SEED)
    ]
    if len(row) != 1:
        raise LookupError(f"expected one run at {n_col} points and s = {severity}, got {len(row)}")
    row = row.iloc[0]
    return np.load(folder / "networks" / f"{row.run_id}.npz"), row


def _edges(centres, low, high):
    """Return cell edges for ``pcolormesh`` around grid points that start at ``low``."""
    middles = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[low], middles, [high]])


def residual_field_figure(out):
    """Write the residual of the conservative equation over the domain, at two densities.

    (a) The control and (b) s = 1 at 600 collocation points, and (c) s = 1 at 9600,
    with the collocation points of (a) and (b) shaded by their own residual. (d) The
    mass of the neural state at s = 1 at both densities.
    """
    panels = [
        ("(a) $s=0$, 600 points", *_sweep_record(600, 0.0), True),
        (f"(b) $s={ADV_SEVERITY:g}$, 600 points", *_sweep_record(600, ADV_SEVERITY), True),
        (f"(c) $s={ADV_SEVERITY:g}$, 9600 points", *_sweep_record(9600, ADV_SEVERITY), False),
    ]
    with plt.rc_context(rf.style()):
        fig, axes = plt.subplots(2, 2, figsize=rf.figure_size(0.96, 4.7))
        for ax, (title, record, row, show_points) in zip(axes.flat, panels):
            x, t = record["x"], record["t"]
            image = ax.pcolormesh(
                _edges(x, 0.0, 1.0),
                _edges(t, 0.0, float(t[-1])),
                np.abs(record["residual_dense"]),
                cmap=RESIDUAL_MAP,
                norm=RESIDUAL_NORM,
                shading="flat",
                rasterized=True,
            )
            if show_points:
                points = record["collocation"]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=np.abs(record["residual_collocation"]),
                    cmap=RESIDUAL_MAP,
                    norm=RESIDUAL_NORM,
                    s=5,
                    edgecolors=rf.INK,
                    linewidths=0.35,
                    zorder=3,
                )
            ax.set_title(title, loc="left")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$t$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, float(t[-1]))
            print(
                f"[check] {title}: run {row.run_id}, mean squared residual "
                f"{row.mse_collocation:.2e} at the points, {row.mse_dense:.2e} held out "
                f"(ratio {row.mse_dense / row.mse_collocation:.0f}), max |r| held out "
                f"{np.abs(record['residual_dense']).max():.2f}, diffusivity error "
                f"{100 * row.diffusivity_rel_error_true:+.1f}%"
            )
        colourbar = fig.colorbar(image, ax=axes[0, :].tolist(), extend="max", shrink=0.9)
        colourbar.set_label(r"$|r(x,t)|$")

        mass = axes[1, 1]
        sparse, dense = panels[1][1], panels[2][1]
        t = sparse["t"]
        mass.axhline(1.0, color=rf.SOLUTION_END, ls="--", lw=1.4, label="imposed")
        mass.plot(
            t,
            sparse["mass_true"] / sparse["mass_true"][0],
            color=rf.MUTED,
            ls=(0, (1, 2)),
            lw=1.4,
            label="generating",
        )
        mass.plot(t, sparse["mass_pinn"] / sparse["mass_pinn"][0], color=LIGHT_PINN, lw=1.5)
        mass.plot(t, dense["mass_pinn"] / dense["mass_pinn"][0], color=DARK_PINN, lw=1.5)
        # The two PINN states are labelled on the panel, the reference lines in the legend.
        mass.text(0.59, 0.91, "PINN, 9600 points", ha="right", va="top", fontsize=8, color=rf.INK)
        mass.text(0.59, 0.6, "PINN, 600 points", ha="right", va="bottom", fontsize=8, color=rf.INK)
        mass.legend(loc="lower left", handlelength=2.2)
        mass.set_title(f"(d) mass of the state, $s={ADV_SEVERITY:g}$", loc="left")
        mass.set_xlabel("$t$")
        mass.set_ylabel(r"$M(t)/M_0$")
        mass.set_xlim(0, float(t[-1]))
        mass.set_ylim(0.2, 1.06)
        mass.grid(True, alpha=0.9)
        mass.set_axisbelow(True)
        fig.savefig(out / "advdiff-residual-field.pdf", facecolor="white", dpi=300)
        plt.close(fig)

    ratio = sparse["mass_pinn"] / sparse["mass_pinn"][0]
    lost = 1.0 - ratio[-1]
    shed_by = float(t[np.argmax(1.0 - ratio >= 0.9 * lost)])
    print(
        f"[check] mass at t = T: 600 points {ratio[-1]:.3f}, 9600 points "
        f"{dense['mass_pinn'][-1] / dense['mass_pinn'][0]:.3f}, generating "
        f"{sparse['mass_true'][-1] / sparse['mass_true'][0]:.3f}; the 600-point state has "
        f"shed 90% of its loss by t = {shed_by:.2f}"
    )
    return ["advdiff-residual-field.pdf"]


def _oscillator_state(weight, t):
    """Return the saved run at ``OSC_SEVERITY`` and ``weight``: its row, state and data."""
    config = osc_study.study_config(OSC_SEVERITY, weight, NOISE_SEED, NETWORK_SEED)
    run_id = runner.run_id(config)
    rows = rf.read_csv(RESIDUAL_STRUCTURE / "study.csv")
    row = rows[rows.run_id == run_id].iloc[0]
    net = eqx.tree_deserialise_leaves(
        RESIDUAL_STRUCTURE / "networks" / f"{run_id}.eqx", build_network(config, in_size=1)
    )
    trial = osc_model.trial_spec(config)
    t_obs, y_obs, _ = osc_model.make_observations(config)
    # The reloaded network must reproduce the recorded fit to the data.
    data_loss = float(jnp.mean((osc_model.predict(net, t_obs, trial) - y_obs) ** 2))
    if not np.isclose(data_loss, row.data_loss, rtol=1e-9, atol=0.0):
        raise RuntimeError(f"run {run_id}: reloaded data loss {data_loss} != {row.data_loss}")
    state = np.asarray(osc_model.predict(net, jnp.asarray(t), trial))
    imposed = np.asarray(
        osc_model.exact_solution(t, row.w0_final, osc_model.imposed_damping(config))
    )
    generating = np.asarray(osc_model.exact_solution(t, config["w0_true"], config["g_true"]))
    return row, state, imposed, generating, np.asarray(t_obs), np.asarray(y_obs)


def oscillator_state_figure(out):
    """Write the oscillator's neural state at a weak and a strong weight, at s = 2.

    Each panel shows the observations, the generating state, the neural state, and the
    exact imposed solution at the network's own frequency. The distance between the last
    two is the imposed-solution gap.
    """
    t = np.linspace(0.0, 3.0, 1201)
    with plt.rc_context(rf.style()):
        fig, axes = plt.subplots(1, 2, figsize=rf.figure_size(0.96, 2.9), sharey=True)
        for ax, weight, letter in zip(axes, OSC_WEIGHTS, "ab"):
            row, state, imposed, generating, t_obs, y_obs = _oscillator_state(weight, t)
            ax.plot(
                t_obs,
                y_obs,
                "o",
                color=rf.MUTED,
                mfc="white",
                mew=0.6,
                ms=2.2,
                zorder=2,
                label="observations",
            )
            ax.plot(
                t, generating, color=rf.MUTED, ls=(0, (1, 2)), lw=1.4, zorder=3, label="generating"
            )
            ax.plot(
                t,
                imposed,
                color=rf.SOLUTION_END,
                ls="--",
                lw=1.4,
                zorder=4,
                label=r"imposed solution at the fitted $\omega$",
            )
            ax.plot(t, state, color=rf.PINN, lw=1.5, zorder=5, label="neural state")
            ax.set_title(f"({letter}) $\\Lambda={weight:g}$", loc="left")
            ax.set_xlabel("$t$")
            ax.set_xlim(0, 3)
            ax.grid(True, alpha=0.9)
            ax.set_axisbelow(True)
            print(
                f"[check] oscillator s = {OSC_SEVERITY:g}, Lambda = {weight:g}: run {row.run_id}, "
                f"frequency error {100 * row.w0_rel_error_true:+.2f}%, imposed-solution gap "
                f"{100 * row.imposed_gap_rel_l2:.1f}%, state error "
                f"{100 * row.state_rel_l2_pinn:.1f}%, data loss {row.data_loss:.2e}"
            )
        axes[0].set_ylabel("$u(t)$")
        handles, labels = axes[0].get_legend_handles_labels()
        order = [3, 2, 1, 0]
        fig.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="outside lower center",
            ncol=4,
            handlelength=2.2,
            columnspacing=1.2,
        )
        fig.savefig(out / "osc-neural-state.pdf", facecolor="white")
        plt.close(fig)
    return ["osc-neural-state.pdf"]


def network_figures(out):
    """Write the two figures that need saved networks to ``out``; return their names."""
    out.mkdir(parents=True, exist_ok=True)
    return residual_field_figure(out) + oscillator_state_figure(out)
