"""Analyses of the diagnostic runs cited in the thesis.

Each prints its measures and verdict, without training a network. The archived
collocation runs keep the format of the one-off scripts that made them
(``cells/*.json``, or ``summary.csv`` and ``state-*.npz``); reruns through
``studies/`` write ``study.csv``, and both are read here. Columns that only archived
runs have are shown only when present.
"""

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_misspec import runner
from pinn_misspec.advection_diffusion import study as adv_study
from pinn_misspec.oscillator import study as osc_study
from pinn_misspec.oscillator.targets import fit_relaxed, fit_solution
from pinn_misspec.paths import ADVECTION_DIFFUSION, OSCILLATOR

from . import frame as rf

OSC_N_COL, ADV_N_COL = 201, 600  # the studies' standard collocation counts
# Study-row columns under the names the original cell records used.
CELL_NAMES = {
    "w0_final": "w0",
    "velocity_final": "velocity",
    "diffusivity_final": "diffusivity",
    "state_rel_l2_pinn": "state_rel_l2",
    "imposed_gap_rel_l2": "gap_rel_l2",
}


def _control_failures(rows):
    """Recompute each row's control failures against its study's control limits."""
    failures = []
    for row in rows:
        if float(row["severity"]) != 0.0:
            failures.append([])
        elif "w0_final" in row:
            failures.append(
                runner.control_row_failures(row, osc_study.CONTROL_LIMITS, osc_study.DOMAIN_FLAG)
            )
        else:
            failures.append(
                runner.control_row_failures(row, adv_study.CONTROL_LIMITS, adv_study.DOMAIN_FLAG)
            )
    return failures


def _study_rows(folder):
    """Return the rows of ``study.csv`` in ``folder`` and its immediate subfolders."""
    paths = sorted([*folder.glob("study.csv"), *folder.glob("*/study.csv")])
    if not paths:
        raise FileNotFoundError(f"no cells/ and no study.csv under {folder}")
    return pd.concat([rf.read_csv(path) for path in paths], ignore_index=True)


def cells(folder):
    """Return one record per run: the archived ``cells/`` files, or a rerun's study rows.

    Study rows are renamed to the cell fields and their control failures computed.
    """
    folder = Path(folder)
    if (folder / "cells").exists():
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (folder / "cells").glob("cell-*.json")
        ]
        return pd.DataFrame(records)
    rows = _study_rows(folder)
    frame = rows.rename(columns=CELL_NAMES)
    frame["control_failures"] = _control_failures(rows.astype(str).to_dict("records"))
    frame["reproduced_abs_diff"] = np.nan
    return frame


def pct(value, truth):
    return 100.0 * (value / truth - 1.0)


def present(frame, columns):
    """Keep the columns that ``frame`` has; archived runs have a few more than new ones."""
    return [column for column in columns if column in frame]


def with_diffusivity_position(frame):
    frame["diffusivity_pct"] = 100 * (frame.diffusivity / 0.01 - 1)
    span = frame.diffusivity_target - frame.diffusivity_residual_target
    frame["position"] = np.where(
        frame.severity > 0, (frame.diffusivity - frame.diffusivity_residual_target) / span, np.nan
    )
    return frame


def with_frequency_position(frame):
    frame["w0_pct"] = 100 * (frame.w0 / frame.w0_true - 1)
    span = frame.w0_target - frame.w0_residual_target
    frame["position"] = np.where(
        frame.severity > 0, (frame.w0 - frame.w0_residual_target) / span, np.nan
    )
    frame["ratio_R"] = frame.mse_dense / frame.mse_collocation
    return frame


def failing_controls(frame):
    controls = frame[frame.severity == 0.0]
    return controls, controls[controls.control_failures.map(len) > 0]


def density_frame(folder):
    """Return the first density test's runs with their diffusivity position."""
    frame = with_diffusivity_position(cells(folder))
    return frame.sort_values(["severity", "scaled_physics_weight", "noise_seed", "n_col"])


def density(folder):
    """Does denser collocation move the diffusivity towards theta_sol?"""
    frame = density_frame(folder)
    pd.set_option("display.width", 250)
    columns = present(
        frame,
        [
            "severity",
            "scaled_physics_weight",
            "noise_seed",
            "n_col",
            "diffusivity_pct",
            "position",
            "mse_collocation",
            "mse_dense",
            "data_loss",
            "gap_rel_l2",
            "mass_ratio_pinn",
            "speed_state",
            "velocity",
            "verdict",
        ],
    )
    print(frame[columns].round(4).to_string(index=False))
    controls, failing = failing_controls(frame)
    print(f"\ncontrols failing the frozen limits: {len(failing)} of {len(controls)}")
    for row in failing.itertuples():
        print(
            f"  n_col={row.n_col} Lambda={row.scaled_physics_weight} seed={row.noise_seed}: "
            f"{'; '.join(row.control_failures)}"
        )
    reproduced = frame[frame.n_col == ADV_N_COL].reproduced_abs_diff
    print(
        f"600-point reruns: largest coefficient difference from the record {reproduced.max():.3g}"
    )
    flagged = {(r.n_col, r.scaled_physics_weight) for r in failing.itertuples()}
    if flagged:
        print(f"flagged (failing control) densities and weights: {sorted(flagged)}")

    pairs = frame[frame.severity > 0].pivot_table(
        index=["severity", "scaled_physics_weight", "noise_seed"],
        columns="n_col",
        values=["mse_dense", "position"],
    )
    dense = pairs["mse_dense"]
    monotone = int(((dense[600] > dense[2400]) & (dense[2400] > dense[9600])).sum())
    fall = float((dense[600] / dense[9600]).median())
    rise = float((pairs["position"][9600] - pairs["position"][600]).median())
    print(
        f"\nheld-out residual falls monotonically in {monotone} of {len(dense)} pairs; "
        f"median 600/9600 ratio {fall:.2f}; median position rise {rise:+.3f}"
    )
    if monotone >= 7 and fall >= 10 and rise >= 0.25:
        verdict = "SUPPORTED"
    elif fall < 2 or rise < 0.05:
        verdict = "CONTRADICTED"
    else:
        verdict = "PARTIAL"
    print(f"verdict by the recorded rule: {verdict}")


def nested(folder):
    """Does nested, denser collocation reach theta_sol for every network seed?"""
    densities, planned = (9600, 2400, 600), 54
    frame = with_diffusivity_position(cells(folder))
    keys = ["severity", "noise_seed", "network_seed", "n_col"]
    pd.set_option("display.width", 250)
    columns = present(
        frame,
        keys
        + [
            "diffusivity_pct",
            "position",
            "mse_collocation",
            "mse_dense",
            "gap_rel_l2",
            "mass_ratio_pinn",
            "speed_state",
            "velocity",
            "verdict",
        ],
    )
    print(frame.sort_values(keys)[columns].round(4).to_string(index=False))
    print(f"\ncells recorded: {len(frame)} of {planned}")

    controls, failing = failing_controls(frame)
    print(f"controls failing the frozen limits: {len(failing)} of {len(controls)}")
    for row in failing.itertuples():
        print(
            f"  n_col={row.n_col} net={row.network_seed} noise={row.noise_seed}: "
            f"{'; '.join(row.control_failures)}"
        )
    flagged = {(r.n_col, r.network_seed) for r in failing.itertuples()}

    series = frame[frame.severity > 0].pivot_table(
        index=["severity", "noise_seed", "network_seed"],
        columns="n_col",
        values=["mse_dense", "position"],
    )
    usable = [i for i in series.index if not any((n, i[2]) in flagged for n in densities)]
    series = series.loc[usable]
    dense, position = series["mse_dense"], series["position"]
    monotone = int(((dense[600] > dense[2400]) & (dense[2400] > dense[9600])).sum())
    fall = float((dense[600] / dense[9600]).median())
    rise = position[9600] - position[600]
    by_seed = rise.groupby(level="network_seed").median()
    print(
        f"\nseries used: {len(series)} of 12; held-out residual falls at each step in {monotone}; "
        f"median 600/9600 fall {fall:.1f}x; median position rise {rise.median():+.3f}"
    )
    print("median position rise by network seed:", by_seed.round(3).to_dict())
    print(
        "median position at 600 / 2400 / 9600:",
        [round(float(position[n].median()), 3) for n in (600, 2400, 9600)],
    )
    if monotone >= 10 and fall >= 10 and rise.median() >= 0.25 and (by_seed >= 0.25).all():
        verdict = "SUPPORTED"
    elif fall < 2 or rise.median() < 0.05 or (by_seed < 0.05).any():
        verdict = "CONTRADICTED"
    else:
        verdict = "PARTIAL"
    print(f"verdict by the recorded rule: {verdict}")

    # Medians over the six runs of each cell. The ratio is the median of the
    # run-level quotients, so it need not equal the quotient of the medians.
    frame["ratio"] = frame.mse_dense / frame.mse_collocation
    table = frame.groupby(["severity", "n_col"]).agg(
        data_loss=("data_loss", "median"),
        residual_at_points=("mse_collocation", "median"),
        held_out_residual=("mse_dense", "median"),
        ratio=("ratio", "median"),
        gap=("gap_rel_l2", "median"),
        runs=("ratio", "size"),
    )
    print("\nTable 6.5, medians over the runs of each cell:")
    print(table.to_string(float_format=lambda v: f"{v:.3g}"))


def thinned(folder):
    """Does sparse collocation stop the oscillator reaching theta_sol?"""
    planned = 120
    frame = with_frequency_position(cells(folder))
    pd.set_option("display.width", 250)
    print(f"cells recorded: {len(frame)} of {planned}")
    reproduced = frame[frame.n_col == OSC_N_COL].reproduced_abs_diff.dropna()
    print(
        f"201-point reruns: {len(reproduced)} matched to the main grid, "
        f"largest frequency difference {reproduced.max():.3g}"
    )
    by_density = frame.groupby(["severity", "n_col"]).agg(
        runs=("run_id", "size"),
        w0_pct=("w0_pct", "median"),
        position=("position", "median"),
        ratio_R=("ratio_R", "median"),
        gap=("gap_rel_l2", "median"),
        res_target_pct=("w0_residual_target", lambda v: 100 * (v.median() / 2 / np.pi - 1)),
    )
    print(by_density.round(4).to_string())
    controls, failing = failing_controls(frame)
    print(f"\ncontrols failing the frozen limits: {len(failing)} of {len(controls)}")
    for n_col, part in failing.groupby("n_col"):
        print(
            f"  n_col={n_col}: {len(part)} failing, e.g. {'; '.join(part.control_failures.iloc[0])}"
        )
    flagged = set(failing.n_col)
    misspecified = by_density.loc[2.0]
    if 11 in flagged:
        print("verdict by the recorded rule: FLAGGED (the 11-point controls fail)")
        return
    drop = float(misspecified.position[201] - misspecified.position[11])
    ratio = float(misspecified.ratio_R[11])
    print(
        f"\nposition at 201 vs 11 points: {misspecified.position[201]:.3f} vs "
        f"{misspecified.position[11]:.3f} (drop {drop:+.3f}); median R at 11 points {ratio:.3g}"
    )
    if drop >= 0.25 and ratio >= 10:
        verdict = "COVERAGE EFFECT PRESENT"
    elif abs(drop) <= 0.1 and ratio < 3:
        verdict = "COVERAGE EFFECT ABSENT"
    else:
        verdict = "PARTIAL"
    print(f"verdict by the recorded rule: {verdict}")


def residual_summary(folder):
    """Return the residual diagnostic's table: the archived ``summary.csv``, or a rerun's rows.

    A rerun through ``studies/advdiff_collocation.py`` has one subfolder per
    group, holding study rows with the saved-network entries.
    """
    folder = Path(folder)
    if (folder / "summary.csv").exists():
        return rf.read_csv(folder / "summary.csv")
    rows = []
    for row in _study_rows(folder).to_dict("records"):
        oscillator = "w0_final" in row and not pd.isna(row.get("w0_final"))
        rows.append(
            {
                "model": "oscillator" if oscillator else "advection-diffusion",
                "severity": row["severity"],
                "scaled_physics_weight": row["scaled_physics_weight"],
                "physics_steps": row["physics_steps"],
                "run_id": row["run_id"],
                "mse_collocation": row["mse_collocation"],
                "mse_dense": row["mse_dense"],
                "ratio_R": row["mse_dense"] / row["mse_collocation"],
                "velocity_fitted": np.nan if oscillator else row["velocity_final"],
                "diffusivity_fitted": np.nan if oscillator else row["diffusivity_final"],
                "speed_state": np.nan if oscillator else row["speed_state"],
                "speed_true": np.nan if oscillator else row["speed_true"],
                "speed_imposed": np.nan if oscillator else row["speed_imposed"],
                "mass_ratio_pinn": np.nan if oscillator else row["mass_ratio_pinn"],
                "mass_ratio_true": np.nan if oscillator else row["mass_ratio_true"],
                "state_rel_l2": row["state_rel_l2_pinn"],
                "gap_rel_l2": row["imposed_gap_rel_l2"],
            }
        )
    return pd.DataFrame(rows)


def residual(folder):
    """The squared residual on and off the points, the mass and the transport speed."""
    pd.set_option("display.width", 250)
    summary = residual_summary(folder)
    print(summary.drop(columns="run_id").round(5).to_string(index=False))


def budget_checks(folder=None):
    """Where the diffusivity settles at 20000 updates, and the weight screen."""
    diagnostics = ADVECTION_DIFFUSION / "diagnostics"
    coefficients = (("velocity", "velocity_true"), ("diffusivity", "diffusivity_true"))
    cell = ["severity", "scaled_physics_weight"]

    def percent(frame, name, truth):
        return 100 * (frame[f"{name}_final"] - frame[truth]) / frame[truth]

    standard = rf.read_csv(ADVECTION_DIFFUSION / "study" / "study.csv")
    budgets = {
        5000: standard[standard.network_seed == 0],
        10000: rf.read_csv(ADVECTION_DIFFUSION / "study" / "steps-10000" / "study.csv"),
        20000: rf.read_csv(diagnostics / "budget-20000-2026-10-08" / "study.csv"),
    }
    rows = []
    for _, key in budgets[20000][cell].drop_duplicates().iterrows():
        row = {"severity": key.severity, "scaled_physics_weight": key.scaled_physics_weight}
        for name, truth in coefficients:
            medians = {}
            for steps, frame in budgets.items():
                part = frame[
                    (frame.severity == key.severity)
                    & (frame.scaled_physics_weight == key.scaled_physics_weight)
                ]
                medians[steps] = percent(part, name, truth).median()
                row[f"{name}_{steps}"] = medians[steps]
            same = standard[
                (standard.severity == key.severity)
                & (standard.scaled_physics_weight == key.scaled_physics_weight)
            ]
            errors = same.assign(e=percent(same, name, truth))
            network = errors.groupby("noise_seed").e.agg(lambda v: v.max() - v.min()).median()
            noise = errors[errors.network_seed == 0].e.agg(lambda v: v.max() - v.min())
            resolution = max(network, noise)
            first = abs(medians[10000] - medians[5000])
            second = abs(medians[20000] - medians[10000])
            row[f"{name}_resolution"] = resolution
            if second > resolution:
                row[f"{name}_status"] = "not settled"
            elif first > resolution:
                row[f"{name}_status"] = "settled by 10,000"
            else:
                row[f"{name}_status"] = "settled by 5,000"
        rows.append(row)
    settling = pd.DataFrame(rows)

    frame = rf.read_csv(diagnostics / "weight-screen-20000-2026-10-08" / "study.csv")
    frame["failures"] = _control_failures(frame.astype(str).to_dict("records"))
    frame["passed"] = frame.failures.map(lambda f: not f)
    screen = (
        frame.groupby("scaled_physics_weight")
        .agg(
            runs=("passed", "size"),
            passed=("passed", "sum"),
            velocity_err=("velocity_rel_error_sol", "max"),
            diffusivity_err=("diffusivity_rel_error_sol", "max"),
            state=("state_rel_l2_pinn", "max"),
            gap=("imposed_gap_rel_l2", "max"),
            mass=("mass_ratio_abs_error_true", "max"),
        )
        .reset_index()
    )
    pd.set_option("display.width", 200)
    print("(a) median error in % of truth, by budget\n")
    print(settling.round(3).to_string(index=False))
    print("\n(b) 20,000-update controls by weight\n")
    print(screen.round(4).to_string(index=False))


def solution_fit_monte_carlo(count, path):
    """Refit the noisy solution fits for noise seeds ``0..count-1`` into ``path``; no network."""
    rows = []
    for seed in range(count):
        relaxed = fit_relaxed(osc_study.relaxed_config(0.0, 1.0, seed, 0), noise_free=False)
        single = fit_solution(osc_study.study_config(0.0, 1.0, seed, 0), noise_free=False)
        rows.append(
            {
                "noise_seed": seed,
                "w0_two": relaxed["w0"],
                "g_two": relaxed["g"],
                "at_boundary_two": relaxed["at_boundary"],
                "w0_one": single["w0"],
            }
        )
        if seed % 100 == 99:
            print(seed + 1, "done", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("finished", flush=True)


def damping_bias(folder):
    """The freed damping by noise seed against the solution fit to the same data."""
    pd.set_option("display.width", 220)
    root = OSCILLATOR / "study"
    seeds = osc_study.FULL_DESIGN["noise_seeds"]
    config = osc_study.relaxed_config(0.0, 1.0, seeds[0], 0)
    w0_true, g_true = config["w0_true"], config["g_true"]
    print(f"truth w0={w0_true:.6f} g={g_true}")
    sol = pd.read_csv(root / "relaxed" / "references.csv")
    sol = sol[sol.reference == "solution_fit"].set_index("noise_seed")
    refs = pd.DataFrame({"g_solfit": pct(sol.g, g_true), "w0_solfit": pct(sol.w0, w0_true)})

    budgets = (("9000", root / "relaxed"), ("18000", root / "relaxed" / "steps-18000"))
    for label, source in budgets:
        runs = pd.read_csv(source / "study.csv")
        runs["g_pinn"] = pct(runs.g_final, g_true)
        for weight in sorted(runs.scaled_physics_weight.unique()):
            part = runs[runs.scaled_physics_weight == weight]
            table = part.pivot(index="noise_seed", columns="init_severity", values="g_pinn").round(
                2
            )
            table["g_solfit"] = refs.g_solfit.round(2)
            print(f"\n== freed damping error (%) at Lambda={weight:g}, {label} updates")
            print(table.to_string())
            print("mean over seeds:", table.mean().round(2).to_dict())

    print("\n== frequency references (%)")
    print(refs[["w0_solfit"]].round(3).to_string())

    print("\n== damping change over the last 10% of training (percentage points of truth)")
    for label, source in budgets:
        runs = pd.read_csv(source / "study.csv")
        rows = []
        for _, row in runs.iterrows():
            g = np.load(source / row.trace_file)[:, 1]
            tail = int(0.9 * (len(g) - 1))
            rows.append(
                {
                    "weight": row.scaled_physics_weight,
                    "init_severity": row.init_severity,
                    "noise_seed": row.noise_seed,
                    "drift": 100 * (g[-1] - g[tail]) / g_true,
                }
            )
        drift = pd.DataFrame(rows).pivot_table(
            index="init_severity", columns="weight", values="drift", aggfunc="median"
        )
        print(f"{label} updates, median over noise seeds")
        print(drift.round(3).to_string())

    mc_path = Path(folder) / "solfit_mc.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        errors = pd.DataFrame(
            {
                "g_two": pct(mc.g_two, g_true),
                "w0_two": pct(mc.w0_two, w0_true),
                "w0_one": pct(mc.w0_one, w0_true),
            }
        )
        n = len(errors)
        print(
            f"\n== noisy solution fits over noise seeds 0-{n - 1} ({n} seeds, "
            f"{int(mc.at_boundary_two.sum())} at a boundary), errors in % of truth"
        )
        for column, name in (
            ("g_two", "damping, two-coefficient fit"),
            ("w0_two", "frequency, two-coefficient fit"),
            ("w0_one", "frequency, one-coefficient fit at s = 0"),
        ):
            x = errors[column]
            sd = x.std(ddof=1)
            study_five = x[x.index.isin(seeds)]
            print(
                f"{name}: mean {x.mean():+.3f} (se {sd / np.sqrt(n):.3f}), sd {sd:.3f}, "
                f"share positive {np.mean(x > 0):.2f}; study seeds mean {study_five.mean():+.3f}, "
                f"sd of a five-seed mean {sd / np.sqrt(len(seeds)):.3f}, "
                f"z {(study_five.mean() - x.mean()) / (sd / np.sqrt(len(seeds))):+.2f}"
            )


def _relaxed_rows(source, w0_true=2 * np.pi, g_true=0.3):
    runs = pd.read_csv(source / "study.csv")
    runs = runs[runs.scaled_physics_weight == 1.0]
    return pd.DataFrame(
        {
            "noise_seed": runs.noise_seed,
            "start": runs.init_severity,
            "g_pinn": pct(runs.g_final, g_true),
            "g_sol": pct(runs.g_sol, g_true),
            "w0_pinn": pct(runs.w0_final, w0_true),
            "w0_sol": pct(runs.w0_sol, w0_true),
        }
    )


def negative_seeds(folder):
    """The PINN against its own seed's solution fit, adding three negative-leaning noise seeds."""
    w0_true = 2 * np.pi
    study = OSCILLATOR / "study"
    pd.set_option("display.width", 200)
    new = _relaxed_rows(Path(folder) / "relaxed-negative-seeds")
    old = _relaxed_rows(study / "relaxed" / "steps-18000")
    old = old[old.start.isin(new.start.unique())]
    both = pd.concat([old.assign(seeds="study 0-4"), new.assign(seeds="negative")])
    both["g_diff"] = both.g_pinn - both.g_sol
    both["w0_diff"] = both.w0_pinn - both.w0_sol
    print("== relaxed check, Lambda = 1, 18000 updates, errors in % of truth")
    print(both.round(3).to_string(index=False))
    print("\nper seed, mean over starts:")
    per_seed = both.groupby(["seeds", "noise_seed"])[
        ["g_pinn", "g_sol", "g_diff", "w0_pinn", "w0_sol"]
    ].mean()
    print(per_seed.round(3).to_string())
    print("\nmean over seeds of PINN minus solution fit (damping, frequency):")
    diffs = both.groupby("noise_seed")[["g_diff", "w0_diff"]].mean()
    print(
        f"  all {len(diffs)} seeds: {diffs.g_diff.mean():+.3f} (sd {diffs.g_diff.std(ddof=1):.3f}), "
        f"{diffs.w0_diff.mean():+.4f} (sd {diffs.w0_diff.std(ddof=1):.4f})"
    )
    print(
        f"  correlation of PINN and solution-fit damping over seeds: "
        f"{per_seed.g_pinn.corr(per_seed.g_sol):.3f}"
    )

    control = pd.read_csv(Path(folder) / "control-negative-seeds" / "study.csv")
    main = pd.read_csv(study / "study.csv")
    main = main[(main.severity == 0) & (main.scaled_physics_weight == 1.0)]
    print(
        "\n== s = 0 control, Lambda = 1, 9000 updates: frequency error (% of truth), "
        "mean over network seeds"
    )
    for label, frame in (("study 0-4", main), ("negative", control)):
        table = frame.groupby("noise_seed").agg(
            w0_pinn=("w0_final", "mean"), w0_sol=("w0_sol", "first")
        )
        table = table.apply(lambda column: pct(column, w0_true))
        print(label)
        print(table.round(3).to_string())
        print(
            f"  mean PINN {table.w0_pinn.mean():+.3f}, mean solution fit {table.w0_sol.mean():+.3f}"
        )


def relaxed_damping_noise(folder):
    """Why the freed damping errs high, and the negative-seed comparison."""
    damping_bias(folder)
    print()
    negative_seeds(folder)


# Median control frequency errors (%) at 9000 updates, from the oscillator's control screen.
SCREEN_9000 = {300.0: -12.958, 1000.0: -25.814}


def large_weight_budget(folder):
    """Is the large-weight failure only slow training? (36000 updates)"""
    folder = Path(folder)
    frame = rf.read_csv(folder / "study.csv")
    frame = frame.assign(error=pct(frame.w0_final, frame.w0_true))
    frame["failures"] = _control_failures(frame.astype(str).to_dict("records"))
    window = []
    for row in frame.itertuples():
        trace = np.load(folder / row.trace_file)
        start = int(round((1 - 0.2) * (len(trace) - 1)))
        window.append(100 * (trace[-1, 0] - trace[start, 0]) / row.w0_true)
    frame["final_window_move"] = window

    for weight, cell in frame[frame.severity == 0].groupby("scaled_physics_weight"):
        passed = int((cell.failures.str.len() == 0).sum())
        median = cell.error.median()
        before = SCREEN_9000.get(weight)
        kind = "slow" if before is not None and abs(median) <= abs(before) / 2 else "stall"
        print(
            f"Lambda = {weight:g}: {passed} of {len(cell)} controls pass; frequency error "
            f"median {median:+.3f}% ({cell.error.min():+.2f}% to {cell.error.max():+.2f}%), "
            f"state median {100 * cell.state_rel_l2_pinn.median():.2f}%, "
            f"gap median {100 * cell.imposed_gap_rel_l2.median():.2f}%"
        )
        verdicts = (
            f"verdicts {cell.verdict.value_counts().to_dict()}; " if "verdict" in cell else ""
        )
        print(
            f"  {verdicts}frequency move over the "
            f"final 20%: median {cell.final_window_move.abs().median():.3f} points, "
            f"largest {cell.final_window_move.abs().max():.3f}"
        )
        print(f"  against {before:+.3f}% at 9000 updates -> {kind}")
        for row in cell[cell.failures.str.len() > 0].itertuples():
            print(f"    noise {row.noise_seed}, network {row.network_seed}: {row.failures}")
    if frame[frame.severity != 0].empty:
        print("No misspecified cells (stage 2 did not run).")


DENSE_GRID_RULED = (0.25, 0.5, 1.0)  # severities in the rule; 0.1 is reported only


def dense_grid_summary(folder):
    """Return the controls passing by density and weight, and one row per density, severity and weight."""
    frames = []
    for n_col in (600, 9600):
        frame = rf.read_csv(Path(folder) / f"n{n_col}" / "study.csv")
        frame["failures"] = _control_failures(frame.astype(str).to_dict("records"))
        frames.append(rf.decompose(frame, "diffusivity", "diffusivity_true"))
    frame = pd.concat(frames, ignore_index=True)
    controls = frame[frame.severity == 0]
    passed = (
        controls.assign(ok=controls.failures.str.len() == 0)
        .groupby(["n_col", "scaled_physics_weight"])
        .ok.agg(["sum", "count"])
    )
    misspecified = frame[frame.severity > 0]

    def pct_of(column, truth="diffusivity_true"):
        return 100 * (misspecified[column] / misspecified[truth] - 1)

    misspecified = misspecified.assign(
        diffusivity_pct=pct_of("diffusivity_final"),
        velocity_pct=pct_of("velocity_final", "velocity_true"),
        ratio_R=misspecified.mse_dense / misspecified.mse_collocation,
        p=misspecified.diffusivity_position.where(~misspecified.diffusivity_off_range),
    )
    rows = []
    for (n_col, severity, weight), cell in misspecified.groupby(
        ["n_col", "severity", "scaled_physics_weight"]
    ):
        low, high = rf.bootstrap_median(cell, "p")
        rows.append(
            {
                "n_col": n_col,
                "severity": severity,
                "scaled_physics_weight": weight,
                "runs": len(cell),
                "off_range": int(cell.diffusivity_off_range.sum()),
                "position": cell.p.median(),
                "position_low": low,
                "position_high": high,
                "diffusivity_pct": cell.diffusivity_pct.median(),
                "velocity_pct": cell.velocity_pct.median(),
                "state_pct": 100 * cell.state_rel_l2_pinn.median(),
                "gap_pct": 100 * cell.imposed_gap_rel_l2.median(),
                "mass_ratio": cell.mass_ratio_pinn.median(),
                "data_loss": cell.data_loss.median(),
                "physics_loss": cell.physics_loss.median(),
                "ratio_R": cell.ratio_R.median(),
                "diffusivity_residual_pct": pct_of("diffusivity_residual_target")
                .loc[cell.index]
                .mean(),
                "diffusivity_target_pct": pct_of("diffusivity_target").loc[cell.index].mean(),
            }
        )
    return passed, misspecified, pd.DataFrame(rows)


def _comparison_weights(passed, misspecified):
    """Return the weights run at both densities whose controls all pass at both."""
    weights = sorted(
        set.intersection(
            *(set(misspecified[misspecified.n_col == n].scaled_physics_weight) for n in (600, 9600))
        )
    )
    return [
        w
        for w in weights
        if all(passed.loc[(n, w), "sum"] == passed.loc[(n, w), "count"] for n in (600, 9600))
    ]


def dense_grid(folder):
    """Does advection--diffusion follow the oscillator once enforced densely?

    Positions are per noise seed, and off-range runs are left out as in the thesis.
    """
    passed, misspecified, summary = dense_grid_summary(folder)
    print("controls passing, by density and weight:")
    print(passed.to_string())
    weights = _comparison_weights(passed, misspecified)
    print(f"weights in the comparison: {weights}")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))

    table = summary.set_index(["n_col", "severity", "scaled_physics_weight"])
    matches, fails = [], []
    for severity in DENSE_GRID_RULED:
        dense = [table.loc[(9600, severity, w)] for w in weights]
        sparse = [table.loc[(600, severity, w)] for w in weights]
        steps = sum(b.position > a.position for a, b in itertools.pairwise(dense))
        rises = steps >= len(weights) - 2
        reaches = dense[-1].position >= 0.8
        separated = all(
            d.position > s.position and d.position_low > s.position_high
            for d, s in zip(dense, sparse, strict=True)
        )
        print(
            f"s = {severity:g}: p rises in {steps} of {len(weights) - 1} steps; "
            f"p({weights[-1]:g}) = {dense[-1].position:.3f}; 9600 above 600 with separate "
            f"intervals at every weight: {separated}"
        )
        matches.append(rises and reaches and separated)
        fails.append(dense[-1].position < 0.5)
    verdict = (
        "MATCHES THE OSCILLATOR PATTERN"
        if all(matches)
        else "DOES NOT MATCH"
        if any(fails)
        else "PARTIAL"
    )
    print(f"verdict by the recorded rule: {verdict}")
    sparse = summary[
        (summary.n_col == 600)
        & (summary.severity >= 0.25)
        & summary.scaled_physics_weight.isin(weights)
    ]
    above = sparse[sparse.position > 0.5]
    print("600-point claim: " + ("CONFIRMED" if above.empty else "NOT CONFIRMED, cells above 0.5:"))
    if not above.empty:
        print(
            above[["severity", "scaled_physics_weight", "position"]].round(3).to_string(index=False)
        )


# One shade and marker per severity, shared by both panels: one hue, light to
# dark as severity grows (L* 60 to 25), the lightest at 3.2:1 contrast on white.
SEVERITY_STYLE = {
    -0.75: ("#5090f1", "v"),
    -0.5: ("#367edd", "^"),
    0.25: ("#0c6dc9", "o"),
    0.5: ("#055cac", "s"),
    1.0: ("#004b8f", "D"),
    2.0: ("#013b73", "P"),
}
# No position exists at s = 0, and at s = -0.25 the targets are too close for a stable one.
OSC_POSITION_LEFT_OUT = (0.0, -0.25)


def oscillator_positions():
    """Return the oscillator's median frequency position by severity (rows) and weight.

    Lambda = 100 comes from the 18000-update runs of every network seed, as in the
    frequency table. Medians use every run: the oscillator's targets are the same for
    every run, so each median position is the position of the median frequency.
    """
    study = rf.read_csv(rf.oscillator_study() / "study.csv")
    longer = pd.concat(
        [rf.read_csv(path) for path in rf.oscillator_lambda100_sources()], ignore_index=True
    )
    reported = pd.concat(
        [
            study[study.scaled_physics_weight != 100.0],
            longer[longer.scaled_physics_weight == 100.0],
        ],
        ignore_index=True,
    )
    frame = rf.decompose(reported, "w0", "w0_true", rf.FLOOR["w0"])
    frame = frame[~frame.severity.isin(OSC_POSITION_LEFT_OUT)]
    return frame.groupby(["severity", "scaled_physics_weight"]).w0_position.median().unstack()


def _weight_axis(ax, weights):
    ax.set_xscale("log")
    ax.set_xticks(weights, [f"{w:g}" for w in weights])
    ax.minorticks_off()
    ax.set_xlabel(r"physics weight $\Lambda$")
    ax.grid(True, alpha=0.9)
    ax.set_axisbelow(True)


def _target_lines(ax):
    ax.axhline(0.0, color=rf.RESIDUAL_END, lw=1.4, ls="--", zorder=2)
    ax.axhline(1.0, color=rf.SOLUTION_END, lw=1.4, ls="-.", zorder=2)


def position_figure(out):
    """Write the position figure: the median position against the weight in both models.

    (a) The oscillator frequency in the main grid. (b) The advection--diffusion
    diffusivity in the full-grid sweep at 600 and 9600 collocation points, at the weights
    where every control passed at both densities. The weight axes differ because Lambda
    cannot be compared between the models.
    """
    oscillator = oscillator_positions()
    passed, misspecified, summary = dense_grid_summary(
        ADVECTION_DIFFUSION / "diagnostics" / "dense-grid-2026-10-25"
    )
    weights = _comparison_weights(passed, misspecified)
    summary = summary[
        summary.severity.isin(DENSE_GRID_RULED) & summary.scaled_physics_weight.isin(weights)
    ]
    with plt.rc_context(rf.style()):
        fig, (left, right) = plt.subplots(1, 2, figsize=rf.figure_size(0.96, 2.9), sharey=True)
        for ax in (left, right):
            _target_lines(ax)
        left.set_ylim(-1.15, 1.35)
        # The targets are labelled on the lines, where panel (a) leaves room.
        left.text(
            100, 0.04, "equation-fit target", ha="right", va="bottom", color=rf.INK, fontsize=8.5
        )
        left.text(
            0.3, 1.06, "solution-fit target", ha="left", va="bottom", color=rf.INK, fontsize=8.5
        )
        for severity, row in oscillator.iterrows():
            shade, marker = SEVERITY_STYLE[severity]
            left.plot(
                row.index,
                row.to_numpy(),
                marker=marker,
                color=shade,
                ms=4,
                lw=1.5,
                zorder=4,
                label=f"$s = {severity:g}$",
            )
        _weight_axis(left, oscillator.columns.to_numpy())
        left.set_title("(a) oscillator frequency", loc="left")
        left.set_ylabel("median position $p$")
        left.legend(loc="lower right", ncol=2, handlelength=1.6)
        for severity in DENSE_GRID_RULED:
            shade, marker = SEVERITY_STYLE[severity]
            for n_col, style in (
                (9600, dict(ls="-", mfc=shade)),
                (600, dict(ls="--", mfc="white")),
            ):
                cell = summary[(summary.n_col == n_col) & (summary.severity == severity)]
                cell = cell.sort_values("scaled_physics_weight")
                right.plot(
                    cell.scaled_physics_weight,
                    cell.position,
                    marker=marker,
                    color=shade,
                    ms=4,
                    lw=1.5,
                    zorder=4,
                    **style,
                )
        _weight_axis(right, np.array(weights))
        right.set_title("(b) advection–diffusion diffusivity", loc="left")
        right.text(
            weights[-1], 1.06, "9600 points", ha="right", va="bottom", color=rf.INK, fontsize=8.5
        )
        right.text(
            weights[-1], -0.95, "600 points", ha="right", va="top", color=rf.INK, fontsize=8.5
        )
        # The densities are labelled on the panel, so the legend names the severities only.
        severity_handles = [
            plt.Line2D(
                [],
                [],
                color=SEVERITY_STYLE[s][0],
                marker=SEVERITY_STYLE[s][1],
                ms=4,
                lw=1.5,
                label=f"$s = {s:g}$",
            )
            for s in DENSE_GRID_RULED
        ]
        right.legend(
            handles=severity_handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.6),
            handlelength=1.6,
        )
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "position-both-models.pdf", facecolor="white")
        plt.close(fig)
    return ["position-both-models.pdf"]


DIAGNOSTICS = {
    "collocation-residual-2026-10-08": (ADVECTION_DIFFUSION / "diagnostics", residual),
    "collocation-density-2026-10-08": (ADVECTION_DIFFUSION / "diagnostics", density),
    "collocation-nested-2026-10-11": (ADVECTION_DIFFUSION / "diagnostics", nested),
    "dense-grid-2026-10-25": (ADVECTION_DIFFUSION / "diagnostics", dense_grid),
    "budget-20000-2026-10-08": (ADVECTION_DIFFUSION / "diagnostics", budget_checks),
    "weight-screen-20000-2026-10-08": (ADVECTION_DIFFUSION / "diagnostics", budget_checks),
    "collocation-thinned-2026-10-11": (OSCILLATOR / "diagnostics", thinned),
    "large-weight-budget-2026-10-25": (OSCILLATOR / "diagnostics", large_weight_budget),
    "relaxed-damping-noise-2026-10-15": (OSCILLATOR / "diagnostics", relaxed_damping_noise),
}
