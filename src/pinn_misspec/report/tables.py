"""Every number the thesis quotes, recomputed from the saved results.

``tables`` prints the table rows as LaTeX, ``numbers`` the other quoted values and
``intervals`` the cluster-bootstrap intervals (noise seeds resampled, 2000 draws, fixed
seed). Medians, positions and intervals come from ``report.frame``, so they match the
figures.
"""

import numpy as np
import pandas as pd

from pinn_misspec.paths import ADVECTION_DIFFUSION

from . import diagnostics
from .frame import (
    FLOOR,
    advdiff_study,
    bootstrap_median,
    decompose,
    oscillator_lambda100_sources,
    oscillator_study,
    read_csv,
)


def _osc():
    return oscillator_study()


def _adv():
    return advdiff_study()


def _diag():
    return ADVECTION_DIFFUSION / "diagnostics"


def num(x, digits=2, sign=True):
    """Format a number for a table cell in math mode."""
    if x is None or not np.isfinite(x):
        return "--"
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if text.lstrip("+-") == f"{0:.{digits}f}":
        text = f"{0:.{digits}f}"
    return f"${text}$"


def interval(low, high, digits=2, sign=True):
    return f"{{\\scriptsize[{num(low, digits, sign)}, {num(high, digits, sign)}]}}"


def section(title):
    print(f"\n%% ---- {title}")


def cell_summary(frame, coefficient):
    """Return the median error and position, and their intervals, per cell, in-range runs only."""
    rows = {}
    for key, everything in frame.groupby(["severity", "scaled_physics_weight"]):
        cell = everything[~everything[f"{coefficient}_off_range"]]
        total = cell[f"{coefficient}_total"]
        low, high = bootstrap_median(cell, f"{coefficient}_total")
        p_low, p_high = bootstrap_median(cell, f"{coefficient}_position")
        rows[key] = dict(
            total=100 * total.median(),
            low=100 * low,
            high=100 * high,
            p=cell[f"{coefficient}_position"].median(),
            p_low=p_low,
            p_high=p_high,
            off=int(everything[f"{coefficient}_off_range"].sum()),
            runs=len(everything),
            shift=100 * cell[f"{coefficient}_shift"].mean(),
            noise=100 * cell[f"{coefficient}_noise"].mean(),
            pinn=100 * cell[f"{coefficient}_pinn"].mean(),
            residual=100 * cell[f"{coefficient}_residual_error"].mean(),
        )
    return rows


def reported_oscillator(study):
    """Return the main-grid rows with Lambda = 100 taken from the 18000-update runs.

    The standard budget is too short at Lambda = 100, so the frequency and state tables
    report that column from every network seed at 18000 updates.
    """
    longer = pd.concat(
        [read_csv(path) for path in oscillator_lambda100_sources()], ignore_index=True
    )
    longer = longer[longer.scaled_physics_weight == 100.0]
    assert len(longer) == 105, f"expected 105 runs at Lambda 100, found {len(longer)}"
    return pd.concat([study[study.scaled_physics_weight != 100.0], longer], ignore_index=True)


def _residual_fits(frame, severity):
    """The residual fit along a data-only network, in archived references only."""
    return frame[(frame.severity == severity) & (frame.reference == "residual_fit")]


def oscillator_rows():
    study = read_csv(_osc() / "study.csv")
    frame = decompose(study, "w0", "w0_true", FLOOR["w0"])
    reported = reported_oscillator(study)
    reported_frame = decompose(reported, "w0", "w0_true", FLOOR["w0"])
    cells = cell_summary(frame, "w0")
    weights = sorted(frame.scaled_physics_weight.unique())
    severities = sorted(frame.severity.unique())
    truth = float(study.w0_true.iloc[0])

    section("oscillator references by severity")
    refs = read_csv(_osc() / "references.csv")
    res = read_csv(_osc() / "residual_references.csv")
    for s in severities:
        r, q = refs[refs.severity == s], res[res.severity == s]
        target = 100 * (r[r.reference == "target"].w0.iloc[0] - truth) / truth
        sol = 100 * (r[r.reference == "solution_fit"].w0 - truth) / truth
        rtarget = 100 * (q[q.reference == "residual_target"].w0.iloc[0] - truth) / truth
        gamma = float(r.gamma_imposed.iloc[0])
        line = (
            f"    {num(s, 2, False)} & ${gamma:.3f}$ & {num(target)} & {num(sol.median())} "
            f"{interval(sol.min(), sol.max())} & {num(rtarget)}"
        )
        rfit = 100 * (_residual_fits(res, s).w0 - truth) / truth
        if len(rfit):
            line += f" & {num(rfit.median())} {interval(rfit.min(), rfit.max())}"
        print(line + " \\\\")

    section("oscillator median frequency error by weight (all runs; Lambda 100 at 18000 updates)")
    for s in severities:
        cols = [
            num(
                100
                * reported_frame[
                    (reported_frame.severity == s) & (reported_frame.scaled_physics_weight == w)
                ].w0_total.median()
            )
            for w in weights
        ]
        r = cells[(s, weights[0])]
        print(
            f"    {num(s, 2, False)} & "
            + " & ".join(cols)
            + f" & {num(r['residual'])} & {num(r['shift'])} \\\\"
        )

    section("oscillator median position, all runs (Lambda 100 at 18000 updates; Figure 6.3)")
    for s, row in diagnostics.oscillator_positions().iterrows():
        print(f"    {num(s, 2, False)} & " + " & ".join(num(p, 2, False) for p in row) + " \\\\")

    section("oscillator median position with intervals")
    for s in severities:
        if s == 0:
            continue
        cols = []
        for w in weights:
            c = cells[(s, w)]
            text = f"{num(c['p'], 2, False)} {interval(c['p_low'], c['p_high'], 2, False)}"
            if c["off"]:
                text += f"$^{{{c['off']}}}$"
            cols.append(text)
        print(f"    {num(s, 2, False)} & " + " & ".join(cols) + " \\\\")

    section("oscillator brackets (mean over cells, %), by severity")
    for s in severities:
        noise = np.mean([cells[(s, w)]["noise"] for w in weights])
        pinn = [cells[(s, w)]["pinn"] for w in weights]
        print(
            f"    s={s}: shift {cells[(s, weights[0])]['shift']:.3f}  noise {noise:.3f}  "
            f"pinn by weight {[round(v, 3) for v in pinn]}"
        )

    section("oscillator state error, gap (%) and data loss by weight (Lambda 100 at 18000 updates)")
    for s in (-0.75, 0.0, 1.0, 2.0):
        part = reported[reported.severity == s]
        state = [
            100 * part[part.scaled_physics_weight == w].state_rel_l2_pinn.median() for w in weights
        ]
        gap = [
            100 * part[part.scaled_physics_weight == w].imposed_gap_rel_l2.median() for w in weights
        ]
        data = [1e3 * part[part.scaled_physics_weight == w].data_loss.median() for w in weights]
        sol = 100 * part.state_rel_l2_sol.median()
        print(
            f"    {num(s, 2, False)} & state & "
            + " & ".join(num(v, 2, False) for v in state)
            + f" & {num(sol, 2, False)} \\\\"
        )
        print("     & gap & " + " & ".join(num(v, 2, False) for v in gap) + " & -- \\\\")
        print("     & data & " + " & ".join(num(v, 2, False) for v in data) + " & -- \\\\")

    section("oscillator Lambda = 100 budget check, network seed 0")
    long_frame = decompose(
        read_csv(_osc() / "steps-18000" / "study.csv"), "w0", "w0_true", FLOOR["w0"]
    )
    base = frame[(frame.network_seed == 0) & (frame.scaled_physics_weight == 100.0)]
    long_frame = long_frame[long_frame.scaled_physics_weight == 100.0]
    for s in severities:
        b = base[base.severity == s]
        long = long_frame[long_frame.severity == s]
        lo, hi = bootstrap_median(long, "w0_total")
        plo, phi = bootstrap_median(long, "w0_position")
        p = long.w0_position.median()
        print(
            f"    {num(s, 2, False)} & {num(100 * b.w0_total.median())} & "
            f"{num(100 * long.w0_total.median())} {interval(100 * lo, 100 * hi)} & "
            f"{num(cells[(s, 100.0)]['shift'])} & "
            + ("--" if s == 0 else f"{num(p, 2, False)} {interval(plo, phi, 2, False)}")
            + " \\\\"
        )


def advection_diffusion_rows():
    study = read_csv(_adv() / "study.csv")
    weights = sorted(study.scaled_physics_weight.unique())
    severities = sorted(study.severity.unique())
    a0, k0 = float(study.velocity_true.iloc[0]), float(study.diffusivity_true.iloc[0])

    section("advection-diffusion references by severity (diffusivity, velocity)")
    refs = read_csv(_adv() / "references.csv")
    res = read_csv(_adv() / "residual_references.csv")
    for name, truth in (("diffusivity", k0), ("velocity", a0)):
        print(f"%   {name}")
        for s in severities:
            r, q = refs[refs.severity == s], res[res.severity == s]
            target = 100 * (r[r.reference == "target"][name] - truth) / truth
            sol = 100 * (r[r.reference == "solution_fit"][name] - truth) / truth
            rtarget = 100 * (q[q.reference == "residual_target"][name] - truth) / truth
            digits = 1 if name == "diffusivity" else 2
            line = (
                f"    {num(s, 2, False)} & ${np.exp(-s):.3f}$ & {num(target.mean(), digits)} "
                f"{interval(target.min(), target.max(), digits)} & {num(sol.median(), digits)} "
                f"{interval(sol.min(), sol.max(), digits)} & {num(rtarget.mean(), digits)} "
                f"{interval(rtarget.min(), rtarget.max(), digits)}"
            )
            rfit = 100 * (_residual_fits(res, s)[name] - truth) / truth
            if len(rfit):
                line += (
                    f" & {num(rfit.median(), digits)} {interval(rfit.min(), rfit.max(), digits)}"
                )
            print(line + " \\\\")

    for name, truth_col in (("diffusivity", "diffusivity_true"), ("velocity", "velocity_true")):
        frame = decompose(study, name, truth_col, FLOOR[name])
        cells = cell_summary(frame, name)
        digits = 1 if name == "diffusivity" else 2
        section(f"advection-diffusion median {name} error with intervals, and references")
        for s in severities:
            cols = [
                f"{num(cells[(s, w)]['total'], digits)} "
                f"{interval(cells[(s, w)]['low'], cells[(s, w)]['high'], digits)}"
                for w in weights
            ]
            r = cells[(s, weights[0])]
            off = ", ".join(str(cells[(s, w)]["off"]) for w in weights)
            print(
                f"    {num(s, 2, False)} & "
                + " & ".join(cols)
                + f" & {num(r['residual'], digits)} & {num(r['shift'], digits)} \\\\  % off {off}"
            )
        section(f"advection-diffusion median {name} position with intervals")
        for s in severities:
            if s == 0:
                continue
            cols = [
                f"{num(cells[(s, w)]['p'], 2, False)} "
                f"{interval(cells[(s, w)]['p_low'], cells[(s, w)]['p_high'], 2, False)}"
                for w in weights
            ]
            print(f"    {num(s, 2, False)} & " + " & ".join(cols) + " \\\\")

    section("advection-diffusion state error, gap (%), mass ratio, data loss")
    for s in (0.0, 0.5, 1.0):
        part = study[study.severity == s]

        def med(column, scale=1.0, part=part):
            return [scale * part[part.scaled_physics_weight == w][column].median() for w in weights]

        print(
            f"    {num(s, 2, False)} & state & "
            + " & ".join(num(v, 2, False) for v in med("state_rel_l2_pinn", 100))
            + f" & {num(100 * part.state_rel_l2_sol.median(), 2, False)} \\\\"
        )
        print(
            "     & gap & "
            + " & ".join(num(v, 1, False) for v in med("imposed_gap_rel_l2", 100))
            + " & -- \\\\"
        )
        print(
            "     & mass & "
            + " & ".join(num(v, 3, False) for v in med("mass_ratio_pinn"))
            + f" & ${np.exp(-s):.3f}$ \\\\"
        )
        print(
            f"%    data loss x1e3 {[round(v, 3) for v in med('data_loss', 1e3)]}  "
            f"physics {[round(v, 4) for v in med('physics_loss')]}"
        )


def settling():
    section("settling check, Lambda = 0.3, network seed 0, median diffusivity error")
    main = read_csv(_adv() / "study.csv")
    mid = read_csv(_adv() / "steps-10000" / "study.csv")
    long = read_csv(_diag() / "budget-20000-2026-10-08" / "study.csv")
    k0 = float(main.diffusivity_true.iloc[0])
    for s in (0.0, 0.5, 1.0):
        values = []
        for frame in (main, mid, long):
            part = frame[
                (frame.severity == s)
                & (frame.scaled_physics_weight == 0.3)
                & (frame.network_seed == 0)
            ]
            values.append(100 * (part.diffusivity_final.median() - k0) / k0)
        part = main[
            (main.severity == s) & (main.scaled_physics_weight == 0.3) & (main.network_seed == 0)
        ]
        res = 100 * (part.diffusivity_residual_target.mean() - k0) / k0
        print(
            f"    {num(s, 2, False)} & "
            + " & ".join(num(v, 1) for v in values)
            + f" & {num(res, 1)} \\\\"
        )


def budget_positions():
    """Rows of the position-by-weight-and-budget table, network seed 0, 600 points."""
    section("advection-diffusion position by weight and budget, network seed 0, median [run range]")
    sources = {
        5000: [_adv() / "study.csv"],
        10000: [_adv() / "steps-10000" / "study.csv"],
        20000: [
            _diag() / "budget-20000-2026-10-08" / "study.csv",
            _diag() / "supplementary-20000-2026-10-08" / "study.csv",
        ],
    }
    weights = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    for steps, paths in sources.items():
        frame = pd.concat([read_csv(path) for path in paths], ignore_index=True)
        frame = decompose(
            frame[frame.network_seed == 0], "diffusivity", "diffusivity_true", FLOOR["diffusivity"]
        )
        for s in (0.5, 1.0):
            cols = []
            for w in weights:
                p = frame[
                    (frame.severity == s) & (frame.scaled_physics_weight == w)
                ].diffusivity_position
                cols.append(
                    "--"
                    if p.empty
                    else f"{num(p.median(), 2, False)} {interval(p.min(), p.max(), 2, False)}"
                )
            label = f"${steps}$" if s == 0.5 else ""
            print(
                f"    {label} & {num(s, 1, False).replace('.0$', '$')} & "
                + " & ".join(cols)
                + " \\\\"
            )


def supplementary_rows():
    study = read_csv(_diag() / "supplementary-20000-2026-10-08" / "study.csv")
    frame = decompose(study, "diffusivity", "diffusivity_true", FLOOR["diffusivity"])
    cells = cell_summary(frame, "diffusivity")
    section("supplementary set, median diffusivity error and position")
    for s in sorted(study.severity.unique()):
        if s == 0:
            continue
        r = cells[(s, 1.0)]
        row = [num(r["residual"], 1), num(r["shift"], 1)]
        for w in (1.0, 3.0):
            c = cells[(s, w)]
            row += [
                num(c["total"], 1),
                f"{num(c['p'], 2, False)} {interval(c['p_low'], c['p_high'], 2, False)}",
            ]
        print(f"    {num(s, 2, False)} & " + " & ".join(row) + " \\\\")
    a0 = float(study.velocity_true.iloc[0])
    section("supplementary velocity (all runs), gap, mass")
    for (s, w), part in study.groupby(["severity", "scaled_physics_weight"]):
        v = 100 * (part.velocity_final - a0) / a0
        print(
            f"%   s={s} L={w}: velocity median {v.median():.2f} range [{v.min():.2f}, {v.max():.2f}] "
            f"gap {100 * part.imposed_gap_rel_l2.median():.1f} mass {part.mass_ratio_pinn.median():.3f} "
            f"state {100 * part.state_rel_l2_pinn.median():.2f} data {part.data_loss.median():.5f} "
            f"phys {part.physics_loss.median():.5f} phys max {part.physics_loss.max():.5f}"
        )


def diffusivity_sweep():
    """Rows of the diffusivity table: weights up to 0.3 at 5000 updates, 1 and 3 at 20000, medians over all runs."""
    main = read_csv(_adv() / "study.csv")
    extra = read_csv(_diag() / "supplementary-20000-2026-10-08" / "study.csv")
    k0 = float(main.diffusivity_true.iloc[0])
    targets = cell_summary(
        decompose(main, "diffusivity", "diffusivity_true", FLOOR["diffusivity"]), "diffusivity"
    )
    section("advection-diffusion diffusivity across all weights (all runs), with references")
    for s in sorted(main.severity.unique()):
        cols = []
        for frame in (main, extra):
            part = frame[frame.severity == s]
            for w in sorted(frame.scaled_physics_weight.unique()):
                values = part[part.scaled_physics_weight == w].diffusivity_final
                cols.append(num(100 * (values.median() - k0) / k0, 1))
        r = targets[(s, 0.01)]
        print(
            f"    {num(s, 2, False)} & "
            + " & ".join(cols)
            + f" & {num(r['residual'], 1)} & {num(r['shift'], 1)} \\\\"
        )


def collocation():
    section("collocation-residual diagnostic")
    print(diagnostics.residual_summary(_diag() / "collocation-residual-2026-10-08").to_string())
    section("collocation density, every run")
    d = diagnostics.density_frame(_diag() / "collocation-density-2026-10-08")
    d["R"] = d.mse_dense / d.mse_collocation
    for (s, w, seed), part in d.groupby(["severity", "scaled_physics_weight", "noise_seed"]):
        part = part.sort_values("n_col")
        mse = " / ".join(f"{v:.2g}" for v in part.mse_dense)
        pos = " / ".join("--" if not np.isfinite(v) else f"{v:.2f}" for v in part.position)
        kap = " / ".join(f"{v:+.0f}" for v in part.diffusivity_pct)
        target = 100 * (part.diffusivity_target.iloc[0] - 0.01) / 0.01
        line = (
            f"    s={s} L={w} seed={seed}: mse {mse} | R {' / '.join(f'{v:.3g}' for v in part.R)} | "
            f"p {pos} | kappa {kap} target {target:+.0f} | mass {' / '.join(f'{v:.2f}' for v in part.mass_ratio_pinn)} | "
            f"gap {' / '.join(f'{100 * v:.1f}' for v in part.gap_rel_l2)} | data {' / '.join(f'{v:.4f}' for v in part.data_loss)} | "
            f"a {' / '.join(f'{v:.4f}' for v in part.velocity)} speed {' / '.join(f'{v:.4f}' for v in part.speed_state)}"
        )
        if "verdict" in part:
            line += f" | verdict {list(part.verdict)}"
        print(line)


def pct(x, truth):
    return 100.0 * (x - truth) / truth


def show(title, frame):
    print(f"\n## {title}\n")
    print(frame.to_string(float_format=lambda v: f"{v:.4g}"))


def _verdicts(control):
    """The verdict counts of archived controls; new runs have no verdict."""
    return {"verdicts": [control.verdict.value_counts().to_dict()]} if "verdict" in control else {}


def oscillator_numbers():
    study = pd.read_csv(_osc() / "study.csv")
    truth = float(study.w0_true.iloc[0])
    refs = pd.read_csv(_osc() / "references.csv")
    res = pd.read_csv(_osc() / "residual_references.csv")
    show("oscillator reference kinds", refs.groupby("reference").size().to_frame("rows"))
    show("oscillator residual reference kinds", res.groupby("reference").size().to_frame("rows"))
    rows = []
    for s, part in refs.groupby("severity"):
        rpart = res[res.severity == s]
        row = {"s": s}
        for kind, sub in part.groupby("reference"):
            v = pct(sub.w0, truth)
            row[f"{kind}_med"] = v.median()
            row[f"{kind}_min"] = v.min()
            row[f"{kind}_max"] = v.max()
            row[f"{kind}_interior_minima"] = sub.n_interior_minima.max()
        for kind, sub in rpart.groupby("reference"):
            v = pct(sub.w0, truth)
            row[f"res_{kind}_med"] = v.median()
            row[f"res_{kind}_min"] = v.min()
            row[f"res_{kind}_max"] = v.max()
        rows.append(row)
    show("oscillator references by severity (% of truth)", pd.DataFrame(rows).set_index("s"))

    control = study[study.severity == 0]
    show(
        "oscillator control maxima",
        pd.DataFrame(
            {
                "w0_rel_error_sol_max_pct": [100 * control.w0_rel_error_sol.max()],
                "w0_rel_error_true_max_pct": [100 * control.w0_rel_error_true.max()],
                "state_max_pct": [100 * control.state_rel_l2_pinn.max()],
                "gap_max_pct": [100 * control.imposed_gap_rel_l2.max()],
                **_verdicts(control),
                "runs": [len(control)],
            }
        ),
    )
    cells = study.groupby(["severity", "scaled_physics_weight"]).agg(
        state=("state_rel_l2_pinn", "median"),
        state_sol=("state_rel_l2_sol", "median"),
        gap=("imposed_gap_rel_l2", "median"),
        data=("data_loss", "median"),
        physics=("physics_loss", "median"),
    )
    cells[["state", "state_sol", "gap"]] *= 100
    show("oscillator state %, solution-fit state %, gap %, losses", cells.unstack(0).round(4))


def advection_diffusion_numbers():
    study = pd.read_csv(_adv() / "study.csv")
    a0, k0 = float(study.velocity_true.iloc[0]), float(study.diffusivity_true.iloc[0])
    refs = pd.read_csv(_adv() / "references.csv")
    res = pd.read_csv(_adv() / "residual_references.csv")
    show("advdiff reference kinds", refs.groupby("reference").size().to_frame("rows"))
    show("advdiff residual reference kinds", res.groupby("reference").size().to_frame("rows"))
    rows = []
    for s, part in refs.groupby("severity"):
        rpart = res[res.severity == s]
        row = {"s": s}
        for kind, sub in part.groupby("reference"):
            for name, truth in (("velocity", a0), ("diffusivity", k0)):
                v = pct(sub[name], truth)
                row[f"{kind}_{name}_med"] = v.median()
                row[f"{kind}_{name}_min"] = v.min()
                row[f"{kind}_{name}_max"] = v.max()
        for kind, sub in rpart.groupby("reference"):
            for name, truth in (("velocity", a0), ("diffusivity", k0)):
                v = pct(sub[name], truth)
                row[f"res_{kind}_{name}_med"] = v.median()
                row[f"res_{kind}_{name}_min"] = v.min()
                row[f"res_{kind}_{name}_max"] = v.max()
        rows.append(row)
    show("advdiff references by severity (% of truth)", pd.DataFrame(rows).set_index("s").T)

    control = study[study.severity == 0]
    show(
        "advdiff control maxima",
        pd.DataFrame(
            {
                "velocity_sol_max_pct": [100 * control.velocity_rel_error_sol.max()],
                "diffusivity_sol_max_pct": [100 * control.diffusivity_rel_error_sol.max()],
                "state_max_pct": [100 * control.state_rel_l2_pinn.max()],
                "gap_max_pct": [100 * control.imposed_gap_rel_l2.max()],
                "mass_abs_max": [control.mass_ratio_abs_error_true.max()],
                **_verdicts(control),
                "runs": [len(control)],
            }
        ),
    )
    cells = study.groupby(["severity", "scaled_physics_weight"]).agg(
        state=("state_rel_l2_pinn", "median"),
        state_sol=("state_rel_l2_sol", "median"),
        gap=("imposed_gap_rel_l2", "median"),
        mass=("mass_ratio_pinn", "median"),
        data=("data_loss", "median"),
        physics=("physics_loss", "median"),
    )
    cells[["state", "state_sol", "gap"]] *= 100
    show("advdiff state %, solution-fit state %, gap %, mass, losses", cells.unstack(0).round(4))


def supplementary_numbers():
    study = pd.read_csv(_diag() / "supplementary-20000-2026-10-08" / "study.csv")
    a0 = float(study.velocity_true.iloc[0])
    study["velocity_pct"] = pct(study.velocity_final, a0)
    cells = study.groupby(["severity", "scaled_physics_weight"]).agg(
        velocity=("velocity_pct", "median"),
        vmin=("velocity_pct", "min"),
        vmax=("velocity_pct", "max"),
        state=("state_rel_l2_pinn", "median"),
        gap=("imposed_gap_rel_l2", "median"),
        mass=("mass_ratio_pinn", "median"),
        data=("data_loss", "median"),
        physics=("physics_loss", "median"),
        runs=("run_id", "size"),
    )
    cells[["state", "gap"]] *= 100
    show("supplementary 20,000 updates", cells.round(4))
    control = study[study.severity == 0]
    show(
        "supplementary control maxima",
        pd.DataFrame(
            {
                "velocity_sol_max_pct": [100 * control.velocity_rel_error_sol.max()],
                "diffusivity_sol_max_pct": [100 * control.diffusivity_rel_error_sol.max()],
                "state_max_pct": [100 * control.state_rel_l2_pinn.max()],
                **_verdicts(control),
                "runs": [len(control)],
            }
        ),
    )


def interval_table(frame, coefficient, truth, floor, group):
    """Return the median error and position with intervals, per group of rows."""
    decomposed = decompose(frame, coefficient, truth, floor)
    decomposed = decomposed[~decomposed[f"{coefficient}_off_range"]]
    rows = []
    for key, cell in decomposed.groupby(group):
        key = key if isinstance(key, tuple) else (key,)
        low, high = bootstrap_median(cell, f"{coefficient}_total")
        p_low, p_high = bootstrap_median(cell, f"{coefficient}_position")
        rows.append(
            {
                **dict(zip(group, key, strict=True)),
                "runs": len(cell),
                "median_pct": 100 * cell[f"{coefficient}_total"].median(),
                "ci_pct": f"[{100 * low:+.2f}, {100 * high:+.2f}]",
                "position": cell[f"{coefficient}_position"].median(),
                "position_ci": f"[{p_low:.2f}, {p_high:.2f}]",
            }
        )
    return pd.DataFrame(rows)


def show_intervals(title, table):
    print(f"\n## {title}\n")
    print(table.round(3).to_string(index=False))


def tables_part():
    oscillator_rows()
    advection_diffusion_rows()
    settling()
    budget_positions()
    supplementary_rows()
    diffusivity_sweep()
    collocation()


def numbers_part():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    oscillator_numbers()
    advection_diffusion_numbers()
    supplementary_numbers()


def intervals_part():
    pd.set_option("display.width", 200)
    group = ["severity", "scaled_physics_weight"]
    main = read_csv(_osc() / "study.csv")
    show_intervals(
        "Oscillator, frequency, main grid", interval_table(main, "w0", "w0_true", 0.01, group)
    )
    long = read_csv(_osc() / "steps-18000" / "study.csv")
    show_intervals(
        "Oscillator, frequency, 18,000 updates, Lambda = 100 (network seed 0)",
        interval_table(long[long.scaled_physics_weight == 100.0], "w0", "w0_true", 0.01, group),
    )
    adv = read_csv(_adv() / "study.csv")
    for name, truth, floor in (
        ("diffusivity", "diffusivity_true", 0.10),
        ("velocity", "velocity_true", 0.02),
    ):
        show_intervals(
            f"Advection--diffusion, {name}, main grid",
            interval_table(adv, name, truth, floor, group),
        )
    for folder in ("budget-20000-2026-10-08", "supplementary-20000-2026-10-08"):
        path = _diag() / folder / "study.csv"
        if path.exists():
            show_intervals(
                f"Advection--diffusion, diffusivity, {folder}",
                interval_table(read_csv(path), "diffusivity", "diffusivity_true", 0.10, group),
            )
