# uq-math7012-code

Code for the MATH7012 thesis *System identification with physics-informed neural
networks: a simulation study of parameter compensation under model
misspecification* (Joshua Cox, The University of Queensland, November 2026).

The studies fit inverse physics-informed neural networks (PINNs) to synthetic
data from two models whose imposed equation is deliberately wrong: a damped
oscillator with its damping fixed away from the truth, and advection--diffusion
with a reaction sink omitted. Each design crosses the misspecification severity
`s` with the physics weight `Λ`, repeats every cell over noise and network seeds,
and compares each PINN estimate with two targets that need no network: the
equation-fit target `θ_eq(s)` and the solution-fit target `θ_sol(s)`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Everything runs on the CPU in double
precision.

```sh
uv sync
```

The dependencies are pinned to the versions that reproduce the saved results
exactly (JAX 0.11.2, Equinox 0.13.8, Optax 0.2.8). Python 3.12.

## The results

The saved runs are published separately as a results archive (about 73 MB),
[doi:10.5281/zenodo.24096153](https://doi.org/10.5281/zenodo.24096153). Unzip it in the root of this repository, which creates
`results/`, or anywhere else with `PINN_MISSPEC_RESULTS` pointing at its
`results/` folder. `ARCHIVE-README.md` inside it describes the contents, and
`SHA256SUMS` lets you check them.

## Reproducing the thesis

These commands read the archive and train nothing; together they take a few
minutes.

```sh
uv run python -m pinn_misspec.report tables      # table rows, as LaTeX
uv run python -m pinn_misspec.report numbers     # other quoted values
uv run python -m pinn_misspec.report intervals   # cluster-bootstrap intervals (not reported in the thesis)
uv run python -m pinn_misspec.report checks      # budget-check counts
uv run python -m pinn_misspec.report screens     # control screens
uv run python -m pinn_misspec.report figures     # the nine figures, to results/thesis_figures/
uv run python -m pinn_misspec.report network-figures  # the two drawn from saved networks (below)
uv run python -m pinn_misspec.report diagnostic <name>
```

| In the thesis | Command | Where in the output |
|---|---|---|
| Figures 1.2, 1.3, 3.1 | `figures` | `falling-object*.pdf`, `advdiff-misspecification.pdf` (the command also draws `osc-misspecification.pdf`, which the final thesis does not use) |
| Figures 6.1, C.2 | `figures` | `osc-w0-vs-weight.pdf`, `advdiff-diffusivity-vs-weight.pdf` |
| Figures C.1, C.3 | `figures` | `osc-relaxed.pdf`, `advdiff-relaxed.pdf` |
| Figure 6.3 | `figures` | `position-both-models.pdf` |
| Figures 6.2, 6.4 | `network-figures` | `osc-neural-state.pdf`, `advdiff-residual-field.pdf`; these need the networks saved by `dense-grid-2026-10-25` and `data-residual-structure-2026-10-25`, which the results archive does not include |
| Tables 5.2, 5.3 | — | the constants of `oscillator/` and `advection_diffusion/` (`model.py`, `study.py`) |
| Table 6.1 | `tables` | oscillator median frequency error by weight; its target columns from oscillator references by severity |
| Section 6.2, the oscillator's positions | `tables` | oscillator median position, all runs |
| Table 6.2 | `tables` | advection-diffusion diffusivity across all weights; its target columns from advection-diffusion references by severity |
| Table 6.3 | `tables`, `numbers` | a selection from the blocks behind Tables C.1 and C.3, with the two advection-diffusion gaps to two decimals from `numbers` (advdiff state %, solution-fit state %, gap %) |
| Table 6.4 | `diagnostic dense-grid-2026-10-25` | the `position` column |
| Table C.4 | `diagnostic collocation-nested-2026-10-11` | the block headed Table 6.5 at the end, the table's number before it moved to Appendix C |
| Tables B.1, B.2 | `screens` | |
| Table C.1 | `tables` | oscillator state error, gap and data loss |
| Table C.2 | `tables` | oscillator Lambda = 100 budget check |
| Table C.3 | `tables` | advection-diffusion state error, gap, mass ratio |
| Sections 5.7, 5.8, 6.1 to 6.8, and Appendices C and D | `numbers`, `checks`, and the diagnostics below | |

| Diagnostic | Thesis |
|---|---|
| `collocation-residual-2026-10-08` | the residual between the collocation points and the velocity drift, Appendices D.3 and D.4 |
| `collocation-density-2026-10-08` | the first density test, Appendix D.3 |
| `collocation-nested-2026-10-11` | nested collocation, Section 6.6 and Table C.4 |
| `dense-grid-2026-10-25` | denser collocation over the full grid, Section 6.6 and Table 6.4 |
| `budget-20000-2026-10-08` | the 20000-update controls and settling, Sections 6.1 and 6.4 |
| `collocation-thinned-2026-10-11` | thinning the oscillator's collocation, Appendix D.2 |
| `large-weight-budget-2026-10-25` | stronger oscillator weights, Appendix D.1 |
| `relaxed-damping-noise-2026-10-15` | why the freed damping errs high (no longer reported in the thesis) |

## Rerunning the studies

Each script in `studies/` runs the designs behind one group of thesis results and
writes where the archive keeps them, so the report commands read a rerun exactly
as they read the archive. A finished run is recorded at once, and a restarted
script skips what is recorded.

| Script | Parts | Thesis |
|---|---|---|
| `oscillator_main.py` | `screen`, `main`, `budget`, `lambda100` | Sections 5.6, 5.7, 6.2, and 6.4 |
| `oscillator_checks.py` | `relaxed`, `damping-noise`, `large-weight` | Sections 5.7 and 6.4, Appendix D.1 (`damping-noise` is no longer reported) |
| `oscillator_thinned.py` | `thinned` | Section 6.7, Appendix D.2 |
| `advdiff_main.py` | `screen`, `main`, `budget` | Sections 5.6, 5.7, 6.3, and 6.4 |
| `advdiff_checks.py` | `relaxed`, `controls-20000`, `supplementary`, `budget-20000` | Sections 5.6, 5.7, 6.1, 6.3, and 6.4 |
| `advdiff_collocation.py` | `residual`, `density`, `nested`, `full-grid` | Sections 5.8, 6.5, and 6.6, Appendices D.3 and D.4 |

```sh
uv run python studies/oscillator_main.py --only main --workers 6
uv run python studies/advdiff_collocation.py --only nested --workers 6
uv run python studies/oscillator_checks.py --only large-weight --steps 50 --output scratch   # a quick check
```

`--workers N` trains cells in `N` single-threaded subprocesses; the worker count
changes the speed, not the coefficients. `--steps` replaces every training budget
for a quick check of the pipeline and needs `--output`, so a shortened run never
mixes with real results. Controls trained that briefly fail their limits, so a
quick check of a gated part stops at the gate; the example uses a part made of
controls only. The full studies took about 1400 CPU-hours, most of them
in the 9600-point half of the full-grid density sweep (thesis Appendix A.4 lists the runs).

Each design that contains misspecified cells first runs its zero-severity
controls and stops before any misspecified cell if one of them fails the limits
of Table 5.3. The main grids were frozen on 7 October 2026 from control-only
screens, before any misspecified run; `freeze.json` in the archive records them.

## Layout

```
src/pinn_misspec/
  training.py              configuration checks, the network, Adam with cosine decay, checkpoints
  runner.py                designs, run ids, saved rows, the control gate, worker processes
  oscillator/              model.py (closed form, trial state, one run),
  advection_diffusion/       targets.py (theta_eq, theta_sol, metrics), study.py (designs, hooks)
  report/                  the thesis's tables, figures and diagnostics, from saved results
studies/                   one script per group of thesis studies
```

## Notation

The thesis notation differs from the names in the code and the saved tables,
which are kept because each run is identified by a hash of its configuration.

| Thesis | Code and saved tables |
|---|---|
| `ω`, `ω†` | `w0`, `w0_true` |
| `γ`, `γ†`, `γ_s` | `g`, `g_true`, `gamma_imposed` |
| `a`, `κ`, `μ` | `velocity`, `diffusivity`, `sink` (`sink_rate`) |
| `θ_sol(s)` | `<c>_target`, references of kind `target` |
| `θ_eq(s)` | `<c>_residual_target`, references of kind `residual_target` |
| `θ̂_sol(s)` | `<c>_sol`, references of kind `solution_fit` |
| `Λ`, `λ` | `scaled_physics_weight`, `physics_weight` |
| position `p` | `<c>_position`, computed in `report/frame.py` |

## Reproducibility

A run is identified by the first 12 hex digits of the SHA-256 of its full
configuration, so a configuration key may not be renamed or removed: a few keys
the studies no longer read are kept for that reason (`INERT` in each `model.py`).
Runs recorded with JAX 0.11.2, the pinned version, are reproduced bit for bit;
runs recorded before 15 October 2026 used JAX 0.11.1 and agree with reruns to
about 1e-4 relative. With `SOURCE_DATE_EPOCH` set, the figure PDFs are
byte-reproducible.

## Licence

MIT; see `LICENSE`. To cite, see `CITATION.cff`.
