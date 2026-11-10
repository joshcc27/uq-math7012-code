"""Run a study design: run ids, saved rows, the control gate and worker processes.

A run is identified by a hash of its full configuration, and its trace and row are
written as soon as it finishes, so a restarted study skips what is recorded. The
misspecified cells wait until every control of the design has passed.

Workers are single-threaded subprocesses fed through ``<output>/.staging``, because
``multiprocessing`` pools failed with invalid handles on Windows. Only the parent
writes rows.
"""

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import numpy as np

from pinn_misspec.paths import RESULTS

DESIGN_KEYS = ("severities", "scaled_weights", "noise_seeds", "network_seeds")
# Configuration entries a design sets itself, which overrides may not touch.
DESIGN_AXES = ("severity", "physics_weight", "noise_seed", "network_seed")
SINGLE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def design_cells(design):
    return list(itertools.product(*(design[key] for key in DESIGN_KEYS)))


def control_slice(design):
    return {**design, "severities": [s for s in design["severities"] if float(s) == 0.0]}


def is_control(cell):
    return float(cell[0]) == 0.0


def check_overrides(overrides, axes=DESIGN_AXES):
    overrides = dict(overrides or {})
    clash = sorted(set(overrides) & set(axes))
    if clash:
        raise ValueError(f"Overrides may not set the design axes {clash}")
    return overrides


def config_hash(mapping):
    canonical = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in mapping.items()
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def run_id(config):
    return config_hash(config)


def number_token(value):
    """Format a number for a filename: ``-0.5`` becomes ``m0p5``."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p").replace("+", "")


def format_value(value):
    """Format a CSV value: empty for ``None``, round-tripping precision for floats."""
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


def read_rows(path):
    """Return the header and the complete rows of a CSV file.

    A row cut short by an interrupted write has missing fields and is left out,
    so its run counts as not done.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if None not in row and None not in row.values()]
        return list(reader.fieldnames or []), rows


def write_rows(path, header, rows):
    """Replace a CSV file atomically with the given header and rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def drop_incomplete_rows(path):
    """Remove a final row cut short by an interrupted write; return how many went.

    A last row without its line ending counts as cut short even with every
    field present, since its last value may be partial.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        terminated = handle.read(1) == b"\n"
    with path.open(newline="", encoding="utf-8") as handle:
        total = sum(1 for _ in csv.DictReader(handle))
    header, rows = read_rows(path)
    if not terminated and len(rows) == total and rows:
        rows = rows[:-1]
    if len(rows) != total or not terminated:
        write_rows(path, header, rows)
    return total - len(rows)


def append_row(path, row):
    """Append one row, writing the header on first use.

    A row whose columns differ from the file's means the code changed under an
    existing study, so it raises ``RuntimeError`` instead.
    """
    path = Path(path)
    values = {key: format_value(value) for key, value in row.items()}
    header, _ = read_rows(path)
    if not header:
        write_rows(path, list(row), [values])
        return
    if set(header) != set(row):
        raise RuntimeError(
            f"{path} has different columns from this run; "
            "start a new output directory for the changed code."
        )
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=header).writerow(values)
        handle.flush()
        os.fsync(handle.fileno())


def merge_rows(path, rows, key):
    """Write ``rows`` into a CSV, replacing rows with the same ``key`` and keeping the rest."""
    path = Path(path)
    header, existing = read_rows(path)
    merged = {key(row): row for row in existing}
    for row in rows:
        values = {name: format_value(value) for name, value in row.items()}
        merged[key(values)] = values
    if merged:
        write_rows(path, header or list(rows[0]), list(merged.values()))


def _true(value):
    """Read a boolean that may have come back from a CSV as text."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def control_row_failures(row, limits, domain_flag):
    """Return the reasons a control row fails its acceptance limits, or an empty list.

    Args:
        row: One recorded run.
        limits: Maps a metric column to its largest admissible value.
        domain_flag: The column flagging an estimate outside Theta; a missing
            flag counts as a failure.
    """
    failures = []
    for name, limit in limits.items():
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{name} is missing or non-numeric")
            continue
        if not np.isfinite(value) or value > limit:
            failures.append(f"{name}={value:g} exceeds {limit:g}")
    if _true(row.get(domain_flag, True)):
        failures.append("an estimate is outside Theta")
    return failures


def control_acceptance(study_csv, expected_ids, limits, domain_flag):
    """Assess the control runs ``expected_ids`` recorded in a study CSV.

    Returns:
        ``passed``, the ``missing`` ids, the per-run ``failures`` and the
        number ``checked``.
    """
    expected = set(expected_ids)
    rows = {row["run_id"]: row for row in read_rows(Path(study_csv))[1]}
    missing = sorted(expected - set(rows))
    failures = {
        rid: problems
        for rid in sorted(expected & set(rows))
        if (problems := control_row_failures(rows[rid], limits, domain_flag))
    }
    return {
        "passed": bool(expected) and not missing and not failures,
        "missing": missing,
        "failures": failures,
        "checked": len(expected),
    }


def print_control_acceptance(assessment, limits):
    limits_text = ", ".join(f"{name}<={limit:g}" for name, limit in limits.items())
    status = "PASS" if assessment["passed"] else "FAIL"
    print(
        f"Control acceptance: {status} over {assessment['checked']} run(s) "
        f"({limits_text}; inside Theta).",
        flush=True,
    )
    if assessment["missing"]:
        print(f"  Missing {len(assessment['missing'])} control run(s).")
    for rid, problems in assessment["failures"].items():
        print(f"  {rid}: {'; '.join(problems)}")


class ControlGateError(RuntimeError):
    """Raised when a design's control fails, before any misspecified cell runs."""

    def __init__(self, assessment):
        super().__init__(
            f"{len(assessment['failures'])} control run(s) failed and "
            f"{len(assessment['missing'])} are missing; no misspecified cell was run"
        )
        self.assessment = assessment


def _default_reference_key(cell):
    return float(cell[0]), int(cell[2])


def _default_cell_fields(cell):
    return {"scaled_physics_weight": float(cell[1])}


@dataclass(frozen=True)
class StudyHooks:
    """What one kind of study supplies to ``run_study``.

    ``compute_references(design, output, overrides)`` returns a mapping keyed by
    ``reference_key(cell)``. ``network_record(result)`` returns extra row
    entries and arrays for saved networks. ``source`` names the module-level
    factory of these hooks, ``(module, function)``, so a worker process can
    rebuild them.
    """

    study_config: Callable
    compute_references: Callable
    run_experiment: Callable
    summarise_run: Callable
    study_metrics: Callable
    network_record: Callable
    describe_run: Callable
    control_limits: dict
    domain_flag: str
    protocol: str
    source: tuple
    reference_key: Callable = _default_reference_key
    cell_fields: Callable = _default_cell_fields


def study_config(model, severity, scaled_weight, noise_seed, network_seed, overrides=None):
    """Return the configuration of one cell of ``model``; ``overrides`` may not set a design axis."""
    overrides = check_overrides(overrides)
    return model.default_config(
        **overrides,
        severity=float(severity),
        physics_weight=model.raw_physics_weight(scaled_weight, model.default_config(**overrides)),
        noise_seed=int(noise_seed),
        network_seed=int(network_seed),
    )


def _run_cell(hooks, cell, config, rid, references, network_dir=None):
    """Train one cell and return ``(rid, row, trace, description)``.

    With ``network_dir`` set, the trained network and the arrays of
    ``hooks.network_record`` are saved there, and its entries join the row.
    """
    result = hooks.run_experiment(config)
    metrics = hooks.study_metrics(result, references)
    if network_dir is not None:
        extra, arrays = hooks.network_record(result)
        metrics = {**metrics, **extra}
        network_dir = Path(network_dir)
        network_dir.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(network_dir / f"{rid}.eqx", result["net"])
        np.savez(network_dir / f"{rid}.npz", **arrays)
    row = {
        "run_id": rid,
        **hooks.cell_fields(cell),
        **hooks.summarise_run(result),
        **metrics,
        "trace_file": f"traces/{rid}.npy",
    }
    description = f"{hooks.describe_run(cell, result, metrics)} {result['elapsed_seconds']:.1f}s"
    return rid, row, np.asarray(result["trace"]), description


_WORKER_HOOKS = {}


def _worker_hooks(source):
    """Rebuild a study's hooks once per worker process."""
    if source not in _WORKER_HOOKS:
        module, factory = source
        _WORKER_HOOKS[source] = getattr(importlib.import_module(module), factory)()
    return _WORKER_HOOKS[source]


def _patiently(action, *args, attempts=40, pause=0.25):
    """Retry a file operation while another process, such as a virus scanner, briefly locks it."""
    for attempt in range(attempts):
        try:
            return action(*args)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def _read_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_atomically(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def worker_loop(staging):
    """Claim and run staged cells until none is left.

    A task is claimed by renaming it to a name unique to this process, which
    succeeds for exactly one worker. The claim is removed only after the result
    is written, so an interrupted worker's cell is requeued.
    """
    staging = Path(staging)
    tasks, results = staging / "tasks", staging / "results"
    suffix = f".claimed-{os.getpid()}"
    while True:
        claimed = None
        for task in sorted(tasks.glob("*.task")):
            try:
                os.rename(task, task.with_name(task.name + suffix))
            except OSError:
                continue
            claimed = task.with_name(task.name + suffix)
            break
        if claimed is None:
            return
        try:
            source, cell, config, rid, references, network_dir = _patiently(_read_pickle, claimed)
        except FileNotFoundError:
            # Seen on Windows under load: the claim vanished and the cell went
            # to another worker. The parent records each cell at most once.
            continue
        try:
            hooks = _worker_hooks(source)
            payload = ("ok", _run_cell(hooks, cell, config, rid, references, network_dir))
        except Exception:
            payload = ("error", (rid, traceback.format_exc()))
        _patiently(_write_atomically, results / f"{rid}.result", payload)
        _patiently(claimed.unlink)


class WorkerPool:
    """Single-threaded worker subprocesses fed through ``<output>/.staging``."""

    RESTARTS = 5

    def __init__(self, output, workers):
        self.staging = Path(output) / ".staging"
        shutil.rmtree(self.staging, ignore_errors=True)
        (self.staging / "tasks").mkdir(parents=True)
        (self.staging / "results").mkdir()
        self.workers = int(workers)
        self.processes = []

    def _alive(self):
        return [process for process in self.processes if process.poll() is None]

    def _start(self, wanted):
        flags = os.environ.get("XLA_FLAGS", "")
        if SINGLE_THREAD_FLAGS not in flags:
            flags = f"{flags} {SINGLE_THREAD_FLAGS}".strip()
        command = [sys.executable, "-m", "pinn_misspec.runner", "--worker", str(self.staging)]
        for _ in range(max(0, wanted - len(self._alive()))):
            self.processes.append(subprocess.Popen(command, env={**os.environ, "XLA_FLAGS": flags}))

    def _collect(self, expected, record):
        for path in sorted((self.staging / "results").glob("*.result")):
            status, payload = _patiently(_read_pickle, path)
            _patiently(path.unlink)
            if status == "error":
                rid, text = payload
                raise RuntimeError(f"run {rid} failed in a worker:\n{text}")
            if payload[0] in expected:
                expected.discard(payload[0])
                record(*payload)

    def run(self, jobs, record):
        """Stage ``(source, cell, config, rid, references, network_dir)`` jobs and record each.

        If every worker exits with cells unfinished, their claims are requeued
        and the workers restarted, at most ``RESTARTS`` times.
        """
        tasks = self.staging / "tasks"
        expected = set()
        for job in jobs:
            _write_atomically(tasks / f"{job[3]}.task", job)
            expected.add(job[3])
        restarts = 0
        self._start(min(self.workers, len(expected)))
        while expected:
            self._collect(expected, record)
            if expected and not self._alive():
                self._collect(expected, record)
                if not expected:
                    break
                for claimed in tasks.glob("*.task.claimed-*"):
                    _patiently(
                        os.replace, claimed, claimed.with_name(claimed.name.split(".claimed-")[0])
                    )
                restarts += 1
                if restarts > self.RESTARTS:
                    raise RuntimeError("worker processes keep exiting with cells unfinished")
                print(f"All workers exited with {len(expected)} cell(s) left; restarting them.")
                self._start(min(self.workers, len(expected)))
            time.sleep(0.25)

    def close(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(self.staging, ignore_errors=True)


def run_study(design, output, hooks, overrides=None, gate=True, workers=1, save_networks=False):
    """Run every pending cell of a design, recording each as it finishes.

    The references are computed first, then the zero-severity cells. With
    ``gate`` set, a failing control raises ``ControlGateError`` before any
    misspecified cell starts, leaving the control rows recorded.

    Args:
        design: The design to run.
        output: Directory for ``study.csv``, ``traces/`` and the references.
        hooks: The ``StudyHooks`` of the model and kind of study.
        overrides: Configuration entries applied to every run and reference.
        gate: Whether the controls must pass before the misspecified cells run.
        workers: Worker processes; 1 trains in this process.
        save_networks: Save each trained network under ``networks/`` and add
            ``hooks.network_record`` to its row.

    Returns:
        Counts of cells ``planned``, ``skipped`` (already recorded) and ``run``.
    """
    if save_networks and hooks.network_record is None:
        raise ValueError("saving networks needs hooks with a network_record")
    if workers > 1 and hooks.source is None:
        raise ValueError("worker processes need hooks with a source")
    output = Path(output)
    overrides = check_overrides(overrides)
    (output / "traces").mkdir(parents=True, exist_ok=True)
    study_csv = output / "study.csv"

    cells = []
    for cell in design_cells(design):
        config = hooks.study_config(*cell, overrides=overrides)
        cells.append((cell, config, run_id(config)))
    control = [entry for entry in cells if is_control(entry[0])]
    rest = [entry for entry in cells if not is_control(entry[0])]
    if gate and rest and not control:
        raise ValueError("a gated design needs a zero-severity control slice")

    print(f"Protocol {hooks.protocol}; writing to {output}", flush=True)
    references = hooks.compute_references(design, output, overrides)
    removed = drop_incomplete_rows(study_csv)
    if removed:
        print(f"Removed {removed} incomplete row(s) left by an interrupted write.")
    done = {row["run_id"] for row in read_rows(study_csv)[1]}
    pending = [entry for entry in cells if entry[2] not in done]
    if save_networks:
        unsaved = [
            rid
            for _, _, rid in cells
            if rid in done and not (output / "networks" / f"{rid}.eqx").exists()
        ]
        if unsaved:
            raise ValueError(
                f"{len(unsaved)} run(s) are already recorded without saved networks; "
                "use a new output directory to save networks"
            )
    print(
        f"{len(cells)} runs in the design ({len(control)} control), "
        f"{len(cells) - len(pending)} already recorded, {len(pending)} to run.",
        flush=True,
    )

    progress = {"index": 0, "total": len(pending), "started": time.perf_counter()}

    def record(rid, row, trace, description):
        # Trace first, so a recorded run always has one.
        np.save(output / "traces" / f"{rid}.npy", trace)
        append_row(study_csv, row)
        done.add(rid)
        progress["index"] += 1
        index, total = progress["index"], progress["total"]
        left = (time.perf_counter() - progress["started"]) / index * (total - index)
        print(f"[{index:4d}/{total}] {description}, about {left / 60:.1f} min left", flush=True)

    pool = WorkerPool(output, workers) if workers > 1 and pending else None
    network_dir = output / "networks" if save_networks else None

    def run_pending(entries):
        entries = [entry for entry in entries if entry[2] not in done]
        jobs = [
            (hooks.source, cell, config, rid, references[hooks.reference_key(cell)], network_dir)
            for cell, config, rid in entries
        ]
        if pool is None:
            for job in jobs:
                record(*_run_cell(hooks, *job[1:]))
        elif jobs:
            pool.run(jobs, record)

    try:
        run_pending(control)
        if gate and rest:
            assessment = control_acceptance(
                study_csv, [rid for _, _, rid in control], hooks.control_limits, hooks.domain_flag
            )
            print_control_acceptance(assessment, hooks.control_limits)
            if not assessment["passed"]:
                raise ControlGateError(assessment)
        run_pending(rest)
    finally:
        if pool is not None:
            pool.close()
    return {"planned": len(cells), "skipped": len(cells) - len(pending), "run": progress["index"]}


def study_arguments(description):
    """Parse the options every study script accepts.

    ``--steps`` replaces every budget in the script, for a quick check of the
    pipeline; it writes under ``--output``, which it therefore requires, so a
    shortened run never mixes with real results.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="single-threaded worker processes (default 1, in this process)",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="replace every training budget, for a quick check"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="results root to write under instead of the default",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="PART", help="run only these parts of the script"
    )
    arguments = parser.parse_args()
    if arguments.steps is not None and arguments.output is None:
        parser.error("--steps needs --output, so a shortened run never mixes with real results")
    return arguments


def run_parts(parts, arguments):
    """Run the selected parts of a study script, in order.

    Args:
        parts: Maps each part's name to a function of ``(steps, root, workers)``,
            where ``steps`` is the budget override or ``None`` and ``root`` the
            results root.
        arguments: The parsed options of ``study_arguments``.
    """
    unknown = set(arguments.only or ()) - set(parts)
    if unknown:
        raise SystemExit(f"unknown parts {sorted(unknown)}; choose from {sorted(parts)}")
    root = arguments.output or RESULTS
    for name, part in parts.items():
        if arguments.only is None or name in arguments.only:
            print(f"\n=== {name} ===", flush=True)
            try:
                part(arguments.steps, root, max(1, arguments.workers))
            except ControlGateError as error:
                raise SystemExit(
                    f"{name}: stopped before the misspecified cells: {error}"
                ) from None


if __name__ == "__main__":
    # A worker subprocess started by ``WorkerPool``.
    worker_parser = argparse.ArgumentParser(description="Run staged study cells.")
    worker_parser.add_argument("--worker", type=Path, required=True)
    worker_loop(worker_parser.parse_args().worker)
