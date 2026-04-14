from __future__ import annotations

import argparse
from itertools import product
import json
import pickle
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Tuple

from standardizedarchive.openfold_zarr_archive import (
    OpenFoldRunResult,
    OpenFoldZarrArchive,
    score_from_output_dict,
    select_best_entries,
)


def _flag_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"--{name}" if value else ""
    return f"--{name} {value}"


def _normalize_for_id(value: Any) -> str:
    safe = str(value).replace(" ", "_").replace("/", "_")
    return safe[:80]


def expand_grid(param_grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = sorted(param_grid.keys())
    value_lists = [param_grid[key] for key in keys]
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def _find_output_pickle(run_output_dir: Path) -> Path:
    candidates = sorted(run_output_dir.rglob("*_output_dict.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_output_dict.pkl found under {run_output_dir}. "
            "Ensure the command enables --save_outputs."
        )
    return candidates[-1]


def _run_single_openfold_command(
    base_command: str,
    params: Dict[str, Any],
    run_output_dir: Path,
) -> Tuple[subprocess.CompletedProcess[str], str]:
    attn_map_dir = run_output_dir / "attention_maps"
    attn_map_dir.mkdir(parents=True, exist_ok=True)

    flags = [_flag_value(name, value) for name, value in sorted(params.items())]
    flags = [flag for flag in flags if flag]

    command = " ".join(
        [
            base_command,
            "--save_outputs",
            f"--output_dir {run_output_dir}",
            f"--attn_map_dir {attn_map_dir}",
            *flags,
        ]
    )

    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    return completed, command


def _collect_result(
    run_id: str,
    params: Dict[str, Any],
    run_output_dir: Path,
    command: str,
    score_key: str,
    changed_param: str | None = None,
    from_value: Any | None = None,
    to_value: Any | None = None,
    score_delta: float | None = None,
    step_index: int | None = None,
) -> OpenFoldRunResult:
    output_pickle = _find_output_pickle(run_output_dir)
    with output_pickle.open("rb") as handle:
        output_dict = pickle.load(handle)

    score = score_from_output_dict(output_dict, score_key=score_key)
    return OpenFoldRunResult(
        run_id=run_id,
        score=score,
        params=dict(params),
        output_dir=str(run_output_dir),
        command=command,
        model_output_path=str(output_pickle),
        changed_param=changed_param,
        from_value=from_value,
        to_value=to_value,
        score_delta=score_delta,
        step_index=step_index,
    )


def run_sweep(
    base_command: str,
    param_grid: Dict[str, List[Any]],
    runs_root: Path,
    score_key: str,
) -> List[OpenFoldRunResult]:
    results: List[OpenFoldRunResult] = []
    failures: List[Tuple[str, int, str, str]] = []

    for idx, params in enumerate(expand_grid(param_grid), start=1):
        run_id_parts = [f"{k}-{_normalize_for_id(v)}" for k, v in sorted(params.items())]
        run_id = f"run-{idx:03d}__" + "__".join(run_id_parts)

        run_output_dir = runs_root / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        completed, command = _run_single_openfold_command(base_command, params, run_output_dir)
        if completed.returncode != 0:
            failures.append(
                (
                    run_id,
                    completed.returncode,
                    completed.stderr.strip(),
                    completed.stdout.strip(),
                )
            )
            continue

        results.append(
            _collect_result(
                run_id=run_id,
                params=params,
                run_output_dir=run_output_dir,
                command=command,
                score_key=score_key,
            )
        )

    if failures:
        print(f"[openfold-sweep] failed runs: {len(failures)}")
        for run_id, returncode, stderr, stdout in failures:
            print(f"[openfold-sweep] run={run_id} returncode={returncode}")
            if stderr:
                print("[openfold-sweep] stderr:")
                print(stderr[-2000:])
            elif stdout:
                print("[openfold-sweep] stdout:")
                print(stdout[-2000:])

    if not results:
        raise RuntimeError(
            "All OpenFold sweep runs failed. Check the per-run stderr summaries above."
        )

    return results


def run_incremental_sweep(
    base_command: str,
    param_grid: Dict[str, List[Any]],
    runs_root: Path,
    score_key: str,
) -> Tuple[List[OpenFoldRunResult], List[OpenFoldRunResult], OpenFoldRunResult]:
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter")

    ordered_keys = sorted(param_grid.keys())
    for key in ordered_keys:
        values = param_grid[key]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Parameter '{key}' must map to a non-empty list")

    baseline_params = {key: param_grid[key][0] for key in ordered_keys}
    all_results: List[OpenFoldRunResult] = []
    best_increment_entries: List[OpenFoldRunResult] = []
    failures: List[Tuple[str, int, str, str]] = []

    baseline_run_id = "run-000__baseline"
    baseline_output_dir = runs_root / baseline_run_id
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_completed, baseline_command = _run_single_openfold_command(
        base_command,
        baseline_params,
        baseline_output_dir,
    )
    if baseline_completed.returncode != 0:
        raise RuntimeError(
            "Baseline incremental run failed. stderr:\n"
            + baseline_completed.stderr[-2000:]
        )

    current_best = _collect_result(
        run_id=baseline_run_id,
        params=baseline_params,
        run_output_dir=baseline_output_dir,
        command=baseline_command,
        score_key=score_key,
        step_index=0,
    )
    all_results.append(current_best)

    run_counter = 1
    for step_index, key in enumerate(ordered_keys, start=1):
        current_value = current_best.params[key]
        candidates = [value for value in param_grid[key] if value != current_value]

        best_trial: OpenFoldRunResult | None = None
        for candidate in candidates:
            trial_params = dict(current_best.params)
            trial_params[key] = candidate

            run_id = f"run-{run_counter:03d}__step-{step_index:02d}__{key}-{_normalize_for_id(candidate)}"
            run_counter += 1

            run_output_dir = runs_root / run_id
            run_output_dir.mkdir(parents=True, exist_ok=True)

            completed, command = _run_single_openfold_command(base_command, trial_params, run_output_dir)
            if completed.returncode != 0:
                failures.append(
                    (
                        run_id,
                        completed.returncode,
                        completed.stderr.strip(),
                        completed.stdout.strip(),
                    )
                )
                continue

            trial_result = _collect_result(
                run_id=run_id,
                params=trial_params,
                run_output_dir=run_output_dir,
                command=command,
                score_key=score_key,
                changed_param=key,
                from_value=current_value,
                to_value=candidate,
                step_index=step_index,
            )
            all_results.append(trial_result)

            if best_trial is None or trial_result.score > best_trial.score:
                best_trial = trial_result

        if best_trial is None:
            continue

        score_delta = best_trial.score - current_best.score
        if score_delta > 0:
            improved = OpenFoldRunResult(
                run_id=best_trial.run_id,
                score=best_trial.score,
                params=best_trial.params,
                output_dir=best_trial.output_dir,
                command=best_trial.command,
                model_output_path=best_trial.model_output_path,
                changed_param=best_trial.changed_param,
                from_value=best_trial.from_value,
                to_value=best_trial.to_value,
                score_delta=score_delta,
                step_index=best_trial.step_index,
            )
            best_increment_entries.append(improved)
            current_best = improved

    if failures:
        print(f"[openfold-sweep] failed runs: {len(failures)}")
        for run_id, returncode, stderr, stdout in failures:
            print(f"[openfold-sweep] run={run_id} returncode={returncode}")
            if stderr:
                print("[openfold-sweep] stderr:")
                print(stderr[-2000:])
            elif stdout:
                print("[openfold-sweep] stdout:")
                print(stdout[-2000:])

    if len(all_results) == 1 and not best_increment_entries:
        print("[openfold-sweep] no successful incremental candidate runs; baseline only")

    return all_results, best_increment_entries, current_best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenFold parameter sweep and archive best entries in Zarr")
    parser.add_argument("--base_command", required=True, help="Base command used to launch OpenFold runs")
    parser.add_argument("--grid_json", required=True, help="JSON file mapping parameter names to candidate values")
    parser.add_argument("--runs_root", default="outputs/sweep_runs", help="Directory for per-run outputs")
    parser.add_argument("--archive_path", default="standardizedarchive/openfold_best_runs.zarr", help="Zarr archive output path")
    parser.add_argument("--best_log_path", default="standardizedarchive/best_entries.jsonl", help="Best entries log path")
    parser.add_argument("--top_k", type=int, default=1, help="Number of best entries to keep")
    parser.add_argument("--score_key", default="plddt", help="Output dict key used for scoring")
    parser.add_argument(
        "--sweep_strategy",
        choices=["incremental", "grid"],
        default="incremental",
        help="Sweep strategy: incremental one-parameter-at-a-time, or full grid search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.grid_json, "r", encoding="utf-8") as handle:
        param_grid = json.load(handle)

    if not isinstance(param_grid, dict):
        raise ValueError("grid_json must contain an object mapping parameter names to value lists")

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.sweep_strategy == "grid":
        results = run_sweep(
            base_command=args.base_command,
            param_grid=param_grid,
            runs_root=runs_root,
            score_key=args.score_key,
        )
        best_entries = select_best_entries(results, top_k=args.top_k)
    else:
        results, best_increment_entries, final_best = run_incremental_sweep(
            base_command=args.base_command,
            param_grid=param_grid,
            runs_root=runs_root,
            score_key=args.score_key,
        )
        if best_increment_entries:
            best_entries = best_increment_entries
        else:
            best_entries = [final_best]

    archive = OpenFoldZarrArchive(args.archive_path)
    archive.root.attrs["sweep_strategy"] = args.sweep_strategy
    archive.root.attrs["score_key"] = args.score_key
    archive.append_best_entries(best_entries)
    archive.write_best_log(best_entries, args.best_log_path)


if __name__ == "__main__":
    main()
