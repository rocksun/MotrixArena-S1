"""Rank section01 PPO runs by their furthest logged course distance.

The metric is written by vbot_section01_np.py as TensorBoard scalar
``metrics / course_distance (max)``.  It is the greatest y-direction progress
relative to an episode's own randomized start point, in metres.

Example:
    uv run --no-sync scripts/report_section01_max_distance.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "vbot_navigation_section01"
DISTANCE_TAG = "metrics / course_distance (max)"
AVERAGE_DISTANCE_TAG = "metrics / course_distance (mean)"
CACHE_FILE_NAME = ".section01_max_distance_cache.json"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
CACHE_VERSION = 1


@dataclass(frozen=True)
class RunResult:
    directory: Path
    distance_m: float
    step: int
    best_checkpoint_step: int | None
    best_checkpoint_average_distance_m: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank section01 PPO TensorBoard runs by their furthest course distance."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"PPO run root (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--tag",
        default=DISTANCE_TAG,
        help=f"TensorBoard scalar tag to inspect (default: {DISTANCE_TAG!r})",
    )
    parser.add_argument(
        "--mean-tag",
        default=AVERAGE_DISTANCE_TAG,
        help=f"TensorBoard scalar tag for the best checkpoint average (default: {AVERAGE_DISTANCE_TAG!r})",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def event_directories(runs_dir: Path) -> list[Path]:
    """Return PPO directories that contain TensorBoard event files."""
    return sorted({event_file.parent for event_file in runs_dir.rglob("events.out.tfevents.*")})


def best_agent_file(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "checkpoints").glob("best_agent.*"))
    return files[0] if files else None


def file_fingerprint(path: Path, root: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def cache_fingerprint(run_dir: Path) -> dict[str, object] | None:
    """Fingerprint only completed, older runs that are worth caching."""
    best_file = best_agent_file(run_dir)
    if best_file is None or time.time() - best_file.stat().st_mtime < CACHE_MAX_AGE_SECONDS:
        return None
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    return {
        "best_agent": file_fingerprint(best_file, run_dir),
        "event_files": [file_fingerprint(event_file, run_dir) for event_file in event_files],
    }


def load_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache.get("version") == CACHE_VERSION and isinstance(cache.get("entries"), dict):
            return cache
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": CACHE_VERSION, "entries": {}}


def write_cache(path: Path, cache: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def cache_key(run_dir: Path, runs_dir: Path) -> str:
    return str(run_dir.relative_to(runs_dir))


def result_from_cache(data: dict[str, object], directory: Path) -> RunResult:
    return RunResult(
        directory=directory,
        distance_m=float(data["distance_m"]),
        step=int(data["step"]),
        best_checkpoint_step=(
            int(data["best_checkpoint_step"]) if data["best_checkpoint_step"] is not None else None
        ),
        best_checkpoint_average_distance_m=(
            float(data["best_checkpoint_average_distance_m"])
            if data["best_checkpoint_average_distance_m"] is not None
            else None
        ),
    )


def result_for_cache(result: RunResult) -> dict[str, float | int | None]:
    return {
        "distance_m": result.distance_m,
        "step": result.step,
        "best_checkpoint_step": result.best_checkpoint_step,
        "best_checkpoint_average_distance_m": result.best_checkpoint_average_distance_m,
    }


def best_checkpoint_step(run_dir: Path) -> int | None:
    """Match best_agent.pt to its identical numbered checkpoint safely.

    SKRL's best checkpoint filename has no step number.  Comparing only model
    tensors with PyTorch's ``weights_only`` mode recovers that step without
    executing any pickle data from the checkpoint.
    """
    checkpoint_dir = run_dir / "checkpoints"
    best_file = checkpoint_dir / "best_agent.pt"
    if not best_file.is_file():
        return None
    try:
        import torch

        best_policy = torch.load(best_file, map_location="cpu", weights_only=True).get("policy")
        if best_policy is None:
            return None
        for candidate in checkpoint_dir.glob("agent_*.pt"):
            match = re.fullmatch(r"agent_(\d+)\.pt", candidate.name)
            if match is None:
                continue
            candidate_policy = torch.load(candidate, map_location="cpu", weights_only=True).get("policy")
            if candidate_policy is None or candidate_policy.keys() != best_policy.keys():
                continue
            if all(torch.equal(best_policy[key], candidate_policy[key]) for key in best_policy):
                return int(match.group(1))
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    return None


def scalar_at_step(accumulator, tag: str, step: int) -> float | None:
    values = [event.value for event in accumulator.Scalars(tag) if event.step == step]
    return values[-1] if values else None


def read_furthest_distance(run_dir: Path, tag: str, mean_tag: str) -> RunResult | None:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is required. Run this script with the project's uv environment."
        ) from error

    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    maximum = max(accumulator.Scalars(tag), key=lambda event: event.value)
    checkpoint_step = best_checkpoint_step(run_dir)
    average_distance = (
        scalar_at_step(accumulator, mean_tag, checkpoint_step)
        if checkpoint_step is not None and mean_tag in accumulator.Tags().get("scalars", [])
        else None
    )
    return RunResult(
        directory=run_dir,
        distance_m=maximum.value,
        step=maximum.step,
        best_checkpoint_step=checkpoint_step,
        best_checkpoint_average_distance_m=average_distance,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_results(results: list[RunResult]) -> None:
    if not results:
        return
    name_width = max(len(display_path(result.directory)) for result in results)
    print(
        f"{'PPO directory':<{name_width}}  {'furthest distance (m)':>21}  {'step':>12}  "
        f"{'best checkpoint step':>20}  {'best avg distance (m)':>21}"
    )
    print(f"{'-' * name_width}  {'-' * 21}  {'-' * 12}  {'-' * 20}  {'-' * 21}")
    for result in results:
        checkpoint_step = f"{result.best_checkpoint_step:,}" if result.best_checkpoint_step is not None else "N/A"
        average_distance = (
            f"{result.best_checkpoint_average_distance_m:.3f}"
            if result.best_checkpoint_average_distance_m is not None
            else "N/A"
        )
        print(
            f"{display_path(result.directory):<{name_width}}  "
            f"{result.distance_m:>21.3f}  {result.step:>12,}  "
            f"{checkpoint_step:>20}  {average_distance:>21}"
        )


def write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "ppo_directory",
                "furthest_distance_m",
                "furthest_distance_step",
                "best_checkpoint_step",
                "best_checkpoint_average_distance_m",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ppo_directory": display_path(result.directory),
                    "furthest_distance_m": f"{result.distance_m:.6f}",
                    "furthest_distance_step": result.step,
                    "best_checkpoint_step": result.best_checkpoint_step,
                    "best_checkpoint_average_distance_m": (
                        f"{result.best_checkpoint_average_distance_m:.6f}"
                        if result.best_checkpoint_average_distance_m is not None
                        else ""
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"Runs directory does not exist: {runs_dir}", file=sys.stderr)
        return 2

    directories = event_directories(runs_dir)
    if not directories:
        print(f"No TensorBoard event files found under: {runs_dir}", file=sys.stderr)
        return 2

    results: list[RunResult] = []
    missing_metric: list[Path] = []
    cache_path = runs_dir / CACHE_FILE_NAME
    cache = load_cache(cache_path)
    cache_entries = cache["entries"]
    assert isinstance(cache_entries, dict)
    cache_hits = 0
    cache_updates = 0
    for run_dir in directories:
        fingerprint = cache_fingerprint(run_dir)
        key = cache_key(run_dir, runs_dir)
        cached_entry = cache_entries.get(key) if fingerprint is not None else None
        if (
            isinstance(cached_entry, dict)
            and cached_entry.get("fingerprint") == fingerprint
            and cached_entry.get("tag") == args.tag
            and cached_entry.get("mean_tag") == args.mean_tag
            and isinstance(cached_entry.get("result"), dict)
        ):
            results.append(result_from_cache(cached_entry["result"], run_dir))
            cache_hits += 1
            continue
        try:
            result = read_furthest_distance(run_dir, args.tag, args.mean_tag)
        except Exception as error:
            print(f"Could not read {display_path(run_dir)}: {error}", file=sys.stderr)
            continue
        if result is None:
            missing_metric.append(run_dir)
        else:
            results.append(result)
            if fingerprint is not None:
                cache_entries[key] = {
                    "fingerprint": fingerprint,
                    "tag": args.tag,
                    "mean_tag": args.mean_tag,
                    "result": result_for_cache(result),
                }
                cache_updates += 1

    results.sort(key=lambda result: result.distance_m, reverse=True)
    print_results(results)
    if cache_updates:
        write_cache(cache_path, cache)
    if cache_hits or cache_updates:
        print(f"\nCache: {cache_hits} hit(s), {cache_updates} created/refreshed ({cache_path.name})")
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote {len(results)} row(s) to {args.csv}")
    if missing_metric:
        print(
            f"\nSkipped {len(missing_metric)} run(s) without scalar tag {args.tag!r}. "
            "Runs trained before course_distance was added cannot be ranked by distance.",
            file=sys.stderr,
        )
    if not results:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
