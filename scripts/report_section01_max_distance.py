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
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "vbot_navigation_section01"
DISTANCE_TAG = "metrics / course_distance (max)"


@dataclass(frozen=True)
class RunResult:
    directory: Path
    distance_m: float
    step: int


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
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def event_directories(runs_dir: Path) -> list[Path]:
    """Return PPO directories that contain TensorBoard event files."""
    return sorted({event_file.parent for event_file in runs_dir.rglob("events.out.tfevents.*")})


def read_furthest_distance(run_dir: Path, tag: str) -> RunResult | None:
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
    return RunResult(directory=run_dir, distance_m=maximum.value, step=maximum.step)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_results(results: list[RunResult]) -> None:
    if not results:
        return
    name_width = max(len(display_path(result.directory)) for result in results)
    print(f"{'PPO directory':<{name_width}}  {'furthest distance (m)':>21}  {'step':>12}")
    print(f"{'-' * name_width}  {'-' * 21}  {'-' * 12}")
    for result in results:
        print(
            f"{display_path(result.directory):<{name_width}}  "
            f"{result.distance_m:>21.3f}  {result.step:>12,}"
        )


def write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("ppo_directory", "furthest_distance_m", "step"))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ppo_directory": display_path(result.directory),
                    "furthest_distance_m": f"{result.distance_m:.6f}",
                    "step": result.step,
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
    for run_dir in directories:
        try:
            result = read_furthest_distance(run_dir, args.tag)
        except Exception as error:
            print(f"Could not read {display_path(run_dir)}: {error}", file=sys.stderr)
            continue
        if result is None:
            missing_metric.append(run_dir)
        else:
            results.append(result)

    results.sort(key=lambda result: result.distance_m, reverse=True)
    print_results(results)
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
